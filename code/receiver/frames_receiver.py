from flask import Flask, Response
import threading
import socket
import struct
import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)
PORT = 9000
FRAME_BUFFER = [b'']

# === Load model ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[GroundingDINO] Using device: {DEVICE}")
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "groundingdino_weights/groundingdino_swint_ogc.pth")
model.to(DEVICE).eval()

CAPTION = (
    "hidden ball . small orange soccer ball . orange ball . toy ball . moving ball . blurry ball . ball moving"
    " . blue line wall . blue outline"
)
BALL_TERMS = {
    "hidden ball",
    "small orange soccer ball",
    "orange ball",
    "toy ball",
    "moving ball",
    "blurry ball",
    "ball moving"
}
WALL_TERMS = {"blue line wall", "blue outline"}
BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2

# === Preprocessing ===
def prepare_image(image_array: np.ndarray) -> torch.Tensor:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_array, None)
    return image_transformed

def cxcywh_to_xyxy(box, img_w, img_h):
    """
    Convert a normalized (cx,cy,w,h) box → absolute (x1,y1,x2,y2) in pixels.
    """
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return x1, y1, x2, y2

def compute_ioa(ball_box, wall_box, img_w, img_h):
    # get pixel coords
    bx1, by1, bx2, by2 = cxcywh_to_xyxy(ball_box, img_w, img_h)
    wx1, wy1, wx2, wy2 = cxcywh_to_xyxy(wall_box, img_w, img_h)

    # intersection rectangle
    ix1, iy1 = max(bx1, wx1), max(by1, wy1)
    ix2, iy2 = min(bx2, wx2), min(by2, wy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih

    # ball area
    ball_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    return inter_area / ball_area if ball_area > 0 else 0.0

def run_dino_inference(frame_bgr):
    start = time.perf_counter()
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    boxes, logits, phrases = predict(
        model,
        prepare_image(image_rgb),
        CAPTION,
        box_threshold=0.2,
        text_threshold=0.2
    )
    
    # 4)  Partition detections by which keyword they matched
    ball_idxs = [i for i,p in enumerate(phrases) if any(t in p for t in BALL_TERMS)]
    wall_idxs = [i for i,p in enumerate(phrases) if any(t in p for t in WALL_TERMS)]

    best_ball_idx = max(ball_idxs, key=lambda i: logits[i]) if ball_idxs else None
    best_wall_idx = max(wall_idxs, key=lambda i: logits[i]) if wall_idxs else None


    # 2) build the lists for annotate()
    best_boxes   = []
    best_logits  = []
    best_phrases = []
    
    # print(logits[best_ball_idx])
    if best_ball_idx is not None and logits[best_ball_idx] > 0.29:
        best_boxes  .append(boxes[best_ball_idx])
        best_logits .append(logits[best_ball_idx])
        best_phrases.append(phrases[best_ball_idx])
    
    if best_wall_idx is not None and logits[best_wall_idx] > 0.35:
        best_boxes  .append(boxes[best_wall_idx])
        best_logits .append(logits[best_wall_idx])
        best_phrases.append(phrases[best_wall_idx])

    ioa = None
    if len(best_boxes) == 0:
        best_boxes = torch.empty((0, 4), dtype=torch.float32)
    else:
        if len(best_boxes) == 2:
            ioa = compute_ioa(best_boxes[0], best_boxes[1], 640, 480)
        best_boxes = torch.stack(best_boxes)

    annotated = annotate(image_source=image_rgb, boxes=best_boxes, logits=best_logits, phrases=best_phrases)
    if ioa is not None and ioa > 0.3:
        cv2.putText(
            annotated, 
            "GOAL!", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 0), 
            3
        )
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    logging.info(f"[Inference] Grounding DINO latency: {latency_ms:.2f} ms")
    return annotated

# === TCP Listener Thread ===
def tcp_listener():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', PORT))
    server.listen(1)
    print(f"[TCP] Listening on port {PORT}...")

    while True:
        conn, addr = server.accept()
        print(f"[TCP] Connected from {addr}")
        data = b''
        while True:
            try:
                while len(data) < 4:
                    more = conn.recv(4 - len(data))
                    if not more:
                        raise ConnectionResetError
                    data += more
                msg_len = struct.unpack('>L', data[:4])[0]
                data = data[4:]

                while len(data) < msg_len:
                    more = conn.recv(msg_len - len(data))
                    if not more:
                        raise ConnectionResetError
                    data += more

                jpeg_data = data[:msg_len]
                data = data[msg_len:]

                img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                annotated = run_dino_inference(frame)
                _, result_jpeg = cv2.imencode('.jpg', annotated)
                FRAME_BUFFER[0] = result_jpeg.tobytes()

            except (ConnectionResetError, BrokenPipeError):
                print("[TCP] Disconnected.")
                break
        conn.close()

# === MJPEG stream route ===
def mjpeg_stream():
    while True:
        if FRAME_BUFFER[0]:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   FRAME_BUFFER[0] + b'\r\n')

@app.route('/')
def stream():
    return Response(mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=tcp_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, threaded=True)
