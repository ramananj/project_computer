#!/usr/bin/env python3
import socket
import struct
import threading
import cv2
import numpy as np
import torch
import time
import logging
from flask import Flask, Response
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)
TCP_PORT = 9000
UDP_PORT = 10000
CLIENT_ADDR = None  # Will store (ip, port) of RPi sender for UDP feedback
FRAME_BUFFER = [b'']

# Load Grounding DINO model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "groundingdino_weights/groundingdino_swint_ogc.pth"
)
model.to(DEVICE).eval()

CAPTION = (
    "hidden ball . small orange soccer ball . orange ball . toy ball . moving ball . blurry ball . ball moving"
    " . blue line wall . blue outline"
)
BALL_TERMS = {"hidden ball", "small orange soccer ball", "orange ball", "toy ball", "moving ball", "blurry ball", "ball moving"}
WALL_TERMS = {"blue line wall", "blue outline"}

# Preprocessing helper

def prepare_image(image_array: np.ndarray) -> torch.Tensor:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_array, None)
    return image_transformed


def cxcywh_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return x1, y1, x2, y2


def compute_ioa(ball_box, wall_box, img_w, img_h):
    bx1, by1, bx2, by2 = cxcywh_to_xyxy(ball_box, img_w, img_h)
    wx1, wy1, wx2, wy2 = cxcywh_to_xyxy(wall_box, img_w, img_h)
    ix1, iy1 = max(bx1, wx1), max(by1, wy1)
    ix2, iy2 = min(bx2, wx2), min(by2, wy2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter_area = iw * ih
    ball_area = max(0, bx2-bx1) * max(0, by2-by1)
    return inter_area / ball_area if ball_area>0 else 0.0


def run_dino_inference(frame_bgr):
    global CLIENT_ADDR
    start = time.perf_counter()
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    boxes, logits, phrases = predict(
        model,
        prepare_image(image_rgb),
        CAPTION,
        box_threshold=0.2,
        text_threshold=0.2
    )

    ball_idxs = [i for i,p in enumerate(phrases) if any(t in p for t in BALL_TERMS)]
    wall_idxs = [i for i,p in enumerate(phrases) if any(t in p for t in WALL_TERMS)]
    best_ball = max(ball_idxs, key=lambda i: logits[i]) if ball_idxs else None
    best_wall = max(wall_idxs, key=lambda i: logits[i]) if wall_idxs else None

    best_boxes, best_logits, best_phrases = [], [], []
    if best_ball is not None and logits[best_ball]>0.29:
        best_boxes.append(boxes[best_ball]); best_logits.append(logits[best_ball]); best_phrases.append(phrases[best_ball])
    if best_wall is not None and logits[best_wall]>0.35:
        best_boxes.append(boxes[best_wall]); best_logits.append(logits[best_wall]); best_phrases.append(phrases[best_wall])

    ioa = None
    if len(best_boxes)==2:
        ioa = compute_ioa(best_boxes[0], best_boxes[1], 640, 480)
    best_boxes = torch.stack(best_boxes) if best_boxes else torch.empty((0,4))


    # annotated = None
    annotated = annotate(image_source=image_rgb, boxes=best_boxes, logits=best_logits, phrases=best_phrases)
    if ioa and ioa>0.3:
        cv2.putText(annotated, "GOAL!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        # Send UDP event back to RPi
        if CLIENT_ADDR:
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.sendto(b"GOAL", (CLIENT_ADDR[0], UDP_PORT))

    end = time.perf_counter()
    logging.info(f"[Inference] Latency: {(end-start)*1000:.2f} ms")
    return annotated


def tcp_listener():
    global CLIENT_ADDR
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("", TCP_PORT))
    server.listen(1)
    while True:
        conn, addr = server.accept()
        CLIENT_ADDR = addr  # store IP for UDP feedback
        data = b''
        while True:
            try:
                while len(data)<4:
                    more = conn.recv(4-len(data))
                    if not more: raise ConnectionResetError
                    data += more
                msg_len = struct.unpack('>L', data[:4])[0]
                data = data[4:]
                while len(data)<msg_len:
                    more = conn.recv(msg_len-len(data))
                    if not more: raise ConnectionResetError
                    data += more
                jpeg = data[:msg_len]; data = data[msg_len:]
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    annotated = run_dino_inference(frame)
                    _, out_jpeg = cv2.imencode('.jpg', annotated)
                    FRAME_BUFFER[0] = out_jpeg.tobytes()
            except (ConnectionResetError, BrokenPipeError):
                break
        conn.close()


def mjpeg_stream():
    while True:
        if FRAME_BUFFER[0]:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + FRAME_BUFFER[0] +
                b'\r\n'
            )

@app.route('/')
def stream():
    return Response(mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    threading.Thread(target=tcp_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, threaded=True)