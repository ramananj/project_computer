import time
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# default processer
processor = AutoProcessor.from_pretrained(model_name)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)
total_pixels=20480 * 28 * 28
min_pixels=16 * 28 * 28
prompt = """
Are there people playing soccer? If so, was there a ball that was kicked into the goal demarcated on the wall by the blue square?
Please reply with a return format of:
Soccer ball detected: True [Can be True of soccer ball is detected in the scene anywhere, else False]
People actively playing soccer: False [Can be true only if soccer ball detected is True and there are people playing within the clip sequence, else False]
Goal detected: False [Can be true only if soccer ball detected is True, and people actively playing soccer is True, and the ball is detected hitting the wall within the demarcated area within the clip sequence, else False]
"""
start = time.time()
for c1 in tqdm(range(3)):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": "file:///workspace/data/soccer_short_goal_clip.mp4",
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "fps": 12,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video fps:", fps_inputs)
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    duration_seconds = num_frames / fps_inputs[0]
    print(f"⏱️ Video duration after processing: {duration_seconds:.2f} seconds ({num_frames} frames at {fps_inputs[0]} fps)")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
end = time.time()
print(f"total time: {end - start:.4f} seconds")
print(output_text)
