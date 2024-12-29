import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

args = parser.parse_args()

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, load_in_4bit=True
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

prompt = "USER: <image>\nWrite a description for the photo in one sentence. ASSISTANT:"

img_path = args.img_path
captions = {}

start_time = time.time()

total_images = len(os.listdir(img_path))

i = 0
for image_filename in os.listdir(img_path):
    image_path = os.path.join(img_path, image_filename)
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float32
        )
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        description = processor.decode(output[0][2:], skip_special_tokens=True)
        captions[image_filename.replace(".jpg", "")] = description.split("ASSISTANT:")[
            -1
        ].strip()
    except Exception as e:
        print(f"Error processing {image_filename}: {e}")

    i += 1
    progress = (i / total_images) * 100
    print(f"Progress: {progress:.2f}% ({i}/{total_images})", end="\r")

end_time = time.time()
total_time = end_time - start_time
print(f"\nProcessing completed in {total_time:.2f} seconds.")

with open(args.output_path, "w") as f:
    json.dump(captions, f)
