"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import os
import glob
import time
import gc
import typing
import warnings
import json
from datetime import datetime
from collections import defaultdict

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage

import evaluate
import nltk
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
import math
import cv2
from typing import List, Tuple, Dict, Any, Optional
from decord import VideoReader, cpu
from peft import PeftModel



INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Load the model and processor once when the script starts
# This is more efficient than loading it inside the handler
# The model should be loaded from a local path inside the container.
# Ensure the model files are copied to the 'resources' directory in your project.
print("=+= Loading InternVL3-2B-Instruct model from local path...")
try:
    # The path should correspond to where the model is copied in the Dockerfile.
    local_model_path = RESOURCE_PATH / "InternVL3-2B-Instruct/snapshots/f6c7b60375759170fd49f5e9e298e2178485c5ba"
    if not (Path("/opt/app") / local_model_path).exists():
        raise FileNotFoundError(f"Model directory not found at {local_model_path}. Please ensure you have copied the model files into the 'resources/Qwen/Qwen2.5-VL-3B-Instruct' directory.")
    
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModel.from_pretrained(
        local_model_path,
        torch_dtype=dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='cuda' if use_cuda else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    print("=+= Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None



def run():
    # Grand Challenge style interface requires inputs.json
    inputs_json_path = INPUT_PATH / "inputs.json"
    if not inputs_json_path.exists():
        raise FileNotFoundError(
            "Expected /input/inputs.json to be present"
        )

    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()
    print("Inputs: ", interface_key)

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "endoscopic-robotic-surgery-video",
            "visual-context-question",
        ): interf0_handler,
    }.get(interface_key)

    if handler is None:
        raise ValueError(f"Unsupported input interface: {interface_key}")

    # Call the handler
    return handler()


def interf0_handler():
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not loaded. Cannot run inference.")

    # Resolve inputs via inputs.json to support both /input and /input/interf0 layouts
    video_path = resolve_input_path_by_slug("endoscopic-robotic-surgery-video")
    try:
        input_visual_context_question = get_input_value_by_slug("visual-context-question")
    except Exception:
        # Fallback: try loading from file if provided only via relative_path
        question_path = resolve_input_path_by_slug("visual-context-question")
        input_visual_context_question = load_json_file(location=question_path)

    print('Video path: ', str(video_path))
    print('Question: ', json.dumps(input_visual_context_question, indent=4))

    # Prepare the input for the Qwen-VL model
    if isinstance(input_visual_context_question, dict):
        question = input_visual_context_question.get("question", "")
    else:
        question = str(input_visual_context_question)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    
    # Show detailed GPU information
    _show_torch_cuda_info()
    
    system_message = ("You are a surgical AI assistant in robotic surgery providing assistance and answering surgical trainees' questions in standard tasks. You handle VQA for these tasks: Suturing; Uterine Horn; Suspensory Ligaments; Rectal Artery/Vein; Skills Application; Range of Motion; Retraction and Collision Avoidance; Other. The surgical tools consist of Large Needle Driver, Monopolar Curved Scissors, Force Bipolar, Clip Applier, Vessel Sealer, Permanent Cautery Hook/Spatula, Stapler, Grasping Retractor, Tip-up Fenestrated Grasper and different types of forceps like Cadiere Forceps, Bipolar Forceps and Prograsp Forceps. You may handle questions like examples below, and you need to follow the Answering rules: Use precise surgical terminology. Keep each answer clinically relevant and one short sentence. Answer should either have \"Yes\" or \"No\" at the start, followed by a brief justification or reply with a concise fact. Your answer can't be a single \"Yes\" or \"No\".\n"
                        "Examples:\n"
                        "Q: \"Are there forceps being used here?\"\n"
                        "A: \"No, forceps are not mentioned.\"\n"
                        "Q: \"Is a large needle driver among the listed tools?\"\n"
                        "A: \"No, a large needle driver is not listed.\"\n"
                        "Q: \"What type of forceps is mentioned?\"\n"
                        "A: \"The type of forceps mentioned is Cadiere Forceps.\"\n"
                        "Q: \"Is a suture required in this surgical step?\"\n"
                        "A: \"Yes, sutures are required.\"\n"
                        "Q: \"Was a large needle driver used in this clip?\"\n"
                        "A: \"Yes, a large needle driver was utilized.\"\n"
                        "Q: \"What organ is being manipulated?\"\n"
                        "A: \"The organ being manipulated is the uterine horn.\"\n"
                        "Q: \"Is a needle driver involved in the procedure?\"\n"
                        "A: \"Yes, a needle driver is involved.\"\n"
                        "Q: \"What procedure is this summary describing?\"\n"
                        "A: \"The summary is describing endoscopic or laparoscopic surgery.\"\n"
                        "Q: \"What is the purpose of using forceps in this procedure?\"\n"
                        "A: \"The forceps are used for grasping and holding tissues or objects.\"\n"
                        "Q: \"Is tissue being cut during this clip?\"\n"
                        "A: \"Yes, tissue is being cut.\"\n"
                        "Q: \"Was a large needle driver used during the surgery?\"\n"
                        "A: \"No, a large needle driver was not used.\"\n"
                        "Note that your answer should only be one sentence, it does not need to include \'A: \'. The question is: ")
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    pixel_values, num_patches_list = load_video(str(video_path), [1, 10, 20, 30], max_num=1)
    if use_cuda:
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    print("use cuda", use_cuda)

    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + system_message + question
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    output_visual_context_response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    
    print(f'User: {question}\nresponse: {output_visual_context_response}')

    write_json_file(
        location=OUTPUT_PATH / "visual-context-response.json",
        content=output_visual_context_response,
    )
    print('output saved to  ', OUTPUT_PATH)

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('These are the inputs:' , inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def resolve_input_path_by_slug(slug: str) -> Path:
    # Returns the absolute path to the input file for the given interface slug
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    for sv in inputs:
        if sv["interface"]["slug"] == slug:
            rel = sv["interface"].get("relative_path")
            if rel:
                return INPUT_PATH / rel
    raise FileNotFoundError(f"Could not resolve path for slug '{slug}' from inputs.json")


def get_input_value_by_slug(slug: str):
    # Returns the direct value for Value-like inputs, or reads JSON from relative_path
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    for sv in inputs:
        if sv["interface"]["slug"] == slug:
            # First try to read from relative_path file if it exists
            rel = sv["interface"].get("relative_path")
            if rel:
                file_path = INPUT_PATH / rel
                if file_path.exists():
                    return load_json_file(location=file_path)
            
            # Fallback to direct value if file doesn't exist or no relative_path
            if sv.get("value") is not None:
                return sv["value"]
                
    raise KeyError(f"Could not find input value for slug '{slug}'")


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


# Note to the developer:
#   the following function is very generic and should likely
#   be adopted to something more specific for your algorithm/challenge
def load_file(*, location):
    # Reads the content of a file
    with open(location) as f:
        return f.read()


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
    
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# def load_image(image_file, input_size=448, max_num=12):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

def load_image(image_file, input_size=448, max_num=12):
    image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, frame_seconds_list, input_size=448, max_num=1):
    """
    Load video frames at specified second positions (1fps sampling).
    
    Args:
        video_path: Path to video file
        frame_seconds_list: List of second positions to extract frames from (1-based, e.g., [1, 5, 9, 13])
        input_size: Input image size for preprocessing
        max_num: Maximum number of image patches
    
    Returns:
        pixel_values: Concatenated tensor of all frame patches
        num_patches_list: List of patch counts for each frame
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    max_frame = len(vr) - 1

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    
    for second in frame_seconds_list:
        # Convert second position to frame index (1-based second to 0-based frame)
        # For 1fps equivalent: take frame at (second-1) * fps
        frame_index = int((second - 1) * fps)
        
        # Clamp frame index to valid range
        frame_index = max(0, min(frame_index, max_frame))
        
        # Extract and process the frame
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    import os
    import glob
    
    # First, try to find checkpoint-* directories
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if checkpoint_dirs:
        # Sort by checkpoint number and get the latest one
        checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[1]))
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint)}")
        return latest_checkpoint
    
    # If no checkpoint-* directories, check if the directory itself contains model files
    peft_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
    if any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in peft_files):
        print(f"Using checkpoint directory directly: {checkpoint_dir}")
        return checkpoint_dir
    
    # List contents to help debug
    if os.path.exists(checkpoint_dir):
        contents = os.listdir(checkpoint_dir)
        print(f"Directory contents: {contents}")
    else:
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    return None


if __name__ == "__main__":
    raise SystemExit(run())
