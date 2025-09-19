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
from peft import PeftModel
import numpy as np
import re
import math
import cv2
from typing import List, Tuple, Dict, Any, Optional
from decord import VideoReader, cpu



from huggingface_hub import login
# NOTE: Consider using an environment variable for tokens instead of hardcoding.
HUGGINGFACE_TOKEN = "Your_token"
# Authenticate with Hugging Face
login(HUGGINGFACE_TOKEN)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


"""
This script performs single-image, single-round inference over a dataset without any system messages.
"""



# ---- dataset over case folders ----

class VIVQACaseFolders(Dataset):
    """
    Root directory contains multiple case folders (e.g., case_000, case_001, ...).
    Each case folder contains:
      - Multiple video files (.mp4) with format: case{xxx}_{n}_final.mp4
      - Corresponding QA text files (.txt) with format: qa_case{xxx}_{n}.txt (without _final suffix)
      - Each txt file contains multiple QA pairs separated by newlines
      - QA format: "Question | Answer"

    For training, each video-QA pair combination becomes one sample:
      - pixel_values (video frames as tensors)
      - question as string  
      - answer as string

    Output per sample for training:
      (pixel_values, question_str, answer_str, num_patches_list)
    """

    def __init__(
        self,
        root_dir: str,
        case_dir_glob: str = "case_*",
        num_segments: int = 8,
        max_num: int = 1,
        input_size: int = 448,
        mode: str = "train"  # "train" or "inference"
    ):
        self.root_dir = Path(root_dir)
        self.num_segments = num_segments
        self.max_num = max_num
        self.input_size = input_size
        self.mode = mode

        print(f"Dataset root directory: {self.root_dir}")
        print(f"Looking for case directories with pattern: {case_dir_glob}")
        print(f"Root directory exists: {self.root_dir.exists()}")
        
        if self.root_dir.exists():
            all_dirs = list(self.root_dir.glob("*"))
            print(f"All items in root directory: {[d.name for d in all_dirs if d.is_dir()]}")

        # Discover case folders (non-recursive)
        case_dirs = sorted(
            [p for p in self.root_dir.glob(case_dir_glob) if p.is_dir()]
        )
        
        print(f"Found {len(case_dirs)} case directories: {[d.name for d in case_dirs[:5]]}...")  # Show first 5

        self.samples: List[Dict[str, Any]] = []
        
        for cdir in case_dirs:
            case_id = cdir.name
            print(f"Processing {case_id}...")
            
            # Find all video files in the case directory (only files matching pattern *_final.mp4)
            video_files = list(cdir.glob("*_final.mp4"))
            print(f"  Found {len(video_files)} video files matching pattern *_final.mp4")
            
            for video_file in video_files:
                # Video file follows pattern: case{xxx}_{n}_final.mp4
                video_basename = video_file.stem  # e.g., case127_1_final
                
                # Extract the part without '_final' for QA file matching
                # case127_1_final -> case127_1
                if video_basename.endswith('_final'):
                    qa_basename = video_basename[:-6]  # Remove '_final'
                else:
                    # Fallback if somehow doesn't end with _final
                    qa_basename = video_basename
                
                # Get corresponding txt file: case127_1_final.mp4 -> qa_case127_1.txt
                txt_file = cdir / f"qa_{qa_basename}.txt"
                
                if not txt_file.exists():
                    print(f"Warning: No txt file found for {video_file.name} (expected: {txt_file.name})")
                    continue
                
                # Read QA pairs from txt file
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    print(f"  Processing {txt_file.name} with {len(lines)} lines")
                    qa_pairs_found = 0
                    
                    for line_idx, line in enumerate(lines):
                        line = line.strip()
                        if not line or '|' not in line:
                            continue
                        
                        # Split question and answer
                        parts = line.split('|', 1)  # Split only on first |
                        if len(parts) != 2:
                            continue
                        
                        question = parts[0].strip()
                        answer = parts[1].strip()
                        
                        if not question or not answer:
                            continue
                        
                        qa_pairs_found += 1
                        
                        # Create a sample for each video-QA pair
                        self.samples.append({
                            "case_id": case_id,
                            "video_path": str(video_file),
                            "video_basename": video_basename,
                            "question": question,
                            "answer": answer,
                            "qa_index": line_idx
                        })
                    
                    print(f"    Created {qa_pairs_found} QA pairs from {txt_file.name}")
                        
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
                    continue
        
        print(f"Total samples created: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
            # For inference: return video path, question, answer (original behavior)
        return sample["video_path"], sample["question"], sample["answer"]


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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_videos_Sequence(video_path, frame_seconds_list, input_size=448, max_num=1):
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


def generate_text_from_sample(model, tokenizer, sample):
    """Generate text from a formatted sample"""
    # Prepare image tiles and move to device
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    system_message = ("You are a surgical AI assistant in robotic surgery providing assistance and answering surgical trainees' questions in standard tasks. You handle VQA for these tasks: Suturing; Uterine Horn; Suspensory Ligaments; Rectal Artery/Vein; Skills Application; Range of Motion; Retraction and Collision Avoidance; Other. The surgical tools consist of Large Needle Driver, Monopolar Curved Scissors, Force Bipolar, Clip Applier, Vessel Sealer, Permanent Cautery Hook/Spatula, Stapler, Grasping Retractor, Tip-up Fenestrated Grasper and different types of forceps like Cadiere Forceps, Bipolar Forceps and Prograsp Forceps. You may handle questions like examples below, and you need to follow the Answering rules: Use precise surgical terminology. Keep each answer clinically relevant and one short sentence. Answer should either have \"Yes\" or \"No\" at the start, followed by a brief justification or reply with a concise fact.\n"
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
    pixel_values, num_patches_list = load_videos_Sequence(sample[0], frame_seconds_list=[1, 10, 20, 30], max_num=1)
    if use_cuda:
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + system_message + sample[1]
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    return response


    return output_text[0]

def extract_time(text):
    """Extract time values from text and convert to minutes"""
    total_minutes = 0.0
    pattern = r"([-+]?[0-9]*\.?[0-9]+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
    
    for val, unit in re.findall(pattern, text, re.IGNORECASE):
        v = float(val)
        u = unit.lower()
        if u.startswith("h"):
            total_minutes += v * 60
        elif u.startswith("m"):
            total_minutes += v
        elif u.startswith("s"):
            total_minutes += v / 60

    if total_minutes > 0:
        return total_minutes

    lone_num = re.search(r"([-+]?[0-9]*\.?[0-9]+)\b", text)
    if lone_num:
        return float(lone_num.group(1))

    return None

def get_nlp_metrics(references, hypotheses):
    """Compute NLP evaluation metrics"""
    bleu = evaluate.load("bleu", keep_in_memory=True)
    rouge = evaluate.load("rouge", keep_in_memory=True)
    meteor = evaluate.load("meteor", keep_in_memory=True)
    
    results_bleu = bleu.compute(predictions=hypotheses, references=references)
    results_rouge = rouge.compute(predictions=hypotheses, references=references)
    results_meteor = meteor.compute(predictions=hypotheses, references=references)
    
    return results_bleu, results_rouge, results_meteor

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

def clear_memory():
    """Clear GPU memory"""
    for var in ['inputs', 'model', 'processor']:
        if var in globals():
            del globals()[var]
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def save_results_to_file(results, filename, output_dir="system_message_results"):
    """Save results to JSON file in specified directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine directory and filename
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")

def run_single_test(model, tokenizer, dataset, max_samples=None):
    
    all_predictions = []
    all_references = []
    sample_details = []
    
    # Use full dataset if max_samples is None
    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Processing {total_samples} samples...")
    model.eval()
    
    printed_samples = 0
    num_samples_to_print = 10
    
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            
            # Generate response
            start_time = time.time()
            prediction = generate_text_from_sample(model, tokenizer, sample)
            inference_time = time.time() - start_time
            
            all_predictions.append(prediction)
            all_references.append(sample[2])  # sample[2] is the reference answer

            # Store sample details
            sample_details.append({
                "index": idx,
                "video_path": sample[0],  # sample[0] is now video path string
                "question": sample[1],
                "reference": sample[2],
                "prediction": prediction,
                "inference_time": inference_time
            })
            
            # Print first 10 samples
            if printed_samples < num_samples_to_print:
                print(f"\n--- Sample {idx + 1} ---")
                print(f"Video Path: {sample[0]}")
                print(f"Question: {sample[1]}")
                print(f"Reference: {sample[2]}")
                print(f"Prediction: {prediction}")
                print(f"Inference Time: {inference_time:.3f}s")
                print("-------------------")
                printed_samples += 1
            
            # Print progress
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{total_samples} samples...")

    # Compute overall metrics
    results_bleu, results_rouge, results_meteor = get_nlp_metrics(all_references, all_predictions)
    
    overall_metrics = {
        "bleu_1": results_bleu['precisions'][0],
        "bleu_2": results_bleu['precisions'][1],
        "bleu_3": results_bleu['precisions'][2],
        "bleu_4": results_bleu['precisions'][3],
        "bleu_overall": results_bleu['bleu'],
        "rouge1": results_rouge['rouge1'],
        "rouge2": results_rouge['rouge2'],
        "rougeL": results_rouge['rougeL'],
        "rougeLsum": results_rouge['rougeLsum'],
        "meteor": results_meteor['meteor']
    }
    
    # Categorical evaluation
    cat_result_dict = defaultdict(lambda: {"references": [], "hypotheses": []})


    categorical_metrics = {}
    for category, data in cat_result_dict.items():
        refs_cat = data["references"]
        hyps_cat = data["hypotheses"]
        results_bleu, results_rouge, results_meteor = get_nlp_metrics(refs_cat, hyps_cat)
        
        categorical_metrics[category] = {
            "count": len(refs_cat),
            "bleu": results_bleu['bleu'],
            "rouge1": results_rouge['rouge1'],
            "rougeL": results_rouge['rougeL'],
            "meteor": results_meteor['meteor']
        }

    # Time category MSE
    time_mse = None
    if "time" in cat_result_dict:
        time_refs = cat_result_dict["time"]["references"]
        time_hyps = cat_result_dict["time"]["hypotheses"]
        
        ref_times = [extract_time(ref) for ref in time_refs]
        hyp_times = [extract_time(hyp) for hyp in time_hyps]
        
        valid_pairs = [(r, h) for r, h in zip(ref_times, hyp_times) if r is not None and h is not None]
        
        if valid_pairs:
            squared_errors = [(r - h) ** 2 for r, h in valid_pairs]
            time_mse = np.mean(squared_errors)

    # Calculate average inference time
    avg_inference_time = np.mean([detail["inference_time"] for detail in sample_details])
    

    # Save references and predictions to text files
    output_dir = "results"  # or use output_directory variable
    os.makedirs(output_dir, exist_ok=True)
    ref_txt_path = os.path.join(output_dir, "DCT_references.txt")
    hyp_txt_path = os.path.join(output_dir, "DCT_predictions.txt")
    combined_txt_path = os.path.join(output_dir, "DCT_combined.txt")
    with open(ref_txt_path, "w", encoding="utf-8") as f:
        for ref in all_references:
            f.write(ref + "\n")
    with open(hyp_txt_path, "w", encoding="utf-8") as f:
        for hyp in all_predictions:
            f.write(hyp + "\n")
    # Write combined file: video path, question, reference, prediction, separated by delimiter
    with open(combined_txt_path, "w", encoding="utf-8") as f:
        for detail in sample_details:
            f.write("=== SAMPLE START ===\n")
            f.write(f"Video Path: {detail['video_path']}\n")
            f.write(f"Question: {detail['question']}\n")
            f.write(f"Reference: {detail['reference']}\n")
            f.write(f"Prediction: {detail['prediction']}\n")
            f.write("=== SAMPLE END ===\n\n")
    print(f"Saved references to {ref_txt_path}")
    print(f"Saved predictions to {hyp_txt_path}")
    print(f"Saved combined sample details to {combined_txt_path}")

    return {
        "total_samples": total_samples,
        "overall_metrics": overall_metrics,
        "categorical_metrics": categorical_metrics,
        "time_mse": time_mse,
        "avg_inference_time": avg_inference_time,
        "sample_details": sample_details[:10],  # Save first 10 samples for inspection
        "timestamp": datetime.now().isoformat()
    }

def print_results_summary(result):
    """Print a summary of results for the current run (no system message)."""
    print(f"\n=== Results Summary ===")
    metrics = result['overall_metrics']
    print(f"Overall Metrics:")
    print(f"  BLEU-1: {metrics['bleu_1']:.6f}, BLEU-2: {metrics['bleu_2']:.6f}, "
          f"BLEU-3: {metrics['bleu_3']:.6f}, BLEU-4: {metrics['bleu_4']:.6f}")
    print(f"  Overall BLEU: {metrics['bleu_overall']:.6f}")
    print(f"  Rouge1: {metrics['rouge1']:.6f}, Rouge2: {metrics['rouge2']:.6f}, "
          f"RougeL: {metrics['rougeL']:.6f}")
    print(f"  Meteor: {metrics['meteor']:.6f}")
    print(f"  Avg Inference Time: {result['avg_inference_time']:.3f}s")
    
    if result['time_mse'] is not None:
        print(f"  Time MSE: {result['time_mse']:.6f}")
    
    print(f"\nCategorical Results:")
    for category, cat_metrics in result['categorical_metrics'].items():
        print(f"  {category} (n={cat_metrics['count']}): "
              f"BLEU={cat_metrics['bleu']:.4f}, "
              f"Rouge1={cat_metrics['rouge1']:.4f}, "
              f"Meteor={cat_metrics['meteor']:.4f}")
    
    # Print some sample predictions
    print(f"\nSample Predictions:")
    for i, sample in enumerate(result['sample_details'][:3]):  # Show first 3 samples
        print(f"  Sample {i+1}:")
        print(f"    Video Path: {sample['video_path']}")
        print(f"    Question: {sample['question']}")
        print(f"    Reference: {sample['reference']}")
        print(f"    Prediction: {sample['prediction']}")


# Main execution
def main():
    
    root_dir = "Your_dataset_path"  # Update to your dataset path
    # Initialize dataset with video paths instead of extracted frames
    dataset = VIVQACaseFolders(root_dir=root_dir)

    # Load model and tokenizer
    # model_id = "OpenGVLab/InternVL3-2B-Instruct"
    checkpoint_path = "Your_checkpoint_path"
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    
    print(f"Loading base model: {checkpoint_path}")
    model = AutoModel.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='cuda' if use_cuda else None,
    )
    
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Configuration
    max_samples = None  # Set to None for full dataset, or specify number for testing
    output_directory = "Your_output_directory"  # Updated to reflect DoRA model

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    print(f"Results will be saved to: {output_directory}")
    
    # Run test (no system messages)
    print("\n" + "=" * 60)
    print("Starting test with DoRA fine-tuned model")
    print("=" * 60)
    try:
        result = run_single_test(model, tokenizer, dataset, max_samples)
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    print_results_summary(result)
    
    # Save individual result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results_to_file(result, f"internVL2B_DCT_results_{timestamp}.json", output_directory)
        
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Also save a wrapper object for compatibility
    save_results_to_file([result], f"internVL2B_DCT_all_results_{timestamp}.json", output_directory)
    
    print(f"\nTesting completed! Results saved to files.")
    print(f"Total runs saved: 1")

if __name__ == "__main__":
    main()