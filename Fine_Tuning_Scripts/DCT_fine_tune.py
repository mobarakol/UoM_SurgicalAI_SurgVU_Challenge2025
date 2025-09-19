import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import random
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW

# from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import logging
import glob
import argparse
from decord import VideoReader, cpu
from typing import List, Tuple, Dict, Any, Optional, Callable, Iterable
from pathlib import Path
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import warnings
from transformers.utils.versions import require_version
from torch.optim import Optimizer
import torch.nn as nn

warnings.filterwarnings("ignore")

from huggingface_hub import login

login(token="hf_your_token")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = -100

os.environ["HF_HOME"] = "Your_path"
os.environ["HF_DATASETS_CACHE"] = "Your_path"
print(f"the home directory for Hugging Face is set to: {os.environ['HF_HOME']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with DoRA")
    # model and data paths
    parser.add_argument(
        "--model_name",
        type=str,
        default="OpenGVLab/InternVL3-2B-Instruct",
        help="pretrained model name or path",
    )
    # hyperparameters
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=6, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-6, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # DCT config
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="reverse_std")
    # other
    parser.add_argument(
        "--save_path",
        type=str,
        default="Your_save_path",
        help="save model path",
    )
    parser.add_argument("--train_data_path", type=str, help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, help="Path to the validation data")
    parser.add_argument(
        "--seed", type=int, default=50, help="random seed for reproducibility"
    )
    return parser.parse_args()


# ---------------------------DCT utilities-------------------------------------------

import math


def dct_matrix(n: int, device=None, dtype=torch.float32):
    """Return an n×n orthogonal DCT-III matrix (as defined in the paper)."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # 1 × n (columns)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # n × 1 (rows)
    Q = torch.cos(((2 * k + 1) * i * math.pi) / (2 * n)) * math.sqrt(2.0 / n)
    Q[0] = Q[0] / math.sqrt(2.0)  # first row scaled by 1/√2 for orthogonality
    return Q  # shape n × n


# ---------------------------Galore part-------------------------------------------


class DCTProjector:
    def __init__(
        self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        # ===== Projection =====
        # We dynamically (re)select columns of the DCT matrix every `update_proj_gap` steps
        if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad,
                self.rank,
                type=(
                    "right"
                    if full_rank_grad.shape[0] >= full_rank_grad.shape[1]
                    else "left"
                ),
            )

        # right / left projection use the same multiplication pattern as original code
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:  # right projection
            low_rank_grad = torch.matmul(
                full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
            )
        else:  # left projection
            low_rank_grad = torch.matmul(
                self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
            )
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
            else:
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
        elif self.proj_type == "reverse_std":
            if (
                low_rank_grad.shape[0] <= low_rank_grad.shape[1]
            ):  # note this is different from std
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
            else:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(
                low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
            )
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(
                self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
            )
        elif self.proj_type == "full":
            full_rank_grad = torch.matmul(
                self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad
            ) @ self.ortho_matrix[1].to(low_rank_grad.device.type)

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type):

        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            # I neded to check the shape of the matrix!!!!!!

        m, n = matrix.shape

        if m >= n:  # right projection
            Q = dct_matrix(n, device=matrix.device, dtype=matrix.dtype)  # n × n
            S = torch.matmul(matrix, Q)  # (m × n)
            scores = S.abs().sum(dim=0)  # n
            idx = torch.topk(scores, rank, largest=True).indices
            Q_r = Q[:, idx]  # n × r
            result = Q_r.t()  # r × n (matches original interface for 'right')
        else:  # left projection
            Q = dct_matrix(m, device=matrix.device, dtype=matrix.dtype)  # m × m
            S = torch.matmul(Q.t(), matrix)  # (m × n)
            scores = S.abs().sum(dim=1)  # m
            idx = torch.topk(scores, rank, largest=True).indices
            A = Q[:, idx]  # m × r
            result = A  # matches 'left' interface (m × r)

        if type == "full":
            # For full projection, build both sides explicitly
            if m >= n:
                A = Q[:, idx]  # n × r (since square case n)
                B = result  # r × n
                result = [A, B]
            else:
                # transpose logic symmetrical
                B_Q = dct_matrix(n, device=matrix.device, dtype=matrix.dtype)  # n×n
                S2 = torch.matmul(matrix, B_Q)  # (m × n)
                idx2 = torch.topk(S2.abs().sum(dim=0), rank, largest=True).indices
                B = B_Q[:, idx2].t()  # r × n
                result = [A, B]

        elif type not in ("left", "right"):
            # Guard against unsupported type values.
            raise ValueError("type should be 'left', 'right' or 'full'")

        # If the original data is not float, convert it back to the original type and device
        def cast_back(x):
            if not float_data:
                return x.to(original_device).type(original_type)
            return x

        if isinstance(result, list):
            return [cast_back(result[0]), cast_back(result[1])]
        else:
            return cast_back(result)


import torch
from tensorly.decomposition import tucker
from tensorly import tenalg


# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors of dimension >2.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    A class that represents a projector for the GaLore algorithm.

    Args:
        rank (int): The rank of the projector.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): The number of iterations between updating the orthogonal matrix. Defaults to 200.
        scale (float, optional): The scaling factor for the projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(
            self.ortho_matrix, self.transformed_low_rank
        )
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes the orthogonal matrix using SVD decomposition.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank_all (int): The desired rank of the decomposition.

        Returns:
            tuple: A tuple containing the core and factors of the orthogonal matrix.
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all)
        return tucker_tensor

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)


class DCT_AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                # If 'dim' is not set in the group, default to 2
                if "dim" not in group:
                    group["dim"] = 2

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group["dim"] <= 2:
                            state["projector"] = DCTProjector(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                            )
                        else:
                            state["projector"] = GaLoreProjectorTensor(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                            )
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


# ------------------------- tools functions -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        mode: str = "train",  # "train" or "inference"
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
            print(
                f"All items in root directory: {[d.name for d in all_dirs if d.is_dir()]}"
            )

        # Discover case folders (non-recursive)
        case_dirs = sorted([p for p in self.root_dir.glob(case_dir_glob) if p.is_dir()])

        print(
            f"Found {len(case_dirs)} case directories: {[d.name for d in case_dirs[:5]]}..."
        )  # Show first 5

        self.samples: List[Dict[str, Any]] = []

        for cdir in case_dirs:
            case_id = cdir.name
            print(f"Processing {case_id}...")

            # Find all video files in the case directory (only files matching pattern *_final.mp4)
            video_files = list(cdir.glob("*_final.mp4"))
            print(
                f"  Found {len(video_files)} video files matching pattern *_final.mp4"
            )

            for video_file in video_files:
                # Video file follows pattern: case{xxx}_{n}_final.mp4
                video_basename = video_file.stem  # e.g., case127_1_final

                # Extract the part without '_final' for QA file matching
                # case127_1_final -> case127_1
                if video_basename.endswith("_final"):
                    qa_basename = video_basename[:-6]  # Remove '_final'
                else:
                    # Fallback if somehow doesn't end with _final
                    qa_basename = video_basename

                # Get corresponding txt file: case127_1_final.mp4 -> qa_case127_1.txt
                txt_file = cdir / f"qa_{qa_basename}.txt"

                if not txt_file.exists():
                    print(
                        f"Warning: No txt file found for {video_file.name} (expected: {txt_file.name})"
                    )
                    continue

                # Read QA pairs from txt file
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    print(f"  Processing {txt_file.name} with {len(lines)} lines")
                    qa_pairs_found = 0

                    for line_idx, line in enumerate(lines):
                        line = line.strip()
                        if not line or "|" not in line:
                            continue

                        # Split question and answer
                        parts = line.split("|", 1)  # Split only on first |
                        if len(parts) != 2:
                            continue

                        question = parts[0].strip()
                        answer = parts[1].strip()

                        if question and answer:
                            self.samples.append(
                                {
                                    "video_path": str(video_file),
                                    "question": question,
                                    "answer": answer,
                                    "case_id": case_id,
                                    "video_id": qa_basename,
                                }
                            )
                            qa_pairs_found += 1

                    print(f"    Found {qa_pairs_found} valid QA pairs")

                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
                    continue

        print(f"Total samples created: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        # Handle single index only - let DataLoader handle batching
        sample = self.samples[idx]

        if self.mode == "train":
            # For training: return dictionary format expected by Hugging Face Trainer
            pixel_values, num_patches_list = load_video_1fps(
                sample["video_path"],
                frame_seconds_list=[1, 10, 20, 30],  # Example frame seconds
                max_num=self.max_num,
                input_size=self.input_size,
            )

            num_patches = pixel_values.size(0)
            if num_patches == 0:
                raise ValueError(f"No frames loaded for video: {sample['video_path']}")

            num_frames = len(num_patches_list)
            image_flags = torch.tensor([1] * num_frames, dtype=torch.long)

            # Return dictionary format for Trainer compatibility
            return {
                "pixel_values": pixel_values,
                "question": sample["question"],
                "answer": sample["answer"],
                "num_patches_list": num_patches_list,
                "image_flags": image_flags,
                "num_frames": num_frames,
            }


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

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
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video_1fps(video_path, frame_seconds_list, input_size=448, max_num=1):
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
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class VideoQACollator:
    """Custom collator for video QA training that works with Hugging Face Trainer"""

    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = (
            'You are a surgical AI assistant in robotic surgery providing assistance and answering surgical trainees\' questions in standard tasks. You handle VQA for these tasks: Suturing; Uterine Horn; Suspensory Ligaments; Rectal Artery/Vein; Skills Application; Range of Motion; Retraction and Collision Avoidance; Other. The surgical tools consist of Large Needle Driver, Monopolar Curved Scissors, Force Bipolar, Clip Applier, Vessel Sealer, Permanent Cautery Hook/Spatula, Stapler, Grasping Retractor, Tip-up Fenestrated Grasper and different types of forceps like Cadiere Forceps, Bipolar Forceps and Prograsp Forceps. You may handle questions like examples below, and you need to follow the Answering rules: Use precise surgical terminology. Keep each answer clinically relevant and one short sentence. Answer should either have "Yes" or "No" at the start, followed by a brief justification or reply with a concise fact. Your answer can\'t be a single "Yes" or "No".\n'
            "Examples:\n"
            'Q: "Are there forceps being used here?"\n'
            'A: "No, forceps are not mentioned."\n'
            'Q: "Is a large needle driver among the listed tools?"\n'
            'A: "No, a large needle driver is not listed."\n'
            'Q: "What type of forceps is mentioned?"\n'
            'A: "The type of forceps mentioned is Cadiere Forceps."\n'
            'Q: "Is a suture required in this surgical step?"\n'
            'A: "Yes, sutures are required."\n'
            'Q: "Was a large needle driver used in this clip?"\n'
            'A: "Yes, a large needle driver was utilized."\n'
            'Q: "What organ is being manipulated?"\n'
            'A: "The organ being manipulated is the uterine horn."\n'
            'Q: "Is a needle driver involved in the procedure?"\n'
            'A: "Yes, a needle driver is involved."\n'
            'Q: "What procedure is this summary describing?"\n'
            'A: "The summary is describing endoscopic or laparoscopic surgery."\n'
            'Q: "What is the purpose of using forceps in this procedure?"\n'
            'A: "The forceps are used for grasping and holding tissues or objects."\n'
            'Q: "Is tissue being cut during this clip?"\n'
            'A: "Yes, tissue is being cut."\n'
            'Q: "Was a large needle driver used during the surgery?"\n'
            'A: "No, a large needle driver was not used."\n'
            "Note that your answer should only be one sentence, it does not need to include 'A: '. The question is: "
        )

    def __call__(self, batch):
        """
        Process a batch of samples from the dataset.
        Each sample is now a dictionary with keys: pixel_values, question, answer, num_patches_list, input_ids
        """
        pixel_values_list = []
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        num_patches_list_batch = []
        image_flags_list = []

        for sample in batch:
            # Extract data from dictionary format
            pixel_values = sample["pixel_values"]
            question = sample["question"]
            answer = sample["answer"]
            num_patches_per_image = sample[
                "num_patches_list"
            ]  # e.g. [1,1,1,1] if max_num=1 and 4 frames
            num_frames = sample["num_frames"]

            # Create video prefix
            video_prefix = "".join(
                [f"Frame{i+1}: <image>\n" for i in range(num_frames)]
            )

            # Format input text
            input_text = video_prefix + self.system_message + question
            target_text = answer
            full_text = input_text + " " + target_text

            # Tokenize input and full text separately to get proper lengths
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            full_encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = full_encoding["input_ids"].squeeze()
            attention_mask = full_encoding["attention_mask"].squeeze()

            # Create labels (mask input part, only compute loss on answer part)
            labels = input_ids.clone()

            # Get actual lengths (non-padding tokens)
            input_actual_length = (
                (input_encoding["input_ids"].squeeze() != self.tokenizer.pad_token_id)
                .sum()
                .item()
            )
            full_actual_length = (input_ids != self.tokenizer.pad_token_id).sum().item()

            # Ensure we have some target tokens
            if input_actual_length >= full_actual_length:
                # If input is as long as or longer than full text, use last few tokens as targets
                target_start = max(
                    0, full_actual_length - 5
                )  # Use last 5 tokens as targets
                labels[:target_start] = IGNORE_INDEX
            else:
                # Normal case: mask input tokens, keep answer tokens
                labels[:input_actual_length] = IGNORE_INDEX

            # Mask padding tokens
            labels[full_actual_length:] = IGNORE_INDEX

            pixel_values_list.append(pixel_values)
            image_flags_list.append(sample["image_flags"])
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            num_patches_list_batch.append(num_patches_per_image)

        batch_pixel_values = torch.cat(pixel_values_list, dim=0)

        batch_image_flags = torch.cat(image_flags_list, dim=0)

        return {
            "pixel_values": batch_pixel_values,
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_mask_list),
            "image_flags": batch_image_flags,
            "num_patches_list": num_patches_list_batch,
        }


# Save model the best model based on validation loss
def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path):

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}")
    return best_loss


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


# ------------------------- Main Training process -------------------------
args = parse_args()


set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

use_cuda = torch.cuda.is_available()

dtype = torch.bfloat16 if use_cuda else torch.float32

model = AutoModel.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="cuda" if use_cuda else None,
)

model.config.pad_token_id = tokenizer.eos_token_id


# 3) Freeze everything EXCEPT the language model's self-attention q, k, v, o projection layers
for p in model.parameters():
    p.requires_grad_(False)

if hasattr(model, "language_model"):
    logging.warning("Model has attribute 'language_model', unfreezing its parameters.")
    print("Model has attribute 'language_model', unfreezing its parameters.")
    for p in model.language_model.parameters():
        p.requires_grad_(True)

galore_params = []
target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"]

for module_name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if not any(target_key in module_name for target_key in target_modules_list):
        continue
    print("enable GaLore for weights in module: ", module_name)
    galore_params.append(module.weight)
id_galore_params = [id(p) for p in galore_params]

regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

param_groups = [
    {"params": regular_params},
    {
        "params": galore_params,
        "rank": args.rank,
        "update_proj_gap": args.update_proj_gap,
        "scale": args.galore_scale,
        "proj_type": args.proj_type,
    },
]

print("regular_params num:", len(param_groups[0]["params"]))
print("galore_params num:", len(param_groups[1]["params"]))
logging.warning(f"regular_params num: {len(param_groups[0]['params'])}")
logging.warning(f"galore_params num: {len(param_groups[1]['params'])}")

galore_ids = set(id(p) for p in galore_params)
for name, param in model.named_parameters():
    if id(param) in galore_ids:
        print(f"{name}: {param.requires_grad}")
        logging.warning(f"{name}: {param.requires_grad}")

# attach DoRA directly to the LM

model.to(device)

optimizer = DCT_AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)


train_root_dir = args.train_data_path
val_root_dir = args.val_data_path

dataset_train = train_dataset = VIVQACaseFolders(root_dir=train_root_dir, mode="train")
dataset_valid = VIVQACaseFolders(root_dir=val_root_dir, mode="train")

train_loader = DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=VideoQACollator(tokenizer, max_length=1024),
)
valid_loader = DataLoader(
    dataset_valid,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=VideoQACollator(tokenizer, max_length=1024),
)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
print("DataLoaders and Optimizer configured!")


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            pixel_values = batch["pixel_values"].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            image_flags = batch["image_flags"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_flags=image_flags,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


import time


def train(
    model, train_loader, valid_loader, optimizer, criterion, num_epochs, save_path
):
    best_val_loss = float("inf")
    print("Start Training!")
    logging.warning("Start Training!")

    total_batches = len(train_loader)
    print(f"total {total_batches} batches per epoch")
    epochs_no_improve = 0
    min_delta = 0.00
    patience = 3
    shrink_factor = 0.8

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        batch_times = []

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            image_flags = batch["image_flags"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_flags=image_flags,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            if batch_idx % 50 == 0:
                print(
                    f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{total_batches} Train Loss: {loss.item():.4f}"
                )
                logging.warning(
                    f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{total_batches} Train Loss: {loss.item():.4f}"
                )
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f} seconds"
                )
                logging.warning(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f} seconds"
                )

        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        avg_train_loss = total_train_loss / total_batches
        avg_val_loss = validate(model, valid_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f} seconds, avg_batch_time: {avg_batch_time:.4f} seconds"
        )
        logging.warning(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f} seconds, avg_batch_time: {avg_batch_time:.4f} seconds, LR: {args.lr}"
        )
        improved = avg_val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_loss = save_best_model(
                model, tokenizer, epoch + 1, best_val_loss, avg_val_loss, save_path
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                adjust_learning_rate(optimizer, shrink_factor)
                epochs_no_improve = 0


# Start training
train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    args.num_epochs,
    args.save_path,
)
