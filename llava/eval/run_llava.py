import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    POINTCLOUD_TOKEN_INDEX,
    DEFAULT_POINTCLOUD_TOKEN,
    DEFAULT_PC_START_TOKEN,
    DEFAULT_PC_END_TOKEN,
    POINTCLOUD_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    tokenizer_pointcloud_token,
    tokenizer_multimodal_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def pointcloud_parser(args):
    if hasattr(args, 'pointcloud_file') and args.pointcloud_file:
        out = args.pointcloud_file.split(args.sep)
        return out
    return []


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_pointcloud(pointcloud_file):
    """Load pointcloud from .pt file. Expected shape: (64, 382)"""
    try:
        pointcloud = torch.load(pointcloud_file, map_location='cpu')
        # Ensure the pointcloud has the expected shape
        if pointcloud.dim() == 2 and pointcloud.shape == (64, 382):
            return pointcloud
        else:
            raise ValueError(f"Expected pointcloud shape (64, 382), got {pointcloud.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load pointcloud from {pointcloud_file}: {e}")


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_pointclouds(pointcloud_files):
    out = []
    for pointcloud_file in pointcloud_files:
        pointcloud = load_pointcloud(pointcloud_file)
        out.append(pointcloud)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    
    # Handle pointcloud tokens
    has_pointclouds = hasattr(args, 'pointcloud_file') and args.pointcloud_file
    pointcloud_token_se = DEFAULT_PC_START_TOKEN + DEFAULT_POINTCLOUD_TOKEN + DEFAULT_PC_END_TOKEN
    
    if POINTCLOUD_PLACEHOLDER in qs:
        if hasattr(model.config, 'mm_use_pc_start_end') and model.config.mm_use_pc_start_end:
            qs = re.sub(POINTCLOUD_PLACEHOLDER, pointcloud_token_se, qs)
        else:
            qs = re.sub(POINTCLOUD_PLACEHOLDER, DEFAULT_POINTCLOUD_TOKEN, qs)
    elif has_pointclouds:
        if hasattr(model.config, 'mm_use_pc_start_end') and model.config.mm_use_pc_start_end:
            qs = pointcloud_token_se + "\n" + qs
        else:
            qs = DEFAULT_POINTCLOUD_TOKEN + "\n" + qs
    
    # Handle image tokens
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Load images
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    # Load pointclouds
    pointcloud_files = pointcloud_parser(args)
    pointclouds = None
    if pointcloud_files:
        pointclouds = load_pointclouds(pointcloud_files)
        # Stack into batch tensor: (batch_size, 64, 382)
        pointclouds = torch.stack(pointclouds).to(model.device, dtype=torch.float16)

    # Tokenize with appropriate tokenizer based on modalities present
    if pointcloud_files and image_files:
        # Both pointcloud and image tokens present
        input_ids = (
            tokenizer_multimodal_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, POINTCLOUD_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
    elif pointcloud_files:
        # Only pointcloud tokens present
        input_ids = (
            tokenizer_pointcloud_token(prompt, tokenizer, POINTCLOUD_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
    else:
        # Only image tokens present (original behavior)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            pointclouds=pointclouds,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--pointcloud-file", type=str, default=None, 
                        help="Path to pointcloud .pt files (comma-separated). Expected shape: (64, 382)")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
