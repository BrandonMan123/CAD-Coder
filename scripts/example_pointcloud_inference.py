#!/usr/bin/env python3
"""
Example script demonstrating how to use pointcloud functionality with LLaVA.

This script shows how to:
1. Load a pointcloud-enabled LLaVA model
2. Load pointcloud data from .pt files
3. Combine pointcloud, image, and text inputs
4. Generate responses using multimodal inputs

Usage:
    python scripts/example_pointcloud_inference.py \
        --model-path /path/to/pointcloud-enabled-model \
        --pointcloud-file /path/to/pointcloud.pt \
        --image-file /path/to/image.jpg \
        --query "Describe what you see in the pointcloud and image."
"""

import argparse
import torch
import sys
import os

# Add the llava directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llava.constants import (
    DEFAULT_POINTCLOUD_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    POINTCLOUD_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_multimodal_token,
    get_model_name_from_path,
)
from PIL import Image


def load_pointcloud(pointcloud_file):
    """Load pointcloud from .pt file. Expected shape: (64, 382)"""
    try:
        pointcloud = torch.load(pointcloud_file, map_location='cpu')
        print(f"📊 Loaded pointcloud shape: {pointcloud.shape}")
        
        # Ensure the pointcloud has the expected shape
        if pointcloud.dim() == 2 and pointcloud.shape == (64, 382):
            return pointcloud
        else:
            raise ValueError(f"Expected pointcloud shape (64, 382), got {pointcloud.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load pointcloud from {pointcloud_file}: {e}")


def create_sample_pointcloud(output_path):
    """Create a sample pointcloud tensor for testing purposes"""
    # Create a random pointcloud with the expected shape (64, 382)
    sample_pointcloud = torch.randn(64, 382)
    torch.save(sample_pointcloud, output_path)
    print(f"📁 Created sample pointcloud at: {output_path}")
    return sample_pointcloud


def demonstrate_pointcloud_inference(model_path, pointcloud_file, image_file, query, 
                                   model_base=None, create_sample=False):
    """Main demonstration function"""
    
    print("🚀 Starting Pointcloud + LLaVA Inference Demo")
    print("=" * 50)
    
    # Initialize model
    print("🔧 Loading model...")
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    
    # Check if model supports pointclouds
    has_pointcloud_support = hasattr(model, 'encode_pointclouds')
    print(f"🆕 Pointcloud support: {'✅' if has_pointcloud_support else '❌'}")
    
    if not has_pointcloud_support:
        print("⚠️  Model doesn't seem to have pointcloud support. Proceeding anyway...")
    
    # Load pointcloud data
    if create_sample and pointcloud_file:
        pointcloud = create_sample_pointcloud(pointcloud_file)
    elif pointcloud_file:
        print(f"📂 Loading pointcloud from: {pointcloud_file}")
        pointcloud = load_pointcloud(pointcloud_file)
    else:
        print("📂 No pointcloud file provided, creating sample data...")
        pointcloud = torch.randn(64, 382)
    
    # Load image
    if image_file:
        print(f"🖼️  Loading image from: {image_file}")
        image = Image.open(image_file).convert('RGB')
        images = [image]
        image_sizes = [image.size]
        images_tensor = process_images(images, image_processor, model.config)
    else:
        print("🖼️  No image file provided, skipping image input...")
        images_tensor = None
        image_sizes = None
    
    # Prepare query with tokens
    print(f"💬 Query: {query}")
    
    # Build prompt with both modalities
    if pointcloud_file and image_file:
        # Both pointcloud and image
        prompt_template = f"{DEFAULT_POINTCLOUD_TOKEN}\n{DEFAULT_IMAGE_TOKEN}\n{query}"
        print("🔗 Using both pointcloud and image modalities")
    elif pointcloud_file:
        # Only pointcloud
        prompt_template = f"{DEFAULT_POINTCLOUD_TOKEN}\n{query}"
        print("🔗 Using pointcloud-only modality")
    elif image_file:
        # Only image (fallback)
        prompt_template = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
        print("🔗 Using image-only modality")
    else:
        # Text only
        prompt_template = query
        print("🔗 Using text-only modality")
    
    # Create conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], prompt_template)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"📝 Full prompt: {prompt[:200]}...")
    
    # Tokenize
    if pointcloud_file and image_file:
        input_ids = tokenizer_multimodal_token(prompt, tokenizer, return_tensors='pt')
    else:
        # Fall back to regular tokenization for now
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    input_ids = input_ids.to(model.device)
    
    # Prepare data for model
    if images_tensor is not None:
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    
    if pointcloud is not None:
        pointclouds = pointcloud.unsqueeze(0).to(model.device, dtype=torch.float16)  # Add batch dimension
    else:
        pointclouds = None
    
    print("🧠 Generating response...")
    
    # Generate response
    with torch.inference_mode():
        try:
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                pointclouds=pointclouds,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )
            
            # Decode response
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            print("\n🎯 Model Response:")
            print("-" * 30)
            print(outputs)
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print("This might be due to:")
            print("  - Model not having pointcloud components initialized")
            print("  - Tokenization issues with multimodal tokens")
            print("  - Model configuration mismatches")
            return False
    
    print("\n✅ Demo completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Demonstrate pointcloud inference with LLaVA")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the pointcloud-enabled LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (for LoRA models)")
    parser.add_argument("--pointcloud-file", type=str, default=None,
                        help="Path to pointcloud .pt file (shape: 64, 382)")
    parser.add_argument("--image-file", type=str, default=None,
                        help="Path to image file")
    parser.add_argument("--query", type=str, 
                        default="Describe what you observe in the provided data.",
                        help="Query to ask the model")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a sample pointcloud file for testing")
    
    args = parser.parse_args()
    
    if not args.pointcloud_file and not args.image_file:
        print("❌ Error: Must provide either --pointcloud-file or --image-file (or both)")
        return 1
    
    try:
        success = demonstrate_pointcloud_inference(
            model_path=args.model_path,
            pointcloud_file=args.pointcloud_file,
            image_file=args.image_file,
            query=args.query,
            model_base=args.model_base,
            create_sample=args.create_sample
        )
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())