# Pointcloud Usage Guide for LLaVA

## Quick Start

### 1. Basic Pointcloud Inference
```python
from llava.model.builder import load_pretrained_model
import torch

# Load model with pointcloud support
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path="your-pointcloud-model-path",
    model_base=None,
    model_name="model-name"
)

# Load pointcloud data (shape: 64, 382)
pointcloud = torch.load("pointcloud.pt")  # Your precomputed embeddings
pointclouds = pointcloud.unsqueeze(0).to(model.device, dtype=torch.float16)

# Generate response
output_ids = model.generate(
    input_ids,
    pointclouds=pointclouds,
    max_new_tokens=512
)
```

### 2. Training with Modality Masking
```python
# Configure training phases
training_args.training_phase = 'mixed'  # 'pointcloud_only', 'mixed', 'both'
training_args.pc_mask_probability = 0.3
training_args.image_mask_probability = 0.3

# During training, modalities are automatically masked based on probabilities
```

### 3. Using the Evaluation Script
```bash
python llava/eval/run_llava.py \
    --model-path /path/to/model \
    --pointcloud-file pointcloud.pt \
    --image-file image.jpg \
    --query "Describe what you see in both the pointcloud and image."
```

### 4. Data Format
- **Pointcloud files**: PyTorch tensors (.pt) with shape `(64, 382)`
- **Sequence order**: Pointcloud → Image → Text (as specified)
- **Token structure**: `<pointcloud>` for pointcloud data, `<image>` for images

### 5. Weight Management
```python
from scripts.pointcloud_weight_utils import save_pointcloud_projector, load_pointcloud_projector

# Save pointcloud weights
save_pointcloud_projector(model, "pc_weights.bin")

# Load into existing model
load_pointcloud_projector(model, "pc_weights.bin")
```