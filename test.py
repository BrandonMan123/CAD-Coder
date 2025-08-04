import torch, types
from llava.constants import (
    DEFAULT_POINTCLOUD_TOKEN, DEFAULT_IMAGE_TOKEN,
    POINTCLOUD_TOKEN_INDEX, IMAGE_TOKEN_INDEX, IGNORE_INDEX)
from llava.model.builder import load_pretrained_model

# -------- 1) load whatever checkpoint you want to test ----------
MODEL_PATH = "CADCODER/CAD-Coder"          # ← change if needed
tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, MODEL_PATH.split("/")[-1])

# -------- 2) build a fake point-cloud projector if the model lacks one ----------
if not (hasattr(model.get_model(), "pc_projector") and model.get_model().pc_projector):
    hidden = model.config.hidden_size
    model.get_model().pc_projector = torch.nn.Linear(382, hidden).to(model.device)

# -------- 3) monkey-patch encode_pointclouds so it accepts raw tensors ----------
def _dummy_encode(self, pcs):
    # pcs  shape  [B, 64, 382]   ->   [64, hidden_size]
    return self.get_model().pc_projector(pcs[0])
model.encode_pointclouds = types.MethodType(_dummy_encode, model)

# -------- 4) craft a minimal prompt containing <pointcloud> and <image> tokens ---
prompt_text = (
    f"{DEFAULT_POINTCLOUD_TOKEN} "          # point-cloud slot
    f"{DEFAULT_IMAGE_TOKEN}\n"              # image slot
    "Describe this engineering part."
)
input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)

from PIL import Image
# create a dummy white PIL image  (size doesn’t matter – processor will resize)
pil = Image.new("RGB", (512, 512), color="white")
img_tensor = image_processor(pil, return_tensors='pt')['pixel_values'].to(model.device)


# -------- 5) fabricate dummy modalities -----------------------------------------
fake_pc  = torch.randn(1, 64, 382, device=model.device)          # point-cloud tokens
fake_img = torch.randn(1, 3, 224, 224, device=model.device)      # one 224×224 RGB image

# -------- 6) do a single forward / generate pass -------------------------------
with torch.inference_mode():
    outs = model.generate(
        input_ids,
        images=img_tensor.half(),
        pointclouds=fake_pc,
        max_new_tokens=20
    )

    outs_no_pc = model.generate(
        input_ids,
        images=img_tensor.half(),
        pointclouds=None,
        max_new_tokens=20
    )
    outs_no_img = model.generate(
        input_ids,
        images=None,              # <-- skip vision tower entirely
        pointclouds=fake_pc,
        max_new_tokens=20
    )
print("\n--- MODEL OUTPUT ---------------------------------\n",
      tokenizer.decode(outs[0], skip_special_tokens=True))
print("\n--- MODEL OUTPUT ---------------------------------\n",
      tokenizer.decode(outs_no_img[0], skip_special_tokens=True))
print("\n--- MODEL OUTPUT ---------------------------------\n",
      tokenizer.decode(outs_no_pc[0], skip_special_tokens=True))