#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, POINTCLOUD_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_POINTCLOUD_TOKEN, DEFAULT_PC_START_TOKEN, DEFAULT_PC_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
        
        # Initialize pointcloud projector if pointcloud config exists
        if hasattr(config, "pc_hidden_size"):
            self.pc_projector = self.build_pointcloud_projector(config)
        else:
            self.pc_projector = None

    def build_pointcloud_projector(self, config):
        """Build pointcloud projector similar to vision projector"""
        projector_type = getattr(config, 'pc_projector_type', 'linear')
        pc_hidden_size = getattr(config, 'pc_hidden_size', 382)  # Default pointcloud token dimension
        
        if projector_type == 'linear':
            return nn.Linear(pc_hidden_size, config.hidden_size)
        elif projector_type == 'identity':
            return nn.Identity()
        else:
            # Use similar logic as vision projector for MLP
            import re
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(pc_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                return nn.Sequential(*modules)
            else:
                raise ValueError(f'Unknown pointcloud projector type: {projector_type}')

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_pointcloud_modules(self, model_args):
        """Initialize pointcloud modules similar to vision modules"""
        pc_hidden_size = getattr(model_args, 'pc_hidden_size', 382)  # Pointcloud token dimension
        pc_projector_type = getattr(model_args, 'pc_projector_type', 'linear')
        pretrain_pc_mlp_adapter = getattr(model_args, 'pretrain_pc_mlp_adapter', None)
        
        self.config.pc_hidden_size = pc_hidden_size
        self.config.pc_projector_type = pc_projector_type
        
        if getattr(self, 'pc_projector', None) is None:
            self.pc_projector = self.build_pointcloud_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.pc_projector.parameters():
                p.requires_grad = True

        if pretrain_pc_mlp_adapter is not None:
            pc_projector_weights = torch.load(pretrain_pc_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.pc_projector.load_state_dict(get_w(pc_projector_weights, 'pc_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_pointclouds(self, pointclouds):
        """Encode pointcloud features using pointcloud projector"""
        model = self.get_model()
        if hasattr(model, 'pc_projector') and model.pc_projector is not None:
            pc_features = model.pc_projector(pointclouds)
            return pc_features
        else:
            # For backwards compatibility, return None if no pointcloud projector
            return None

    # ─────────────────────────────────────────────────────────────────────────────
    #                           Helper functions
    # ─────────────────────────────────────────────────────────────────────────────

    def _normalize_defaults(self, input_ids, attention_mask, position_ids, labels):
        """Ensure non-None masks/positions/labels; return normalized + original flags."""
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        return (attention_mask, position_ids, labels,
                _attention_mask is None, _position_ids is None, _labels is None)

    def _trim_by_attention_mask(self, input_ids, labels, attention_mask):
        """Remove padding per-sample using attention_mask; returns lists of 1-D tensors."""
        input_ids_list = [ids_row[mask_row] for ids_row, mask_row in zip(input_ids, attention_mask)]
        labels_list    = [lab_row[mask_row] for lab_row, mask_row in zip(labels,    attention_mask)]
        return input_ids_list, labels_list

    def _encode_images_normalized(self, images, image_sizes):
        """
        Returns per-sample list of image features, each [N_img_tokens, hidden], or None.
        Mirrors the original logic: list/5D input, mm_patch_merge_type (flat/spatial*).
        """
        if images is None:
            return None

        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            return None

        config = self.config
        mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio  = getattr(config, 'image_aspect_ratio', 'square')

        # Case A: list of tensors or 5D tensor -> concatenate, encode, split
        if (type(images) is list) or (getattr(images, "ndim", 0) == 5):
            if type(images) is list:
                # Ensure leading batch dimension on each image tensor
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                split_sizes   = [image.shape[0] for image in images]
            else:
                concat_images = images
                split_sizes   = [images.shape[0]]  # single entry

            feats = self.encode_images(concat_images)  # list or tensor depending on projector
            # normalize to list of [B_i, P, H]
            if isinstance(feats, (list, tuple)):
                image_features = feats
            else:
                image_features = torch.split(feats, split_sizes, dim=0)

            # Merge strategy
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]  # [sum_patches, H]
            elif mm_patch_merge_type.startswith('spatial'):
                new_list = []
                for idx, img_feat in enumerate(image_features):
                    if img_feat.shape[0] > 1:
                        base = img_feat[0]
                        img_feat_rest = img_feat[1:]
                        height = width = vision_tower.num_patches_per_side
                        assert height * width == base.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_w, num_h = get_anyres_image_grid_shape(
                                image_sizes[idx],
                                config.image_grid_pinpoints,
                                vision_tower.config.image_size
                            )
                            img_feat_rest = img_feat_rest.view(num_h, num_w, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            img_feat_rest = img_feat_rest.permute(4, 0, 2, 1, 3).contiguous()
                            img_feat_rest = img_feat_rest.flatten(1, 2).flatten(2, 3)
                            img_feat_rest = unpad_image(img_feat_rest, image_sizes[idx])
                            img_feat_rest = torch.cat((
                                img_feat_rest,
                                self.model.image_newline[:, None, None]
                                    .expand(*img_feat_rest.shape[:-1], 1)
                                    .to(img_feat_rest.device)
                            ), dim=-1)
                            img_feat_rest = img_feat_rest.flatten(1, 2).transpose(0, 1)
                        else:
                            img_feat_rest = img_feat_rest.permute(0, 2, 1, 3, 4).contiguous()
                            img_feat_rest = img_feat_rest.flatten(0, 3)
                        merged = torch.cat((base, img_feat_rest), dim=0)
                    else:
                        # only base
                        merged = img_feat[0]
                        if 'unpad' in mm_patch_merge_type:
                            merged = torch.cat((merged, self.model.image_newline[None].to(merged.device)), dim=0)
                    new_list.append(merged)
                image_features = new_list
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")

            return image_features  # list of [N_img_tokens, H]

        # Case B: simple tensor already encoded or single (B, N, H)
        feats = self.encode_images(images)
        if feats is None:
            return None
        if feats.ndim == 3 and feats.shape[0] == 1:
            return [feats[0]]  # [N, H]
        if feats.ndim == 2:
            return [feats]     # [N, H]
        # If it's (B, N, H), split into list
        if feats.ndim == 3:
            return [f for f in feats]
        return None

    def _encode_pointclouds_normalized(self, pointclouds):
        """
        Returns per-sample list of pointcloud features (each [N_pc_tokens, hidden]) or None.
        Mirrors original: if projector missing, returns None.
        """
        if pointclouds is None:
            return None
        feats = self.encode_pointclouds(pointclouds)
        if feats is None:
            return None
        # Normalize to list of [N, H]
        if isinstance(feats, (list, tuple)):
            return [f if f.ndim == 2 else f[0] for f in feats]  # handle [B,N,H] → [N,H]
        if feats.ndim == 3 and feats.shape[0] == 1:
            return [feats[0]]
        if feats.ndim == 2:
            return [feats]
        if feats.ndim == 3:
            return [f for f in feats]
        return None

    def _maybe_mask_features(self, feature_list, do_mask):
        if feature_list is None or not do_mask:
            return feature_list
        return [torch.zeros_like(f) for f in feature_list]

    def _find_modal_positions(self, ids_row):
        pc_pos  = torch.where(ids_row == POINTCLOUD_TOKEN_INDEX)[0]
        img_pos = torch.where(ids_row == IMAGE_TOKEN_INDEX)[0]
        return pc_pos, img_pos

    def _slice_text_regions(self, ids_row, pc_pos, img_pos):
        # first modality position
        first_modal_idx = pc_pos[0] if len(pc_pos) else (img_pos[0] if len(img_pos) else ids_row.shape[0])
        text_before = ids_row[:first_modal_idx]

        if len(pc_pos) and len(img_pos):
            text_between = ids_row[pc_pos[-1] + 1 : img_pos[0]]
        else:
            text_between = ids_row[:0]  # empty

        text_after = ids_row[img_pos[-1] + 1 :] if len(img_pos) else ids_row[first_modal_idx:]
        return text_before, text_between, text_after

    def _embed_text_slice(self, ids_slice):
        if ids_slice.numel() == 0:
            return None
        return self.get_model().embed_tokens(ids_slice)

    def _assemble_sample_sequence(
        self,
        ids_row,
        labels_row,
        img_feats_list,
        pc_feats_list,
        img_idx,
        pc_idx,
        device,
        hidden_size,
        dtype,
        enforce_pc_before_img=True,
        mask_images=False,   # masking is already applied to features; kept for parity
        mask_pointclouds=False
    ):
        """
        Build fused [T,H] embeddings and [T] labels for one sample.
        Consumes at most one pc block and one img block.
        Returns: (embeds_row, labels_row_new, img_idx_new, pc_idx_new)
        """
        pc_pos, img_pos = self._find_modal_positions(ids_row)

        # Sanity/order check (preserve original behavior)
        if enforce_pc_before_img and len(pc_pos) and len(img_pos):
            assert pc_pos.max() < img_pos.min(), "Point-cloud tokens must come before image tokens"

        text_before, text_between, text_after = self._slice_text_regions(ids_row, pc_pos, img_pos)

        parts_embeds, parts_labels = [], []

        # ---- before-PC text ----
        emb = self._embed_text_slice(text_before)
        if emb is not None:
            parts_embeds.append(emb)
            first_modal_idx = pc_pos[0] if len(pc_pos) else (img_pos[0] if len(img_pos) else ids_row.shape[0])
            parts_labels.append(labels_row[:first_modal_idx])

        # ---- point-cloud block ----
        if len(pc_pos) and (pc_feats_list is not None) and (pc_idx < len(pc_feats_list)):
            cur_pc = pc_feats_list[pc_idx]
            parts_embeds.append(cur_pc.to(device))
            parts_labels.append(torch.full((cur_pc.shape[0],), IGNORE_INDEX, device=labels_row.device, dtype=labels_row.dtype))
            pc_idx += 1
        # else: skip PC block (back-compat)

        # ---- between text ----
        emb = self._embed_text_slice(text_between)
        if emb is not None:
            start = (pc_pos[-1] + 1) if len(pc_pos) else (pc_pos[0] if len(pc_pos) else (img_pos[0] if len(img_pos) else 0))
            end   = img_pos[0] if len(img_pos) else start
            parts_embeds.append(emb)
            parts_labels.append(labels_row[start:end])

        # ---- image block ----
        if len(img_pos):
            cur_img = None
            if (img_feats_list is not None) and (img_idx < len(img_feats_list)):
                cur_img = img_feats_list[img_idx]
            if cur_img is None:
                # Preserve original fallback: 256 dummy image tokens
                cur_img = torch.zeros(256, hidden_size, device=device, dtype=dtype)
            parts_embeds.append(cur_img.to(device))
            parts_labels.append(torch.full((cur_img.shape[0],), IGNORE_INDEX, device=labels_row.device, dtype=labels_row.dtype))
            img_idx += 1

        # ---- trailing text ----
        emb = self._embed_text_slice(text_after)
        if emb is not None:
            tail_start = img_pos[-1] + 1 if len(img_pos) else (pc_pos[0] if len(pc_pos) else 0)
            parts_embeds.append(emb)
            parts_labels.append(labels_row[tail_start:])

        # concat
        parts_embeds = [p for p in parts_embeds if p is not None]
        parts_labels = [p for p in parts_labels if p is not None and p.numel() > 0]

        if len(parts_embeds) == 0:
            # degenerate: no content → zero token (shouldn't happen with valid input)
            embeds_row = torch.zeros(1, hidden_size, device=device, dtype=dtype)
            labels_new = torch.full((1,), IGNORE_INDEX, device=labels_row.device, dtype=labels_row.dtype)
        else:
            embeds_row = torch.cat(parts_embeds, dim=0)
            labels_new = torch.cat(parts_labels, dim=0)

        return embeds_row, labels_new, img_idx, pc_idx

    def _truncate_sequences(self, embeds_list, labels_list, max_len):
        if max_len is None:
            return embeds_list, labels_list
        e_trunc, l_trunc = [], []
        for e, l in zip(embeds_list, labels_list):
            e_trunc.append(e[:max_len])
            l_trunc.append(l[:max_len])
        return e_trunc, l_trunc

    def _pad_and_pack(self, embeds_list, labels_list, attention_mask_like, position_ids_like, padding_side, hidden_size, device, dtype):
        batch_size = len(embeds_list)
        max_len = max(e.shape[0] for e in embeds_list) if batch_size > 0 else 0

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=labels_list[0].dtype,
            device=labels_list[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask_like.dtype, device=attention_mask_like.device)
        position_ids   = torch.zeros((batch_size, max_len), dtype=position_ids_like.dtype, device=position_ids_like.device)

        for i, (emb_i, lab_i) in enumerate(zip(embeds_list, labels_list)):
            cur_len = emb_i.shape[0]
            if padding_side == "left":
                pad = torch.zeros((max_len - cur_len, hidden_size), dtype=dtype, device=device)
                new_input_embeds_padded.append(torch.cat((pad, emb_i), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = lab_i
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids_like.dtype, device=position_ids_like.device)
            else:
                pad = torch.zeros((max_len - cur_len, hidden_size), dtype=dtype, device=device)
                new_input_embeds_padded.append(torch.cat((emb_i, pad), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = lab_i
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids_like.dtype, device=position_ids_like.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        return new_input_embeds, new_labels_padded, attention_mask, position_ids

    # ─────────────────────────────────────────────────────────────────────────────
    #                       Refactored main entry point
    # ─────────────────────────────────────────────────────────────────────────────

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, pointclouds=None, mask_pointclouds=False, mask_images=False
    ):
        vision_tower = self.get_vision_tower()
        no_multimodal = images is None and pointclouds is None

        # Early exit
        if (vision_tower is None or no_multimodal or input_ids.shape[1] == 1):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Normalize defaults and record which were originally None
        (attention_mask, position_ids, labels,
         attn_was_none, pos_was_none, labels_was_none) = self._normalize_defaults(
            input_ids, attention_mask, position_ids, labels
        )

        device = self.device
        hidden_size = self.config.hidden_size
        tok_dtype = self.get_model().embed_tokens.weight.dtype
        padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')

        # Encode modalities into per-sample lists
        image_features_list = self._encode_images_normalized(images, image_sizes)
        pc_features_list    = self._encode_pointclouds_normalized(pointclouds)

        # Apply optional masking to features (once, centralized)
        image_features_list = self._maybe_mask_features(image_features_list, mask_images)
        pc_features_list    = self._maybe_mask_features(pc_features_list,    mask_pointclouds)

        # Remove padding to variable-length
        input_ids_list, labels_list = self._trim_by_attention_mask(input_ids, labels, attention_mask)

        # Assemble per-sample sequences
        new_embeds_list, new_labels_list = [], []
        img_idx, pc_idx = 0, 0

        for ids_row, lab_row in zip(input_ids_list, labels_list):
            # Quick path: no IMG tokens -> plain text embedding (preserve behavior)
            num_images = (ids_row == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                new_embeds_list.append(self.get_model().embed_tokens(ids_row))
                new_labels_list.append(lab_row)
                continue

            embeds_row, labels_row_new, img_idx, pc_idx = self._assemble_sample_sequence(
                ids_row, lab_row,
                image_features_list, pc_features_list,
                img_idx, pc_idx,
                device, hidden_size, tok_dtype,
                enforce_pc_before_img=True,
                mask_images=mask_images,
                mask_pointclouds=mask_pointclouds
            )
            new_embeds_list.append(embeds_row)
            new_labels_list.append(labels_row_new)

        # Optional truncation
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        new_embeds_list, new_labels_list = self._truncate_sequences(
            new_embeds_list, new_labels_list, tokenizer_model_max_length
        )

        # Pad & pack to batch tensors
        new_input_embeds, new_labels_padded, attention_mask_new, position_ids_new = self._pad_and_pack(
            new_embeds_list, new_labels_list,
            attention_mask_like=attention_mask,
            position_ids_like=position_ids,
            padding_side=padding_side,
            hidden_size=hidden_size,
            device=device,
            dtype=tok_dtype
        )

        # Restore original None-ness
        out_labels = None if labels_was_none else new_labels_padded
        out_attention_mask = None if attn_was_none else attention_mask_new.to(dtype=attention_mask.dtype)
        out_position_ids   = None if pos_was_none  else position_ids_new

        # We return new_input_embeds for the model to bypass token embedding
        return None, out_position_ids, out_attention_mask, past_key_values, new_input_embeds, out_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
