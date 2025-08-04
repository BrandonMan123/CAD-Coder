"""Sitecustomize patch for LLAVA bug.
Automatically flattens image features produced by the vision tower so that they
are always 2-D (#tokens, hidden). This avoids a runtime error when torch.cat()
tries to concatenate tensors of different rank (2-D text / point-cloud tokens
vs. 3-D image tokens) inside `prepare_inputs_labels_for_multimodal`.

This patch is only activated when the environment variable
`LLAVA_FLATTEN_IMAGE_FEATURES` (default "1") is set.  Because Python imports
`sitecustomize` automatically at startup (if it is importable), the fix applies
transparently to *all* scripts – both `test.py` and the evaluation shell script
– without requiring any changes to the core LLAVA library.
"""
import os

# Activate by default unless the variable is explicitly set to "0" / "false".
if os.getenv("LLAVA_FLATTEN_IMAGE_FEATURES", "1").lower() not in {"", "0", "false"}:
    try:
        import torch  # noqa: F401  # (required by the patched function)
        from llava.model.llava_arch import LlavaMetaForCausalLM

        def _encode_images_flat(self, images):  # type: ignore[override]
            """Patched version of `encode_images` that always returns 2-D features.

            • Runs the original vision tower + projector to obtain image features.
            • If the output still contains a batch dimension (rank-3 tensor
              shaped `[B, T, H]`), flatten the first two dims into one so the
              final shape is `[B*T, H]`.
            """
            feats = self.get_model().get_vision_tower()(images)
            feats = self.get_model().mm_projector(feats)
            # When `feats` is 3-D, merge the first two axes so that the result is 2-D.
            if feats.ndim == 3:
                feats = feats.flatten(0, 1)
            return feats

        # Monkey-patch the method on the base class so every concrete model type
        # picks it up automatically.
        LlavaMetaForCausalLM.encode_images = _encode_images_flat  # type: ignore[assignment]
    except Exception as e:  # pragma: no cover
        # Silently skip if LLAVA is not available yet – the import will happen
        # later in the user script in which case the original bug cannot occur
        # because no multimodal generation will be attempted.
        pass
