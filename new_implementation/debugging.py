import sys
# Add Derm1M to path so open_clip can find its custom models
sys.path.append('Derm1M')
sys.path.append('Derm1M/src')

import torch
import torch.nn as nn
# This import MUST come after sys.path.append
from Derm1M.src.open_clip import create_model_and_transforms

# --- This is the key ---
# 1. We use the *local model name* that Derm1M's code registers
#    (as seen in your UserWarning logs)
LOCAL_MODEL_NAME = 'cae_base_patch16_224' 

# 2. We use the *Hugging Face ID* as the pretrained weights path
PRETRAINED_WEIGHTS_ID = 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"--- Introspection Script ---")
    print(f"Using device: {DEVICE}")
    print(f"Attempting to load custom model: '{PRETRAINED_WEIGHTS_ID}'")
    print(f"With pretrained weights: '{PRETRAINED_WEIGHTS_ID}'")

    # This variable will be modified by the hook
    # We use a list (a mutable object) to get around the 'nonlocal' scope issue
    hook_output = []

    try:
        model, _, preprocess = create_model_and_transforms(
            PRETRAINED_WEIGHTS_ID,
            pretrained='default', 
            device=DEVICE
        )
        model.eval()
        
        print("\n--- ✅ MODEL LOADED SUCCESSFULLY ---")
        
        # --- 3. Print the visual architecture ---
        # This is what we need to find the hook point
        print("\n--- MODEL.VISUAL ARCHITECTURE ---")
        print(model.visual)
        print("---------------------------------")
        
        # --- 4. Get the feature dimensions ---
        print("\n--- FEATURE DIMENSIONS ---")
        
        # Create a dummy image to pass through
        dummy_image = torch.randn(2, 3, 224, 224).to(DEVICE)
        
        # We need to find the hook path by inspecting the architecture printout.
        # Based on your previous ViT log, it's likely 'model.visual.blocks[-1]'.
        # Let's try that.
        hook_target = None
        try:
            # The 'cae' model from your logs has an 'encoder' attribute
            hook_target = model.visual.encoder.blocks[-1]
            print("Found hook target at: model.visual.encoder.blocks[-1]")
        except AttributeError:
            try:
                hook_target = model.visual.blocks[-1]
                print("Found hook target at: model.visual.blocks[-1]")
            except AttributeError:
                print("\n--- ❌ CRITICAL ERROR ---")
                print("Could not find '.encoder.blocks' or '.blocks' in model.visual.")
                print("Please inspect the architecture printout above to find the real name")
                print("of the transformer layers (e.g., 'resblocks', 'layers', etc.)")
                print("and update the 'hook_target' line in this script.")
                return

        def hook_fn(m, i, o):
            # 'o' is the output of the hooked layer
            # For open_clip ViT, this is [Seq, Batch, Dim]
            # We permute to [Batch, Seq, Dim]
            hook_output.append(o)

        hook = hook_target.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            model.visual(dummy_image)
        
        hook.remove()
        
        if not hook_output:
            print("Error: Hook failed to capture any output.")
            return

        patch_features = hook_output[0]

        print(f"\nSuccessfully hooked!")
        print(f"  Patch Features Shape (Batch, Patches, Dim): {patch_features.shape}")
        
        # The [CLS] token is the first token (index 0)
        # The patch tokens are from index 1 onwards
        patch_features_only = patch_features[:, 1:, :]
        
        print(f"  Patches-Only Shape (B, N, D): {patch_features_only.shape}")
        print(f"  Feature Dimension (d_input for SAE): {patch_features_only.shape[-1]}")
        print(f"  Number of Patches (N): {patch_features_only.shape[1]}")


    except Exception as e:
        print(f"\n--- ❌ MODEL FAILED TO LOAD ---")
        print(f"Error: {e}")
        print("\nThis likely means the local model name is wrong.")
        print("Try changing LOCAL_MODEL_NAME to 'cae_base_patch16_224_8k_vocab' and re-running.")

if __name__ == "__main__":
    main()

