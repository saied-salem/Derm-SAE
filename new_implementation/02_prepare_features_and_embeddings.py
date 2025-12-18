import sys
# Add Derm1M to path so open_clip can find its custom models
sys.path.append('Derm1M')
sys.path.append('Derm1M/src')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# This import MUST come after sys.path.append
from Derm1M.src.open_clip import create_model_and_transforms, get_tokenizer
from tqdm import tqdm
import os
from PIL import Image
import pickle
from pathlib import Path
import argparse

# --- Class names must be globally known ---
CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma',
    'Benign keratosis-like lesions', 'Dermatofibroma',
    'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
]

# --- 1. Dataset Class ---
class HAM10000_Dataset(Dataset):
    def __init__(self, pickle_split_file, image_root_dir, image_ext, processor):
        print(f"Loading data split from {pickle_split_file}...")
        with open(pickle_split_file, 'rb') as f:
            class2images = pickle.load(f)
        self.image_root_dir = Path(image_root_dir)
        self.processor = processor 
        self.image_list = []
        
        print("Creating image list...")
        lower_class_names = [name.lower() for name in CLASS_NAMES]
        for class_name, img_filenames in class2images.items():
            try:
                class_idx = lower_class_names.index(class_name.lower())
            except ValueError:
                continue
            for img_name in img_filenames:
                img_name_stem = img_name.split(".")[0]
                img_path = self.image_root_dir / f"{img_name_stem}{image_ext}"
                if os.path.exists(img_path):
                    self.image_list.append({'path': img_path, 'label_idx': class_idx})

        print(f"Data loaded. Found {len(self.image_list)} images.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]
        img_path = item['path']
        label = item['label_idx']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224))
        
        pixel_values = self.processor(image)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)

# --- 2. Feature Extractor Hook (Verified) ---
class FeatureExtractor:
    def __init__(self, model: nn.Module):
        self.patch_features = None
        
        # Verified from our debug script:
        hook_target = model.visual.blocks[-1]
        
        self.hook = hook_target.register_forward_hook(self.hook_fn)
        print("Hook attached successfully to model.visual.blocks[-1].")
    
    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        # Verified from our debug script:
        # Output 'o' is already [Batch, Sequence, Dim]
        # No permutation is needed.
        self.patch_features = output

    def get_features(self):
        return self.patch_features

    def close(self):
        self.hook.remove()

# --- 3. Main Script ---
def get_args():
    parser = argparse.ArgumentParser(description="Step 2: Extract Patch Features and Text Embeddings")
    
    # --- Paths ---
    parser.add_argument('--data_split_file', type=str, default=None, help='Path to a single .p split file (e.g., class2images_train.p). Use either this OR train/val_split_files.')
    parser.add_argument('--train_split_file', type=str, default=None, help='Path to train split file (e.g., class2images_train.p). If provided with val_split_file, extracts from both.')
    parser.add_argument('--val_split_file', type=str, default=None, help='Path to val split file (e.g., class2images_val.p). If provided with train_split_file, extracts from both.')
    parser.add_argument('--image_root_dir', type=str, required=True, help='Path to the root folder of HAM10000 images')
    parser.add_argument('--image_ext', type=str, default='.jpg', help='Image file extension')
    parser.add_argument('--vocab_file', type=str, default='master_vocabulary.txt', help='Path to the master vocabulary file')
    
    # --- Model ---
    # We keep 'dermlip' as the choice, but the script now knows this means PanDerm
    parser.add_argument('--model_choice', type=str, default='dermlip', choices=['dermlip', 'clip'], help="Model to use")
    
    # --- Outputs ---
    parser.add_argument('--text_embed_output', type=str, default='./dermlip_text_embeddings.pt', help='Path to save text embeddings')
    parser.add_argument('--patch_feature_output', type=str, default=None, help='Path to save patch features (only used with --data_split_file). Otherwise train/val outputs are used.')
    parser.add_argument('--train_feature_output', type=str, default=None, help='Path to save train patch features (used with --train_split_file)')
    parser.add_argument('--val_feature_output', type=str, default=None, help='Path to save val patch features (used with --val_split_file)')

    # --- Config ---
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    return parser.parse_args()


def main():
    args = get_args()
    
    # --- Verified Model IDs ---
    # We use the PanDerm model ID by default when 'dermlip' is chosen
    if args.model_choice == 'dermlip':
        LOCAL_MODEL_NAME = 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'
        PRETRAINED_WEIGHTS_ID = 'default' # 'default' will load standard CLIP weights
    else:
        LOCAL_MODEL_NAME = 'ViT-B-16'
        PRETRAINED_WEIGHTS_ID = 'default' # 'default' will load standard CLIP weights
    
    TOKENIZER_ID = LOCAL_MODEL_NAME
    # --------------------------
    
    print(f"Using device: {args.device}")
    print(f"Loading model: '{LOCAL_MODEL_NAME}'")
    print(f"Loading weights: '{PRETRAINED_WEIGHTS_ID}'")
    
    model, _, preprocess = create_model_and_transforms(
        LOCAL_MODEL_NAME,
        pretrained=PRETRAINED_WEIGHTS_ID, 
        device=args.device
    )
    model.eval()
    
    print(f"Loading tokenizer from: {TOKENIZER_ID}")
    tokenizer = get_tokenizer(TOKENIZER_ID)
    
    # ================================================
    # PART A: Generate Text Embeddings
    # ================================================
    print(f"\n--- Part A: Generating Text Embeddings ---")
    
    if not os.path.exists(args.text_embed_output):
        if not os.path.exists(args.vocab_file):
            print(f"Error: '{args.vocab_file}' not found.")
            print("Please run '01_prepare_vocabulary.py' first.")
            return
            
        with open(args.vocab_file, 'r') as f:
            concepts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(concepts)} concepts from {args.vocab_file}.")
        
        all_text_embeddings = []
        with torch.no_grad(), torch.autocast(args.device):
            for concept in tqdm(concepts, desc="Encoding concepts"):
                text_tokens = tokenizer([concept]).to(args.device)
                all_text_embeddings.append(model.encode_text(text_tokens))
                
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0).cpu()
        torch.save({
            'concepts': concepts,
            'embeddings': all_text_embeddings
        }, args.text_embed_output)
        print(f"Saved {all_text_embeddings.shape[0]} text embeddings to {args.text_embed_output}")
    else:
        print(f"Found existing embeddings at '{args.text_embed_output}'. Skipping generation.")

    # ================================================
    # PART B: Extract and save all patch features
    # ================================================
    print(f"\n--- Part B: Extracting Patch Features ---")
    
    # Determine which split files to use and output files
    if args.train_split_file and args.val_split_file:
        # Extract from both train and val separately
        if not args.train_feature_output:
            args.train_feature_output = args.patch_feature_output.replace('.pt', '_train.pt') if args.patch_feature_output else './dermlip_patch_features_train.pt'
        if not args.val_feature_output:
            args.val_feature_output = args.patch_feature_output.replace('.pt', '_val.pt') if args.patch_feature_output else './dermlip_patch_features_val.pt'
        
        split_configs = [
            (args.train_split_file, "train", args.train_feature_output),
            (args.val_split_file, "val", args.val_feature_output)
        ]
        print("Extracting features from train and val splits separately...")
    elif args.data_split_file:
        # Use single split file (backward compatibility)
        if not args.patch_feature_output:
            print("Error: --patch_feature_output required when using --data_split_file")
            return
        split_configs = [(args.data_split_file, "data", args.patch_feature_output)]
        print(f"Extracting features from single split file...")
    else:
        print(f"Error: Must provide either --data_split_file OR both --train_split_file and --val_split_file")
        return
    
    if not os.path.exists(args.image_root_dir):
        print(f"Error: Image root directory not found: {args.image_root_dir}")
        return

    # This will now attach to model.visual.blocks[-1]
    feature_extractor = FeatureExtractor(model)
    print(f"Using device: {args.device}")
    
    # Extract features from each split file separately
    for split_file, split_name, output_file in split_configs:
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}. Skipping...")
            continue
        
        # Skip if output already exists
        if os.path.exists(output_file):
            print(f"Found existing {split_name} features at '{output_file}'. Skipping extraction.")
            continue
        
        print(f"\n--- Extracting from {split_name} split: {split_file} ---")
        dataset = HAM10000_Dataset(split_file, args.image_root_dir, args.image_ext, preprocess)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        split_patch_features = []
        split_labels = []
        
        with torch.no_grad(), torch.autocast(args.device):
            for pixel_values, labels in tqdm(dataloader, desc=f"Extracting {split_name} patches"):
                pixel_values = pixel_values.to(args.device)
                _ = model.visual(pixel_values)
                
                # Get [Batch, 197, 768]
                patch_features = feature_extractor.get_features() 
                
                # Get [Batch, 196, 768]
                patch_features_only = patch_features[:, 1:, :] # Skip [CLS] token
                
                split_patch_features.append(patch_features_only.cpu())
                split_labels.append(labels.cpu())
        
        if split_patch_features:
            split_patch_features = torch.cat(split_patch_features, dim=0)
            split_labels = torch.cat(split_labels, dim=0)
            
            print(f"\nFinished {split_name} extraction:")
            print(f"  Images: {split_patch_features.shape[0]}")
            print(f"  Patches per image: {split_patch_features.shape[1]}")
            print(f"  Feature dimension: {split_patch_features.shape[2]}")
            
            torch.save({
                'patch_features': split_patch_features,
                'labels': split_labels
            }, output_file)
            print(f"Saved {split_name} patch features to {output_file}")
        else:
            print(f"  Warning: No features extracted from {split_name} split")

    feature_extractor.close()


if __name__ == "__main__":
    main()