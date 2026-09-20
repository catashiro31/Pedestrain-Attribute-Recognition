import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# turn on tfloat32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

#################################### Config ####################################
INPUT_DIR = Path("/workspace/domain_images")
OUTPUT_DIR = Path("/workspace/domain_noper_images")
PROMPT = "person"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

#################################### Load Model ################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("Loading SAM3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)
print("Model loaded.")

#################################### Process ###################################

def process_image(image_path: Path, output_path: Path):
    """Process a single image: detect persons, merge masks, save as binary PNG."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Cannot open {image_path}: {e}")
        return False

    # Run SAM3 inference
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=PROMPT)

    masks = output["masks"]  # shape: [N, 1, H, W], boolean tensor

    if masks.numel() == 0 or masks.shape[0] == 0:
        # No person detected — save an all-zero mask
        w, h = image.size
        merged = np.zeros((h, w), dtype=np.uint8)
    else:
        # Merge all person masks into one (logical OR across N detections)
        merged = masks.squeeze(1).any(dim=0).cpu().numpy().astype(np.uint8) * 255

    # Save mask as binary PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(merged, mode="L")
    mask_img.save(output_path)
    return True


def main():
    # Collect all domains
    domains = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not domains:
        print(f"No domain folders found in {INPUT_DIR}")
        return

    print(f"Found {len(domains)} domains: {[d.name for d in domains]}")

    total_processed = 0
    total_errors = 0

    for domain_dir in domains:
        domain_name = domain_dir.name
        output_domain_dir = OUTPUT_DIR / domain_name

        # Collect images
        images = sorted([
            f for f in domain_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ])

        if not images:
            print(f"[{domain_name}] No images found, skipping.")
            continue

        print(f"\n[{domain_name}] Processing {len(images)} images...")

        for img_path in tqdm(images, desc=domain_name):
            # Build output filename: stem + _mask.png
            out_filename = f"{img_path.stem}_mask.png"
            out_path = output_domain_dir / out_filename

            success = process_image(img_path, out_path)
            if success:
                total_processed += 1
            else:
                total_errors += 1

    print(f"\n{'='*60}")
    print(f"Done! Processed: {total_processed}, Errors: {total_errors}")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
