"""
merge_datasets.py — Smart ASL Dataset Merger
=============================================
Combines multiple ASL image datasets into one clean merged_dataset/
Rules:
  - Only A-Z single-letter classes (skips 0-9, skips whole-word folders like HELLO, YES, etc.)
  - Caps at MAX_PER_CLASS images per letter per source (to keep balance)
  - Deduplication: files are renamed uniquely per source to avoid overwrites
  - Output: merged_dataset/A/, merged_dataset/B/, ..., merged_dataset/Z/
"""

import os
import shutil
import random

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
VALID_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_PER_CLASS = 1000   # max images per letter per source dataset
IMG_EXTS      = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
OUTPUT_DIR    = os.path.join(BASE_DIR, "merged_dataset")

random.seed(42)

# ── Define all source dataset roots (one per source) ──────────────────────────
SOURCES = [
    ("raw",       os.path.join(BASE_DIR, "ASL_Raw_Images",       "asl_dataset")),
    ("processed", os.path.join(BASE_DIR, "ASL_Processed_Images", "asl_processed", "train")),
    ("dynamic",   os.path.join(BASE_DIR, "ASL_dataset",           "ASL_dynamic",   "ASL_dynamic")),
    ("alpha",     os.path.join(BASE_DIR, "ASL_dataset",           "SignAlphaSet",   "SignAlphaSet")),
]


def is_valid_letter_folder(name: str) -> bool:
    """Return True only if folder name is a single A-Z letter."""
    return len(name) == 1 and name.upper() in VALID_LETTERS


def merge_all():
    # Create output letter folders
    for letter in VALID_LETTERS:
        os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

    totals = {l: 0 for l in VALID_LETTERS}

    for source_tag, source_root in SOURCES:
        if not os.path.isdir(source_root):
            print(f"  ⚠️  Skipping '{source_tag}' — folder not found: {source_root}")
            continue

        print(f"\n📂 Processing source: [{source_tag}]  →  {source_root}")
        class_folders = sorted(os.listdir(source_root))

        for folder in class_folders:
            letter = folder.upper()
            if not is_valid_letter_folder(folder):
                print(f"    ⏩ Skipping '{folder}' (not a single A-Z letter)")
                continue

            src_dir = os.path.join(source_root, folder)
            if not os.path.isdir(src_dir):
                continue

            images = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]

            # Cap and shuffle
            if len(images) > MAX_PER_CLASS:
                images = random.sample(images, MAX_PER_CLASS)

            copied = 0
            dest_dir = os.path.join(OUTPUT_DIR, letter)
            for i, img_file in enumerate(images):
                src_path  = os.path.join(src_dir, img_file)
                ext       = os.path.splitext(img_file)[1].lower()
                dest_name = f"{source_tag}_{letter}_{i:04d}{ext}"
                dest_path = os.path.join(dest_dir, dest_name)
                shutil.copy2(src_path, dest_path)
                copied += 1

            totals[letter] += copied
            print(f"    [{letter}]  +{copied:4d}  (total so far: {totals[letter]})")

    print(f"\n{'='*60}")
    print(f"  ✅ Merge Complete! Output: {OUTPUT_DIR}")
    print(f"  📊 Total images per class:")
    for l in sorted(totals):
        bar = "█" * (totals[l] // 100)
        print(f"    {l}: {totals[l]:5d}  {bar}")
    grand_total = sum(totals.values())
    print(f"\n  🏆 Grand Total: {grand_total} images across 26 classes")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("🚀 ASL Dataset Merger Starting...")
    if os.path.exists(OUTPUT_DIR):
        answer = input(f"\n⚠️  Output dir '{OUTPUT_DIR}' already exists. Delete and rebuild? (y/n): ")
        if answer.strip().lower() == "y":
            shutil.rmtree(OUTPUT_DIR)
        else:
            print("Aborted.")
            exit(0)
    merge_all()
