"""
SYNTHETIC TEXT RECOGNITION DATA GENERATOR

Generates synthetic text images for OCR training using trdg library.

Features:
    - Multiple text types (dates, amounts, codes, Vietnamese, English)
    - Automatic train/validation split
    - Simple sequential generation

Usage:
    # Quick generation with defaults
    python -m generator.rec.run

    # Custom generation
    python -m generator.rec.run --num 1000

    # Generate to specific directory
    python -m generator.rec.run --num 500 --output data/custom
"""

import argparse
import sys
import os
import time
import random
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for module imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from generator.rec.text_generator import TextGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic text recognition dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 1000 samples
    python -m generator.rec.run --num 1000

    # Generate to custom output directory
    python -m generator.rec.run --num 500 --output data/custom

Text Types Generated:
    - Dates (various formats)
    - Amounts (currency values)
    - English sentences
    - Vietnamese sentences (with and without diacritics)
    - Numbers (phone, account, codes)
    - Alphanumeric codes
        """
    )

    parser.add_argument("-n", "--num", type=int, default=1000, help="Total number of samples")
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Define text generation tasks with their proportions (must sum to 1.0)
    text_types = [
        ("generate_vietnamese_strings", "vietnamese", 0.40), 
        ("generate_english_strings", "english", 0.20),
        ("generate_amount_strings", "amount", 0.15),
        ("generate_number_strings", "number", 0.15),

        ("generate_date_strings", "date", 0.04),
        ("generate_code_strings", "code", 0.05),

        ("generate_empty_strings", "empty", 0.005),
        ("generate_random_strings", "random", 0.005)
    ]

    # Validate proportions sum to 1.0
    total_proportion = sum(proportion for _, _, proportion in text_types)
    if abs(total_proportion - 1.0) > 0.001:
        raise ValueError(f"Proportions must sum to 1.0, got {total_proportion}")

    # Setup output directories
    train_dir = os.path.join(args.output, "train_rec")
    val_dir = os.path.join(args.output, "val_rec")
    
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("SYNTHETIC TEXT GENERATOR")
    print("=" * 50)
    print(f"  Samples:      {args.num}")
    print(f"  Output:       {args.output}")
    print(f"  Train dir:    {train_dir}")
    print(f"  Val dir:      {val_dir}")
    print("-" * 50)
    print("Type proportions:")
    for _, text_type, proportion in text_types:
        count = int(args.num * proportion)
        print(f"  {text_type:12s}: {proportion:.1%} ({count} samples)")
    print("=" * 50)
    print()

    # Initialize generator
    tg = TextGenerator(
        alignment=1,
        blur=1,               
        random_blur=True,      
        background_type=0,     # Gaussian noise
        distorsion_type=1,     # Sine wave
        distorsion_orientation=0,
        text_color="#282828"   # Gray Black
    )

    start_time = time.time()
    
    train_labels = []
    val_labels = []
    sample_id = 0
    
    # Generate for each text type
    for method_name, text_type, proportion in text_types:
        type_count = int(args.num * proportion)
        # 90/10 train/val split for each type
        train_count = int(type_count * 0.9)
        val_count = type_count - train_count
        
        print(f"Generating {text_type} ({type_count} total, {train_count} train, {val_count} val)...")
        
        method = getattr(tg, method_name)
        
        # Generate train samples
        train_strings = method(count=train_count)
        train_gen = tg.generate(train_strings)
        
        for img, lbl in tqdm(train_gen, total=train_count, desc=f"  Train {text_type}"):
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            filename = f"{text_type}_{sample_id:06d}.jpg"
            filepath = os.path.join(train_dir, filename)
            img.save(filepath)
            train_labels.append((filename, lbl))
            sample_id += 1
        
        # Generate val samples
        val_strings = method(count=val_count)
        val_gen = tg.generate(val_strings)
        
        for img, lbl in tqdm(val_gen, total=val_count, desc=f"  Val   {text_type}"):
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            filename = f"{text_type}_{sample_id:06d}.jpg"
            filepath = os.path.join(val_dir, filename)
            img.save(filepath)
            val_labels.append((filename, lbl))
            sample_id += 1

    elapsed = time.time() - start_time

    # Write label files
    import csv
    
    with open(os.path.join(train_dir, "labels.csv"), "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'text'])
        writer.writerows(train_labels)
    
    with open(os.path.join(val_dir, "labels.csv"), "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'text'])
        writer.writerows(val_labels)

    print()
    print("=" * 50)
    print(f"SUCCESS: Generated {sample_id} samples in {elapsed:.2f}s")
    print("=" * 50)
    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Val samples:   {len(val_labels)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())