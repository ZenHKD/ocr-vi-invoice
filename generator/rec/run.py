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
    - Random alphanumeric with Vietnamese characters
    - English sentences
    - Vietnamese sentences (with and without diacritics)
    - Numbers (phone, account, codes)
    - Alphanumeric codes
        """
    )

    parser.add_argument("-n", "--num", type=int, default=1000, help="Total number of samples")
    parser.add_argument("-o", "--output", type=str, default="data", help="Output directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/val split ratio (default: 0.9)")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Setup output directories
    train_dir = os.path.join(args.output, "train_rec")
    val_dir = os.path.join(args.output, "val_rec")
    
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 50)
        print("SYNTHETIC TEXT GENERATOR")
        print("=" * 50)
        print(f"  Samples:      {args.num}")
        print(f"  Output:       {args.output}")
        print(f"  Train dir:    {train_dir}")
        print(f"  Val dir:      {val_dir}")
        print(f"  Train ratio:  {args.train_ratio}")
        print("=" * 50)
        print()

    # Initialize generator
    tg = TextGenerator(alignment=1)

    # Define text generation tasks
    text_types = [
        ("generate_date_strings", "date"),
        ("generate_amount_strings", "amount"),
        ("generate_random_strings", "random"),
        ("generate_english_strings", "english"),
        ("generate_vietnamese_strings", "vietnamese"),
        ("generate_number_strings", "number"),
        ("generate_code_strings", "code"),
    ]
    
    # Distribute samples across text types
    num_types = len(text_types)
    samples_per_type = args.num // num_types
    train_samples_per_type = int(samples_per_type * args.train_ratio)
    val_samples_per_type = samples_per_type - train_samples_per_type
    
    if not args.quiet:
        print(f"Samples per type: {samples_per_type}")
        print(f"  Train: {train_samples_per_type}")
        print(f"  Val:   {val_samples_per_type}")
        print()

    start_time = time.time()
    
    train_labels = []
    val_labels = []
    sample_id = 0
    
    # Generate for each text type
    for method_name, text_type in text_types:
        if not args.quiet:
            print(f"Generating {text_type}...")
        
        method = getattr(tg, method_name)
        
        # Generate train samples
        train_strings = method(count=train_samples_per_type)
        train_gen = tg.generate(train_strings)
        
        for img, lbl in tqdm(train_gen, total=train_samples_per_type, desc=f"  Train {text_type}", disable=args.quiet):
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            filename = f"{text_type}_{sample_id:06d}.jpg"
            filepath = os.path.join(train_dir, filename)
            img.save(filepath)
            train_labels.append((filename, lbl))
            sample_id += 1
        
        # Generate val samples
        val_strings = method(count=val_samples_per_type)
        val_gen = tg.generate(val_strings)
        
        for img, lbl in tqdm(val_gen, total=val_samples_per_type, desc=f"  Val   {text_type}", disable=args.quiet):
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

    if not args.quiet:
        print()
        print("=" * 50)
        print(f"SUCCESS: Generated {sample_id} samples in {elapsed:.2f}s")
        print("=" * 50)
        print(f"\nTrain samples: {len(train_labels)}")
        print(f"Val samples:   {len(val_labels)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
