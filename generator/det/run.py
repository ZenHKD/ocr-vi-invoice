"""
SYNTHETIC INVOICE GENERATOR

Uses the modular synthetic_data package for generating diverse, realistic invoice data.

Features:
    - Multiple layout types (thermal, VAT, handwritten, cafe)
    - Realistic purchase behavior patterns
    - Visual defects (folds, stains, noise, etc.)
    - Edge cases (partial scans, rotations, multi-receipt)
    - Configurable scenarios for training/validation

Usage:
    # Quick generation with defaults
    python -m generator.det.run

    # Custom generation
    python -m generator.det.run --num 100 --scenario training_balanced --output data/train_det

    # Validation set (cleaner images)
    python -m generator.det.run --num 100 --scenario validation --output data/val_det
"""

import argparse
import sys
import os
import time
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for module imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from generator.det.generator import (
    GenerationScenario,
    SyntheticInvoiceGenerator,
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Vietnamese invoice dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 balanced training samples
    python -m generator.det.run --num 100 --scenario training_balanced --output data/train_det

    # Generate difficult training data
    python -m generator.det.run --num 500 --scenario training_hard --output data/train_det

    # Generate clean validation data
    python -m generator.det.run --num 50 --scenario validation --output data/val_det

    # Focus on edge cases for robustness testing
    python -m generator.det.run --num 200 --scenario edge_cases_focus --output data/train_det

Available scenarios:
    training_balanced  - Balanced mix (75% normal, 15% edge cases, 10% bad)
    training_hard      - More challenging (50% normal, 35% edge cases, 15% bad)
    validation         - Clean data (95% normal, 5% edge cases)
    edge_cases_focus   - Heavy edge cases (30% normal, 60% edge cases)
    retail_focus       - Focus on supermarket/convenience store receipts
    restaurant_focus   - Focus on restaurant/cafe bills
    formal_invoices    - Focus on official VAT invoices
    pure_random_focus  - 100% random text (for debugging)
    pseudo_focus       - 100% pseudo-generated text
        """
    )

    parser.add_argument("-n", "--num", type=int, default=100, help="Number of samples")
    parser.add_argument("-o", "--output", type=str, default="data/train_det", help="Output directory")
    parser.add_argument("-s", "--scenario", type=str, default="training_balanced",
                        choices=[s.value for s in GenerationScenario], help="Generation scenario")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    if not args.quiet:
        print("=" * 50)
        print("SYNTHETIC INVOICE GENERATOR")
        print("=" * 50)
        print(f"  Samples:    {args.num}")
        print(f"  Output:     {args.output}")
        print(f"  Scenario:   {args.scenario}")
        print("=" * 50)
        print()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialize generator
    scenario = GenerationScenario(args.scenario)
    generator = SyntheticInvoiceGenerator.from_scenario(
        scenario, args.output, args.num
    )

    start_time = time.time()
    results = []
    
    # Generate samples sequentially
    for i in tqdm(range(args.num), desc="Generating", disable=args.quiet):
        try:
            import random
            dice = random.random()
            config = generator.config
            
            if dice < config.blank_ratio:
                result = generator._generate_blank(i)
            elif dice < config.blank_ratio + config.unreadable_ratio:
                result = generator._generate_unreadable(i)
            elif dice < (config.blank_ratio + config.unreadable_ratio + config.edge_case_ratio):
                result = generator._generate_with_edge_case(i)
            else:
                result = generator._generate_realistic(i)
            
            results.append(result)
                
        except Exception as e:
            if not args.quiet:
                print(f"Error generating sample {i}: {e}")

    elapsed = time.time() - start_time

    if not args.quiet:
        print()
        print("=" * 50)
        print(f"SUCCESS: Generated {len(results)} invoices in {elapsed:.2f}s")
        print("=" * 50)

        # Print summary
        type_counts = {}
        layout_counts = {}
        for r in results:
            t = r.get("sample_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

            layout = r.get("layout_type", "unknown")
            if layout != "unknown":
                layout_counts[layout] = layout_counts.get(layout, 0) + 1

        print(f"\nSample Types:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t}: {count} ({100 * count / len(results):.1f}%)")

        if layout_counts:
            print(f"\nLayout Types:")
            for layout, count in sorted(layout_counts.items()):
                print(f"  {layout}: {count} ({100 * count / len(results):.1f}%)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
