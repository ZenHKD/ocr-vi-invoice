"""
Quick test to verify beam search implementation.
"""
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.rec.svtr_ctc import SVTRCTC

def test_beam_search():
    print("=" * 60)
    print("Testing Beam Search Implementation")
    print("=" * 60)
    
    # Create model
    model = SVTRCTC(
        img_size=(32, 128),
        in_channels=3
    )
    model.eval()
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 32, 128)
    
    print(f"\nInput shape: {images.shape}")
    
    # Test greedy decoding
    print("\n[1] Greedy Decoding:")
    greedy_results = model.decode_greedy(images)
    for i, text in enumerate(greedy_results):
        print(f"    Sample {i}: '{text}'")
    
    # Test beam search with different beam widths
    beam_widths = [1, 3, 5, 10]
    for bw in beam_widths:
        print(f"\n[2] Beam Search (beam_width={bw}):")
        beam_results = model.decode_beam_search(images, beam_width=bw)
        for i, text in enumerate(beam_results):
            print(f"    Sample {i}: '{text}'")
    
    print("\n" + "=" * 60)
    print("Beam search test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_beam_search()
