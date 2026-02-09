"""
Test script for OCR models: DBNetPP (Detection) and SVTRCTC (Recognition).

Verifies:
1. Input/Output shapes
2. Parameter counts per module
3. Memory usage
4. Forward pass timing
5. Component tests
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.det.dbnet import DBNetPP
from model.rec.svtr_ctc import SVTRCTC


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_params(num):
    """Format parameter count with units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def test_dbnet():
    """Test DBNetPP text detection model."""
    print("=" * 60)
    print("DBNetPP Text Detection Model Test")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    in_channels = 3
    img_size = 640
    
    # Create model
    print("\n[1] Creating DBNetPP model...")
    model = DBNetPP(
        backbone='resnet50',
        pretrained=False,  # Use random initialization for testing
        in_channels=in_channels,
        inner_channels=256,
        k=50,
        dcn=False
    )
    
    # Input tensor
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"    Input shape: {x.shape}")
    
    # Forward pass
    print("\n[2] Running forward pass...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        predictions = model(x)
        elapsed = time.time() - start_time
    
    print(f"    Forward time: {elapsed*1000:.2f} ms")
    
    # Output shapes
    print("\n[3] Output shapes:")
    print(f"    Binary map:       {predictions['binary'].shape}")
    print(f"    Threshold map:    {predictions['thresh'].shape}")
    print(f"    Thresh binary:    {predictions['thresh_binary'].shape}")
    
    # Verify output shape (should be (B, 1, H, W) for each output)
    expected_shape = (batch_size, 1, img_size, img_size)
    assert predictions['binary'].shape == expected_shape, f"Binary shape mismatch: {predictions['binary'].shape}"
    assert predictions['thresh'].shape == expected_shape, f"Thresh shape mismatch: {predictions['thresh'].shape}"
    assert predictions['thresh_binary'].shape == expected_shape, f"Thresh binary shape mismatch: {predictions['thresh_binary'].shape}"
    print("      All output shapes correct!")
    
    # Verify output ranges
    print("\n[4] Output value ranges:")
    print(f"    Binary map:       min={predictions['binary'].min():.4f}, max={predictions['binary'].max():.4f}")
    print(f"    Threshold map:    min={predictions['thresh'].min():.4f}, max={predictions['thresh'].max():.4f}")
    print(f"    Thresh binary:    min={predictions['thresh_binary'].min():.4f}, max={predictions['thresh_binary'].max():.4f}")
    
    # All outputs should be in [0, 1] range
    assert 0 <= predictions['binary'].min() and predictions['binary'].max() <= 1, "Binary map should be in [0, 1]"
    assert 0 <= predictions['thresh'].min() and predictions['thresh'].max() <= 1, "Thresh map should be in [0, 1]"
    assert 0 <= predictions['thresh_binary'].min() and predictions['thresh_binary'].max() <= 1, "Thresh binary should be in [0, 1]"
    print("      All outputs in valid range [0, 1]!")
    
    # Parameter counts
    print("\n[5] Parameter counts:")
    print("-" * 40)
    
    backbone_params = count_parameters(model.backbone)
    neck_params = count_parameters(model.neck)
    head_params = count_parameters(model.head)
    total_params = count_parameters(model)
    
    print(f"    ResNet Backbone:  {format_params(backbone_params):>10} ({backbone_params:>12,})")
    print(f"    FPN-ASF Neck:     {format_params(neck_params):>10} ({neck_params:>12,})")
    print(f"    DB Head:          {format_params(head_params):>10} ({head_params:>12,})")
    print("-" * 40)
    print(f"    TOTAL:            {format_params(total_params):>10} ({total_params:>12,})")
    
    # Memory estimate
    print("\n[6] Memory estimate (FP32):")
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32, in MB
    print(f"    Parameters: {param_memory:.2f} MB")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n[7] CUDA test:")
        device = torch.device("cuda")
        model_cuda = model.to(device)
        x_cuda = x.to(device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model_cuda(x_cuda)
        torch.cuda.synchronize()
        elapsed_cuda = time.time() - start_time
        
        print(f"    CUDA forward time: {elapsed_cuda*1000:.2f} ms")
        print(f"    GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    else:
        print("\n[7] CUDA not available, skipping GPU test.")
    
    print("\n" + "=" * 60)
    print("DBNetPP tests passed!")
    print("=" * 60)



def test_detection_components():
    """Test individual components of DBNetPP."""
    print("\n" + "=" * 60)
    print("DBNetPP Component Tests")
    print("=" * 60)
    
    from model.det.backbone import ResNet
    from model.det.neck import FPN_ASF
    from model.det.head import DBHead
    
    batch_size = 2
    in_channels = 3
    img_size = 640
    
    # Test ResNet backbone
    print("\n[A] ResNet50 Backbone:")
    backbone = ResNet(name='resnet50', pretrained=False, in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    features = backbone(x)
    for i, f in enumerate(features):
        print(f"    Stage {i+1} (C{i+2}): {f.shape}")
    
    # Test FPN-ASF neck
    print("\n[B] FPN-ASF Neck:")
    neck = FPN_ASF(in_channels_list=backbone.out_channels, inner_channels=256)
    neck_features = neck(features)
    print(f"    Input channels:  {backbone.out_channels}")
    print(f"    Output:          {neck_features.shape}")
    
    # Test DB head
    print("\n[C] DB Head:")
    head = DBHead(in_channels=256, k=50)
    predictions = head(neck_features)
    print(f"    Input:           {neck_features.shape}")
    print(f"    Binary output:   {predictions['binary'].shape}")
    print(f"    Thresh output:   {predictions['thresh'].shape}")
    
    print("\n    All detection component tests passed!")

def test_svtr_ctc():
    """Test SVTRCTC text recognition model (SVTRv2)."""
    print("\n" + "=" * 60)
    print("SVTRCTC (SVTRv2) Text Recognition Model Test")
    print("=" * 60)
    
    # Configuration
    batch_size = 4
    in_channels = 3
    img_height = 32
    img_width = 384
    
    # Create model
    print("\n[1] Creating SVTRCTC model (SVTRv2)...")
    model = SVTRCTC(
        img_size=(img_height, None), # Dynamic width
        in_channels=in_channels,
        embed_dim=[192, 256, 512],
        enc_depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        out_channels=512
    )
    
    # Input tensor
    x = torch.randn(batch_size, in_channels, img_height, img_width)
    print(f"    Input shape: {x.shape}")
    
    # Forward pass
    print("\n[2] Running forward pass...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        log_probs = model(x)
        elapsed = time.time() - start_time
    
    print(f"    Forward time: {elapsed*1000:.2f} ms")
    
    # Output shapes
    print("\n[3] Output shapes:")
    print(f"    Log probs: {log_probs.shape}")
    
    # Verify output shape (T, B, num_classes) for CTC
    # T = W/4 typically for SVTR
    T = img_width // 4  
    expected_shape = (T, batch_size, model.tokenizer.num_classes)
    assert log_probs.shape == expected_shape, f"Output shape mismatch: {log_probs.shape} vs {expected_shape}"
    print(f"      Output shape correct! (T={T}, B={batch_size}, V={model.tokenizer.num_classes})")
    
    # Verify output ranges (log probs should be negative)
    print("\n[4] Output value ranges:")
    print(f"    Log probs: min={log_probs.min():.4f}, max={log_probs.max():.4f}")
    assert log_probs.max() <= 0, "Log probs should be <= 0"
    print("      Log probs in valid range!")
    
    # Test with different width (Dynamic Width Check)
    print("\n[5] Testing Dynamic Width:")
    w2 = 256
    x2 = torch.randn(batch_size, in_channels, img_height, w2)
    print(f"    Input shape: {x2.shape}")
    with torch.no_grad():
        log_probs2 = model(x2)
    print(f"    Output shape: {log_probs2.shape}")
    T2 = w2 // 4
    assert log_probs2.shape[0] == T2, f"Dynamic width mismatch: expected T={T2}, got {log_probs2.shape[0]}"
    print(f"      Dynamic width handled correctly!")

    # Decoding test
    print("\n[6] Testing decoding (mock):")
    # Just ensure it runs without error
    _ = model.decode_greedy(x)
    print("      And decode_greedy runs without error.")
    
    # Parameter counts
    print("\n[7] Parameter counts:")
    print("-" * 40)
    
    encoder_params = count_parameters(model.encoder)
    head_params = count_parameters(model.head)
    total_params = count_parameters(model)
    
    print(f"    SVTRv2 Encoder: {format_params(encoder_params):>10} ({encoder_params:>12,})")
    print(f"    CTC Head:       {format_params(head_params):>10} ({head_params:>12,})")
    print("-" * 40)
    print(f"    TOTAL:          {format_params(total_params):>10} ({total_params:>12,})")
    
    print("\n" + "=" * 60)
    print("SVTRCTC (SVTRv2) tests passed!")
    print("=" * 60)


def test_recognition_components():
    """Test individual components of SVTRCTC (SVTRv2)."""
    print("\n" + "=" * 60)
    print("SVTRv2 Component Tests")
    print("=" * 60)
    
    from model.rec.svtr_encoder import SVTREncoder, ConvBNLayer, SVTRStage, PatchMerging
    
    batch_size = 2
    in_channels = 3
    img_height = 32
    img_width = 128
    
    # Test ConvBNLayer
    print("\n[A] ConvBNLayer:")
    conv_layer = ConvBNLayer(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
    x = torch.randn(batch_size, 3, img_height, img_width)
    out = conv_layer(x)
    print(f"    Input:  {x.shape}")
    print(f"    Output: {out.shape}")
    
    # Test SVTR Stage (Local Mixing)
    print("\n[B] SVTRStage (Local Mixing):")
    dim = 64
    stage = SVTRStage(dim=dim, depth=2, num_heads=2, mixing_types=['local', 'local'])
    # Stage expects (B, C, H, W) input
    x_stage = torch.randn(batch_size, dim, 8, 32)
    out_stage = stage(x_stage)
    print(f"    Input:  {x_stage.shape}")
    print(f"    Output: {out_stage.shape}")
    assert x_stage.shape == out_stage.shape, "Stage should preserve shape"
    
    # Test SVTR Encoder
    print("\n[C] SVTREncoder:")
    model = SVTREncoder(
        img_size=(img_height, None),
        in_channels=in_channels,
        embed_dim=[64, 128, 256],
        depth=[2, 2, 2],
        num_heads=[2, 4, 8],
        mixer_types=['local']*2 + ['local']*2 + ['global']*2
    )
    x = torch.randn(batch_size, in_channels, img_height, img_width)
    features = model(x)
    print(f"    Input:  {x.shape}")
    print(f"    Output: {features.shape}") # (B, T, C)
    
    # Expected T = W/4 = 32
    expected_shape = (batch_size, img_width//4, 256) # 256 is last embed_dim (default out_channels)
    assert features.shape == expected_shape, f"Encoder output shape mismatch: {features.shape}"
    print("      Encoder output shape correct!")
    
    print("\n    All recognition component tests passed!")



def test_with_dcn():
    """Test DBNetPP with Deformable Convolution."""
    print("\n" + "=" * 60)
    print("DBNetPP with DCN Test")
    print("=" * 60)
    
    batch_size = 2
    in_channels = 3
    img_size = 640
    
    print("\n[1] Creating DBNetPP with DCN...")
    model = DBNetPP(
        backbone='resnet50',
        pretrained=False,
        in_channels=in_channels,
        dcn=True  # Enable deformable convolution
    )
    
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"    Input shape: {x.shape}")
    
    print("\n[2] Running forward pass with DCN...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        predictions = model(x)
        elapsed = time.time() - start_time
    
    print(f"    Forward time: {elapsed*1000:.2f} ms")
    print(f"    Binary output: {predictions['binary'].shape}")
    
    total_params = count_parameters(model)
    print(f"    Total parameters: {format_params(total_params)} ({total_params:,})")
    
    print("\n    DCN test passed!")


def test_different_backbones():
    """Test DBNetPP with different backbones."""
    print("\n" + "=" * 60)
    print("DBNetPP Backbone Comparison")
    print("=" * 60)
    
    batch_size = 1
    in_channels = 3
    img_size = 640
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    backbones = ['resnet18', 'resnet50']
    
    for backbone_name in backbones:
        print(f"\n[{backbone_name.upper()}]")
        model = DBNetPP(backbone=backbone_name, pretrained=False, in_channels=in_channels)
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions = model(x)
            elapsed = time.time() - start_time
        
        total_params = count_parameters(model)
        param_memory = total_params * 4 / (1024**2)
        
        print(f"    Parameters: {format_params(total_params):>10} ({total_params:>12,})")
        print(f"    Memory:     {param_memory:>10.2f} MB")
        print(f"    Forward:    {elapsed*1000:>10.2f} ms")
        print(f"    Binary out: {predictions['binary'].shape}")
    
    print("\n    Backbone comparison complete!")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "=" * 60)
    print("OCR MODEL TEST SUITE")
    print("=" * 60)
    print("Testing text detection and recognition models")
    print("=" * 60)
    
    # Detection model tests
    test_dbnet()
    test_detection_components()
    test_with_dcn()
    test_different_backbones()
    
    # Recognition model tests
    test_svtr_ctc()
    test_recognition_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    # Calculate final params for summary
    db_model = DBNetPP(backbone='resnet50', pretrained=False)
    db_params = count_parameters(db_model)
    
    rec_model = SVTRCTC(img_size=(32, None))
    rec_params = count_parameters(rec_model)
    
    print("\nSummary:")
    print(f"   DBNetPP (Detection):    Text region detection ({format_params(db_params)})")
    print(f"   SVTRCTC (Recognition):  OCR text recognition  ({format_params(rec_params)})")
    print("   All components verified")
    print("   Memory and timing profiled")
    print("=" * 60)
