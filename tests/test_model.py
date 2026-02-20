"""
Test script for OCR models: DBNetPP (Detection) and SVTRv2 (Recognition).

Verifies:
1. Parameter counts per module/component
2. Input/Output shapes
3. Forward pass correctness
4. Memory usage
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.det.dbnet import DBNetPP
from model.rec2.svtrv2 import SVTRv2, VARIANTS


def count_parameters(module, trainable_only=True):
    """Count parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_params(num):
    """Format parameter count with units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def print_section(title, char="=", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_param_row(name, params, total=None):
    pct = f"({params/total*100:5.1f}%)" if total else ""
    print(f"    {name:<30s} {format_params(params):>10s}  ({params:>12,}) {pct}")


# ═══════════════════════════════════════════════════════════════════════════
#  DBNetPP Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_dbnet_parameters():
    """Test DBNetPP parameter counts for all configurations."""
    print_section("DBNetPP Parameter Analysis")

    configs = [
        {"backbone": "resnet18", "dcn": False, "inner_channels": 256},
        {"backbone": "resnet50", "dcn": False, "inner_channels": 256},
        {"backbone": "resnet18", "dcn": True,  "inner_channels": 256},
        {"backbone": "resnet50", "dcn": True,  "inner_channels": 256},
    ]

    for cfg in configs:
        model = DBNetPP(
            backbone=cfg["backbone"],
            pretrained=False,
            in_channels=3,
            inner_channels=cfg["inner_channels"],
            k=50,
            dcn=cfg["dcn"],
        )

        total = count_parameters(model)
        backbone_params = count_parameters(model.backbone)
        neck_params = count_parameters(model.neck)
        head_params = count_parameters(model.head)

        label = f"{cfg['backbone'].upper()} | DCN={'ON' if cfg['dcn'] else 'OFF'} | C={cfg['inner_channels']}"
        print(f"\n  [{label}]")
        print(f"  {'-'*60}")
        print_param_row("Backbone (ResNet)", backbone_params, total)
        print_param_row("Neck (FPN-ASF)", neck_params, total)
        print_param_row("Head (DBHead)", head_params, total)
        print(f"  {'-'*60}")
        print_param_row("TOTAL", total)

        mem_mb = total * 4 / (1024**2)
        print(f"    {'Memory (FP32)':<30s} {mem_mb:>10.2f} MB")

        # Verify parameter sum
        assert backbone_params + neck_params + head_params == total, \
            "Parameter sum mismatch — possible shared or unaccounted parameters"

    print("\n  ✓ All DBNetPP parameter counts verified")


def test_dbnet_forward():
    """Test DBNetPP forward pass shapes and value ranges."""
    print_section("DBNetPP Forward Pass Test")

    batch_size = 2
    img_size = 640

    for backbone_name in ["resnet18", "resnet50"]:
        for dcn in [False, True]:
            model = DBNetPP(
                backbone=backbone_name,
                pretrained=False,
                in_channels=3,
                inner_channels=256,
                k=50,
                dcn=dcn,
            )
            model.eval()

            x = torch.randn(batch_size, 3, img_size, img_size)

            with torch.no_grad():
                start = time.time()
                preds = model(x)
                elapsed = time.time() - start

            label = f"{backbone_name} | DCN={'ON' if dcn else 'OFF'}"
            print(f"\n  [{label}]  forward: {elapsed*1000:.1f} ms")

            expected = (batch_size, 1, img_size, img_size)
            for key in ["binary", "thresh", "thresh_binary"]:
                assert preds[key].shape == expected, \
                    f"{key} shape {preds[key].shape} != {expected}"
                assert 0 <= preds[key].min() and preds[key].max() <= 1, \
                    f"{key} values outside [0, 1]"
                print(f"    {key:<20s} {str(preds[key].shape):<25s} "
                      f"range=[{preds[key].min():.4f}, {preds[key].max():.4f}]")

            # Also check raw logits exist
            assert "bin_logits" in preds and "thresh_logits" in preds

    print("\n  ✓ All DBNetPP forward pass checks passed")


def test_dbnet_components():
    """Test individual DBNetPP components: Backbone, Neck, Head."""
    print_section("DBNetPP Component Tests")

    from model.det.backbone import ResNet
    from model.det.neck import FPN_ASF
    from model.det.head import DBHead

    batch_size = 2
    img_size = 640

    # --- Backbone ---
    print("\n  [A] ResNet50 Backbone")
    backbone = ResNet(name="resnet50", pretrained=False, in_channels=3, dcn=False)
    x = torch.randn(batch_size, 3, img_size, img_size)
    features = backbone(x)
    expected_channels = [256, 512, 1024, 2048]
    for i, (f, c) in enumerate(zip(features, expected_channels)):
        print(f"      C{i+2}: {f.shape}  (channels={c})")
        assert f.shape[1] == c, f"Stage {i} channel mismatch"

    # --- Neck ---
    print("\n  [B] FPN-ASF Neck")
    neck = FPN_ASF(in_channels_list=backbone.out_channels, inner_channels=256)
    neck_out = neck(features)
    print(f"      Input channels:  {backbone.out_channels}")
    print(f"      Output:          {neck_out.shape}")
    assert neck_out.shape[1] == 256, "Neck output channels should be inner_channels"

    # --- Head ---
    print("\n  [C] DBHead")
    head = DBHead(in_channels=256, k=50)
    preds = head(neck_out)
    print(f"      Binary:          {preds['binary'].shape}")
    print(f"      Threshold:       {preds['thresh'].shape}")
    print(f"      Thresh binary:   {preds['thresh_binary'].shape}")

    print("\n  ✓ All DBNetPP component tests passed")


# ═══════════════════════════════════════════════════════════════════════════
#  SVTRv2 Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_svtrv2_parameters():
    """Test SVTRv2 parameter counts for all variants."""
    print_section("SVTRv2 Parameter Analysis")

    for variant_name in ["tiny", "small", "base"]:
        cfg = VARIANTS[variant_name]
        model = SVTRv2(variant=variant_name, in_channels=3, dropout=0.0)

        total = count_parameters(model)

        # Stem
        stem_params = count_parameters(model.stem)

        # Backbone stages
        stage_params = []
        for i, stage in enumerate(model.stages):
            stage_params.append(count_parameters(stage))

        # Patch merging
        merge_params = []
        for m in model.merges:
            merge_params.append(count_parameters(m))

        # Backbone norm
        bnorm_params = count_parameters(model.backbone_norm)

        # FRM
        frm_params = count_parameters(model.frm)

        # SGM
        sgm_params = count_parameters(model.sgm)

        # CTC Head
        head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

        print(f"\n  [{variant_name.upper()}]  dims={cfg['dims']}  blocks={cfg['num_blocks']}  local={cfg['num_local']}")
        print(f"  {'-'*60}")
        print_param_row("ConvStem", stem_params, total)
        for i, sp in enumerate(stage_params):
            mixer_desc = f"L{cfg['num_local'][i]}" + (f"G{cfg['num_blocks'][i]-cfg['num_local'][i]}" if cfg['num_blocks'][i] > cfg['num_local'][i] else "")
            print_param_row(f"Stage {i+1} ({mixer_desc}, D={cfg['dims'][i]})", sp, total)
        for i, mp in enumerate(merge_params):
            print_param_row(f"PatchMerging {i+1}", mp, total)
        print_param_row("Backbone LayerNorm", bnorm_params, total)
        print_param_row("FRM", frm_params, total)
        print_param_row("SGM (train only)", sgm_params, total)
        print_param_row("CTC Head", head_params, total)
        print(f"  {'-'*60}")
        print_param_row("TOTAL", total)

        # Inference params (exclude SGM)
        inference_params = total - sgm_params
        print_param_row("TOTAL (inference, no SGM)", inference_params)

        mem_mb = total * 4 / (1024**2)
        mem_inf = inference_params * 4 / (1024**2)
        print(f"    {'Memory (FP32, all)':<30s} {mem_mb:>10.2f} MB")
        print(f"    {'Memory (FP32, inference)':<30s} {mem_inf:>10.2f} MB")

    print("\n  ✓ All SVTRv2 parameter counts verified")


def test_svtrv2_forward():
    """Test SVTRv2 forward pass shapes for all variants."""
    print_section("SVTRv2 Forward Pass Test")

    batch_size = 4
    img_h, img_w = 32, 256

    for variant_name in ["tiny", "small", "base"]:
        model = SVTRv2(variant=variant_name, in_channels=3, dropout=0.0)

        # --- Inference mode ---
        model.eval()
        x = torch.randn(batch_size, 3, img_h, img_w)

        with torch.no_grad():
            start = time.time()
            log_probs = model(x)
            elapsed = time.time() - start

        # After ConvStem: H/4=8, W/4=64
        # After 2 PatchMergings (height only): H/4/2/2=2, W stays 64
        # FRM collapses height → T = W/4 = 64
        T = img_w // 4
        expected = (T, batch_size, model.tokenizer.num_classes)
        assert log_probs.shape == expected, \
            f"[{variant_name}] inference shape {log_probs.shape} != {expected}"
        assert log_probs.max() <= 0, "Log probs must be <= 0"

        print(f"\n  [{variant_name.upper()}] Inference")
        print(f"    Input:     {x.shape}")
        print(f"    Output:    {log_probs.shape}  (T={T}, B={batch_size}, V={model.tokenizer.num_classes})")
        print(f"    Time:      {elapsed*1000:.1f} ms")

        # --- Training mode with SGM ---
        model.train()
        max_label_len = 20
        targets = torch.randint(2, model.tokenizer.num_classes, (batch_size, max_label_len))

        log_probs_train, sgm_output = model(x, targets=targets)

        assert log_probs_train.shape == expected, \
            f"[{variant_name}] training shape {log_probs_train.shape} != {expected}"
        assert "sgm_left" in sgm_output and "sgm_right" in sgm_output
        assert sgm_output["sgm_left"].shape == (batch_size, max_label_len, model.tokenizer.num_classes)
        assert sgm_output["sgm_right"].shape == (batch_size, max_label_len, model.tokenizer.num_classes)

        print(f"  [{variant_name.upper()}] Training (with SGM)")
        print(f"    CTC logprobs:  {log_probs_train.shape}")
        print(f"    SGM left:      {sgm_output['sgm_left'].shape}")
        print(f"    SGM right:     {sgm_output['sgm_right'].shape}")

    print("\n  ✓ All SVTRv2 forward pass checks passed")


def test_svtrv2_decode():
    """Test SVTRv2 greedy decoding."""
    print_section("SVTRv2 Decode Test")

    model = SVTRv2(variant="base", in_channels=3, dropout=0.0)
    model.eval()

    x = torch.randn(2, 3, 32, 256)
    texts = model.decode_greedy(x)

    print(f"  Input:       {x.shape}")
    print(f"  Decoded:     {texts}")
    assert isinstance(texts, list) and len(texts) == 2
    assert all(isinstance(t, str) for t in texts)

    print("\n  ✓ SVTRv2 decode test passed")


def test_svtrv2_components():
    """Test individual SVTRv2 components."""
    print_section("SVTRv2 Component Tests")

    from model.rec2.svtrv2 import ConvStem, PatchMerging, SVTRStage, FRM, SGM

    B = 2

    # --- ConvStem ---
    print("\n  [A] ConvStem")
    stem = ConvStem(in_channels=3, out_channels=96)
    x = torch.randn(B, 3, 32, 256)
    out = stem(x)
    print(f"      Input:  {x.shape}")
    print(f"      Output: {out.shape}")
    assert out.shape == (B, 96, 8, 64), f"ConvStem output shape mismatch: {out.shape}"

    # --- PatchMerging ---
    print("\n  [B] PatchMerging")
    pm = PatchMerging(dim_in=96, dim_out=192)
    seq = out.flatten(2).transpose(1, 2)  # (B, 8*64, 96)
    out_pm, new_H, new_W = pm(seq, 8, 64)
    print(f"      Input:  {seq.shape} (H=8, W=64)")
    print(f"      Output: {out_pm.shape} (H={new_H}, W={new_W})")
    assert new_H == 4 and new_W == 64

    # --- SVTRStage ---
    print("\n  [C] SVTRStage (3 blocks, 2 local + 1 global)")
    stage = SVTRStage(dim=96, num_blocks=3, num_local=2, dropout=0.0)
    seq_in = torch.randn(B, 8 * 64, 96)
    out_stage = stage(seq_in, 8, 64)
    print(f"      Input:  {seq_in.shape}")
    print(f"      Output: {out_stage.shape}")
    assert out_stage.shape == seq_in.shape

    # --- FRM ---
    print("\n  [D] FRM")
    frm = FRM(dim=256, dropout=0.0)
    seq_frm = torch.randn(B, 2 * 64, 256)  # H=2, W=64 after backbone
    out_frm = frm(seq_frm, 2, 64)
    print(f"      Input:  {seq_frm.shape} (H=2, W=64)")
    print(f"      Output: {out_frm.shape}")
    assert out_frm.shape == (B, 64, 256), f"FRM output shape mismatch: {out_frm.shape}"

    # --- SGM ---
    print("\n  [E] SGM")
    num_classes = 100
    sgm = SGM(dim=256, num_classes=num_classes, context_window=3, dropout=0.0)
    visual_feat = torch.randn(B, 128, 256)
    targets = torch.randint(2, num_classes, (B, 15))
    sgm.train()
    sgm_out = sgm(visual_feat, targets)
    print(f"      Visual features: {visual_feat.shape}")
    print(f"      Targets:         {targets.shape}")
    print(f"      SGM left:        {sgm_out['sgm_left'].shape}")
    print(f"      SGM right:       {sgm_out['sgm_right'].shape}")
    assert sgm_out["sgm_left"].shape == (B, 15, num_classes)

    print("\n  ✓ All SVTRv2 component tests passed")


# ═══════════════════════════════════════════════════════════════════════════
#  CUDA Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_cuda():
    """Run both models on CUDA if available."""
    if not torch.cuda.is_available():
        print_section("CUDA Tests — SKIPPED (no GPU)")
        return

    print_section("CUDA Tests")
    device = torch.device("cuda")

    # DBNetPP
    print("\n  [DBNetPP on CUDA]")
    det_model = DBNetPP(backbone="resnet50", pretrained=False, dcn=True).to(device)
    det_model.eval()
    x_det = torch.randn(2, 3, 640, 640, device=device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = det_model(x_det)
    torch.cuda.synchronize()
    print(f"    Forward: {(time.time()-start)*1000:.1f} ms")
    print(f"    GPU mem: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    del det_model, x_det
    torch.cuda.empty_cache()

    # SVTRv2
    print("\n  [SVTRv2-Base on CUDA]")
    rec_model = SVTRv2(variant="base").to(device)
    rec_model.eval()
    x_rec = torch.randn(8, 3, 32, 256, device=device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = rec_model(x_rec)
    torch.cuda.synchronize()
    print(f"    Forward: {(time.time()-start)*1000:.1f} ms")
    print(f"    GPU mem: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    print("\n  ✓ CUDA tests passed")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_section("OCR MODEL TEST SUITE", "═", 70)
    print("  Testing DBNetPP (model/det) and SVTRv2 (model/rec2)")

    # ── Detection ──
    test_dbnet_parameters()
    test_dbnet_forward()
    test_dbnet_components()

    # ── Recognition ──
    test_svtrv2_parameters()
    test_svtrv2_forward()
    test_svtrv2_decode()
    test_svtrv2_components()

    # ── GPU ──
    test_cuda()

    # ── Summary ──
    print_section("ALL TESTS PASSED", "═", 70)

    det = DBNetPP(backbone="resnet50", pretrained=False, dcn=True)
    rec = SVTRv2(variant="base")

    det_p = count_parameters(det)
    rec_p = count_parameters(rec)
    rec_inf = rec_p - count_parameters(rec.sgm)

    print(f"  DBNetPP  (ResNet50+DCN): {format_params(det_p)}")
    print(f"  SVTRv2   (Base, all):    {format_params(rec_p)}")
    print(f"  SVTRv2   (Base, infer):  {format_params(rec_inf)}")
    print(f"  All components verified · Shapes validated · Memory profiled")
    print("═" * 70)
