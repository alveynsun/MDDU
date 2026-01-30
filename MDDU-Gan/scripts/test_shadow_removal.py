#!/usr/bin/env python3
"""
Basic sanity test for shadow removal model.

Tests:
1. Network forward passes with dummy data
2. Log space transforms (no NaN)
3. Model with/without masks
4. Integration with existing framework
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import networks
from models.shadow_removal_model import ShadowRemovalModel
from options.train_options import TrainOptions


def test_hypernet():
    """Test HyperNet forward pass."""
    print("\n=== Testing HyperNet ===")
    
    # Create network
    hypernet = networks.HyperNet(input_nc=4, ngf=64)
    hypernet.eval()
    
    # Create dummy input
    batch_size = 2
    shadow = torch.randn(batch_size, 3, 256, 256)
    mask = torch.ones(batch_size, 1, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        alpha, beta = hypernet(shadow, mask)
    
    # Check outputs
    assert alpha.shape == (batch_size, 1, 256, 256), f"Alpha shape mismatch: {alpha.shape}"
    assert beta.shape == (batch_size, 1, 256, 256), f"Beta shape mismatch: {beta.shape}"
    assert not torch.isnan(alpha).any(), "Alpha contains NaN"
    assert not torch.isnan(beta).any(), "Beta contains NaN"
    assert (alpha >= 0).all() and (alpha <= 1).all(), "Alpha not in [0, 1]"
    assert (beta >= 0).all() and (beta <= 1).all(), "Beta not in [0, 1]"
    
    print(f"✓ HyperNet output shapes: alpha={alpha.shape}, beta={beta.shape}")
    print(f"✓ Alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"✓ Beta range: [{beta.min():.4f}, {beta.max():.4f}]")
    print("✓ HyperNet test passed!")


def test_anet_jnet():
    """Test A-Net and J-Net forward passes."""
    print("\n=== Testing A-Net and J-Net ===")
    
    # Create networks
    anet = networks.ANet(input_nc=7, output_nc=3, ngf=64)
    jnet = networks.JNet(input_nc=7, output_nc=3, ngf=64)
    anet.eval()
    jnet.eval()
    
    # Create dummy input
    batch_size = 2
    anet_input = torch.randn(batch_size, 7, 256, 256)
    jnet_input = torch.randn(batch_size, 7, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        a_output = anet(anet_input)
        j_output = jnet(jnet_input)
    
    # Check outputs
    assert a_output.shape == (batch_size, 3, 256, 256), f"A-Net shape mismatch: {a_output.shape}"
    assert j_output.shape == (batch_size, 3, 256, 256), f"J-Net shape mismatch: {j_output.shape}"
    assert not torch.isnan(a_output).any(), "A-Net output contains NaN"
    assert not torch.isnan(j_output).any(), "J-Net output contains NaN"
    
    print(f"✓ A-Net output shape: {a_output.shape}")
    print(f"✓ J-Net output shape: {j_output.shape}")
    print("✓ A-Net and J-Net tests passed!")


def test_unfolding_generator():
    """Test UnfoldingGenerator with various configurations."""
    print("\n=== Testing UnfoldingGenerator ===")
    
    batch_size = 2
    shadow = torch.randn(batch_size, 3, 256, 256) * 0.5  # Scale to reasonable range
    mask = torch.ones(batch_size, 1, 256, 256)
    
    # Test with different iteration counts
    for num_iter in [1, 2, 3]:
        print(f"\n  Testing with {num_iter} iterations...")
        
        # Test without weight sharing
        gen = networks.UnfoldingGenerator(
            num_iterations=num_iter,
            ngf=32,  # Reduced for faster testing
            share_weights=False,
            log_eps=1e-4
        )
        gen.eval()
        
        with torch.no_grad():
            output = gen(shadow, mask)
        
        assert output.shape == (batch_size, 3, 256, 256), f"Output shape mismatch: {output.shape}"
        assert not torch.isnan(output).any(), f"Output contains NaN with {num_iter} iterations"
        print(f"  ✓ Without weight sharing: output range [{output.min():.4f}, {output.max():.4f}]")
        
        # Test with weight sharing
        gen_shared = networks.UnfoldingGenerator(
            num_iterations=num_iter,
            ngf=32,
            share_weights=True,
            log_eps=1e-4
        )
        gen_shared.eval()
        
        with torch.no_grad():
            output_shared = gen_shared(shadow, mask)
        
        assert output_shared.shape == (batch_size, 3, 256, 256), f"Output shape mismatch: {output_shared.shape}"
        assert not torch.isnan(output_shared).any(), f"Output contains NaN with weight sharing"
        print(f"  ✓ With weight sharing: output range [{output_shared.min():.4f}, {output_shared.max():.4f}]")
    
    # Test without mask
    print("\n  Testing without mask...")
    gen = networks.UnfoldingGenerator(num_iterations=2, ngf=32)
    gen.eval()
    
    with torch.no_grad():
        output_no_mask = gen(shadow, None)
    
    assert output_no_mask.shape == (batch_size, 3, 256, 256), f"Output shape mismatch: {output_no_mask.shape}"
    assert not torch.isnan(output_no_mask).any(), "Output contains NaN without mask"
    print(f"  ✓ Without mask: output range [{output_no_mask.min():.4f}, {output_no_mask.max():.4f}]")
    
    print("\n✓ UnfoldingGenerator tests passed!")


def test_log_transforms():
    """Test log space transforms for numerical stability."""
    print("\n=== Testing Log Space Transforms ===")
    
    log_eps = 1e-4
    
    # Test with various input ranges - all in valid [-1, 1] range
    test_cases = [
        torch.randn(2, 3, 64, 64).clamp(-1, 1) * 0.1,  # Small values
        torch.randn(2, 3, 64, 64).clamp(-1, 1) * 0.5,  # Medium values
        torch.randn(2, 3, 64, 64).clamp(-1, 1) * 0.9,  # Large values
        torch.ones(2, 3, 64, 64) * 0.5,                # Constant positive
        torch.ones(2, 3, 64, 64) * -0.5,               # Constant negative
    ]
    
    for i, shadow in enumerate(test_cases):
        # Forward transform (same as in UnfoldingGenerator)
        S_normalized = (shadow + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        S_log = torch.log(S_normalized.clamp(min=log_eps) + log_eps)
        
        # Inverse transform (same as in UnfoldingGenerator)
        J_normalized = torch.exp(S_log).clamp(min=0.0) - log_eps
        J_normalized = J_normalized.clamp(min=0.0, max=1.0)
        J_final = J_normalized * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        # Check for NaN
        assert not torch.isnan(S_log).any(), f"S_log contains NaN in test case {i}"
        assert not torch.isnan(J_final).any(), f"J_final contains NaN in test case {i}"
        
        # Check reconstruction (should be near-perfect for values in valid range)
        reconstruction_error = (shadow - J_final).abs().mean()
        print(f"  Test case {i+1}: reconstruction error = {reconstruction_error:.6f}")
        
        # The reconstruction won't be perfect due to clamping and log/exp operations,
        # but should be reasonable for numerical stability check
        assert reconstruction_error < 0.5, f"Very high reconstruction error in test case {i}: {reconstruction_error}"
        
        # Most importantly, check outputs are in valid range
        assert J_final.min() >= -1.0 and J_final.max() <= 1.0, \
            f"Output not in [-1, 1] range: [{J_final.min():.4f}, {J_final.max():.4f}]"
    
    print("✓ Log space transform tests passed!")


def test_shadow_removal_model():
    """Test ShadowRemovalModel integration."""
    print("\n=== Testing ShadowRemovalModel ===")
    
    # Create minimal options for testing
    class MockOpt:
        def __init__(self):
            self.isTrain = True
            self.checkpoints_dir = '/tmp/checkpoints'
            self.name = 'test_shadow_removal'
            self.device = torch.device('cpu')
            self.preprocess = 'resize_and_crop'
            
            # Model options
            self.num_iterations = 2
            self.use_mask = True
            self.lambda_physical = 10.0
            self.lambda_gan = 1.0
            self.log_eps = 1e-4
            self.share_weights = True
            
            # Network options
            self.ngf = 32  # Reduced for testing
            self.ndf = 32
            self.norm = 'batch'
            self.no_dropout = False
            self.init_type = 'normal'
            self.init_gain = 0.02
            self.netD = 'basic'
            self.n_layers_D = 3
            self.output_nc = 3
            self.input_nc = 3
            
            # Optimizer options
            self.lr = 0.0002
            self.beta1 = 0.5
            self.gan_mode = 'lsgan'
    
    opt = MockOpt()
    
    # Create model
    model = ShadowRemovalModel(opt)
    print(f"  Model created with {len(model.model_names)} networks: {model.model_names}")
    print(f"  Loss names: {model.loss_names}")
    print(f"  Visual names: {model.visual_names}")
    
    # Test set_input
    batch_size = 2
    dummy_input = {
        'shadow': torch.randn(batch_size, 3, 256, 256),
        'gt': torch.randn(batch_size, 3, 256, 256),
        'mask': torch.ones(batch_size, 1, 256, 256),
        'shadow_paths': ['test1.png', 'test2.png']
    }
    
    model.set_input(dummy_input)
    assert hasattr(model, 'shadow'), "Model should have 'shadow' attribute"
    assert hasattr(model, 'real'), "Model should have 'real' attribute"
    assert hasattr(model, 'mask'), "Model should have 'mask' attribute"
    print("  ✓ set_input works correctly")
    
    # Test forward pass
    model.forward()
    assert hasattr(model, 'fake'), "Model should have 'fake' attribute after forward"
    assert model.fake.shape == (batch_size, 3, 256, 256), f"Fake shape mismatch: {model.fake.shape}"
    assert not torch.isnan(model.fake).any(), "Fake output contains NaN"
    print(f"  ✓ Forward pass works: output range [{model.fake.min():.4f}, {model.fake.max():.4f}]")
    
    # Test without mask
    dummy_input_no_mask = {
        'shadow': torch.randn(batch_size, 3, 256, 256),
        'gt': torch.randn(batch_size, 3, 256, 256),
        'shadow_paths': ['test1.png', 'test2.png']
    }
    
    model.set_input(dummy_input_no_mask)
    model.forward()
    assert not torch.isnan(model.fake).any(), "Fake output contains NaN without explicit mask"
    print("  ✓ Forward pass works without explicit mask")
    
    # Test backward passes
    model.set_input(dummy_input)
    model.forward()
    
    # Test backward_D
    model.optimizer_D.zero_grad()
    model.backward_D()
    assert hasattr(model, 'loss_D'), "Model should have 'loss_D' after backward_D"
    assert not torch.isnan(model.loss_D), "loss_D is NaN"
    print(f"  ✓ backward_D works: loss_D = {model.loss_D.item():.4f}")
    
    # Test backward_G
    model.optimizer_G.zero_grad()
    model.backward_G()
    assert hasattr(model, 'loss_G'), "Model should have 'loss_G' after backward_G"
    assert not torch.isnan(model.loss_G), "loss_G is NaN"
    print(f"  ✓ backward_G works: loss_G = {model.loss_G.item():.4f}")
    
    print("\n✓ ShadowRemovalModel integration test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Shadow Removal Model Sanity Tests")
    print("=" * 70)
    
    try:
        test_hypernet()
        test_anet_jnet()
        test_unfolding_generator()
        test_log_transforms()
        test_shadow_removal_model()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
