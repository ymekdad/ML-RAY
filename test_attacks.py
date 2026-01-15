"""
Tests for ML-ray Attack Modules

Unit tests for adversarial attack implementations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 112 * 112, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def model():
    """Create test model."""
    return SimpleCNN(num_classes=10)


@pytest.fixture
def sample_batch():
    """Create sample batch."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    return images, labels


class TestMLSecurityModule:
    """Tests for ML Security Module."""
    
    def test_fgsm_attack(self, model, sample_batch):
        """Test FGSM attack generates adversarial examples."""
        from mlray.modules.ml_security import MLSecurityModule
        
        module = MLSecurityModule(device=torch.device("cpu"))
        images, labels = sample_batch
        
        # Create simple dataloader
        class SimpleLoader:
            def __iter__(self):
                yield images, labels
        
        result = module._run_fgsm(
            model=model,
            dataloader=SimpleLoader(),
            config=type('Config', (), {'epsilon': 0.03})(),
        )
        
        assert "attack_success_rate" in result
        assert "metrics" in result
        assert 0 <= result["attack_success_rate"] <= 1
    
    def test_pgd_attack(self, model, sample_batch):
        """Test PGD attack generates adversarial examples."""
        from mlray.modules.ml_security import MLSecurityModule
        
        module = MLSecurityModule(device=torch.device("cpu"))
        images, labels = sample_batch
        
        class SimpleLoader:
            def __iter__(self):
                yield images, labels
        
        result = module._run_pgd(
            model=model,
            dataloader=SimpleLoader(),
            config=type('Config', (), {
                'epsilon': 0.03,
                'alpha': 0.007,
                'steps': 5,
                'random_start': True
            })(),
        )
        
        assert "attack_success_rate" in result
        assert "metrics" in result


class TestLLMSecurityModule:
    """Tests for LLM Security Module."""
    
    def test_prompt_injection_templates(self):
        """Test prompt injection templates are defined."""
        from mlray.modules.llm_security import LLMSecurityModule
        
        module = LLMSecurityModule()
        
        assert len(module.PROMPT_INJECTION_TEMPLATES) > 0
        assert len(module.SYSTEM_LEAKAGE_TEMPLATES) > 0
        assert len(module.MISINFORMATION_TEMPLATES) > 0
    
    def test_supported_attacks(self):
        """Test supported attacks list."""
        from mlray.modules.llm_security import LLMSecurityModule
        
        module = LLMSecurityModule()
        
        assert "prompt_injection" in module.SUPPORTED_ATTACKS
        assert "system_prompt_leakage" in module.SUPPORTED_ATTACKS
        assert "misinformation" in module.SUPPORTED_ATTACKS


class TestPrivacyModule:
    """Tests for Privacy Module."""
    
    def test_membership_inference(self, model):
        """Test membership inference attack."""
        from mlray.modules.privacy import PrivacyExtractionModule
        
        module = PrivacyExtractionModule(device=torch.device("cpu"))
        
        result = module._run_membership_inference(
            model=model,
            device=torch.device("cpu"),
            num_samples=20,
        )
        
        assert "success_rate" in result
        assert "metrics" in result
        assert "attack_auc" in result["metrics"]


class TestMetricsCalculator:
    """Tests for Metrics Calculator."""
    
    def test_accuracy(self):
        """Test accuracy calculation."""
        from mlray.utils.metrics import MetricsCalculator
        
        preds = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 1, 1])
        
        acc = MetricsCalculator.accuracy(preds, labels)
        
        assert acc == 0.8
    
    def test_attack_success_rate(self):
        """Test attack success rate calculation."""
        from mlray.utils.metrics import MetricsCalculator
        
        clean_preds = np.array([0, 1, 2, 3, 4])
        adv_preds = np.array([1, 1, 3, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        
        asr = MetricsCalculator.attack_success_rate(clean_preds, adv_preds, labels)
        
        assert 0 <= asr <= 1
    
    def test_calibration_error(self):
        """Test calibration error calculation."""
        from mlray.utils.metrics import MetricsCalculator
        
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        accuracies = np.array([1, 1, 0, 1, 0])
        
        ece = MetricsCalculator.expected_calibration_error(confidences, accuracies)
        
        assert 0 <= ece <= 1


class TestHardwareLayer:
    """Tests for Hardware Layer."""
    
    def test_device_resolution(self):
        """Test device resolution."""
        from mlray.layers.hardware import HardwareLayer
        
        hw = HardwareLayer(device="cpu")
        
        assert hw.device == torch.device("cpu")
    
    def test_memory_stats(self):
        """Test memory statistics."""
        from mlray.layers.hardware import HardwareLayer
        
        hw = HardwareLayer(device="cpu")
        stats = hw.memory_stats()
        
        assert "device" in stats


class TestLibraryLayer:
    """Tests for Library Layer."""
    
    def test_framework_patterns(self):
        """Test framework detection patterns are defined."""
        from mlray.layers.library import LibraryLayer
        
        lib = LibraryLayer()
        
        assert "timm" in lib.FRAMEWORK_PATTERNS
        assert "transformers" in lib.FRAMEWORK_PATTERNS
    
    def test_available_frameworks(self):
        """Test listing available frameworks."""
        from mlray.layers.library import LibraryLayer
        
        lib = LibraryLayer()
        frameworks = lib.list_available_frameworks()
        
        assert isinstance(frameworks, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
