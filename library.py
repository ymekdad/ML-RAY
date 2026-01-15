"""
Library Layer

Handles automatic dependency detection, framework abstraction,
and runtime library conflict resolution for diverse ML frameworks.
"""

import importlib
import subprocess
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class FrameworkInfo:
    """Information about detected framework."""
    name: str
    version: str
    model_class: str
    preprocessor_class: str
    requires: List[str]


class LibraryLayer:
    """
    Library Layer for ML-ray.
    
    Provides automatic dependency detection, framework abstraction,
    and conflict resolution for seamless testing across heterogeneous
    ML libraries.
    
    Supported Frameworks:
        - PyTorch / torchvision
        - TensorFlow / Keras
        - Hugging Face Transformers
        - timm (PyTorch Image Models)
        - JAX / Flax
        - scikit-learn
    """
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "timm": {
            "tags": ["timm", "pytorch-image-models"],
            "config_keys": ["architecture", "pretrained_cfg"],
            "model_class": "timm.create_model",
            "preprocessor": "timm.data.create_transform",
        },
        "transformers": {
            "tags": ["transformers", "pytorch", "tf"],
            "config_keys": ["model_type", "architectures", "transformers_version"],
            "model_class": "transformers.AutoModel",
            "preprocessor": "transformers.AutoProcessor",
        },
        "torchvision": {
            "tags": ["torchvision", "pytorch"],
            "config_keys": [],
            "model_class": "torchvision.models",
            "preprocessor": "torchvision.transforms",
        },
        "tensorflow": {
            "tags": ["tensorflow", "tf", "keras"],
            "config_keys": ["tensorflow_version"],
            "model_class": "tensorflow.keras.Model",
            "preprocessor": "tensorflow.keras.preprocessing",
        },
        "jax": {
            "tags": ["jax", "flax"],
            "config_keys": ["flax_version"],
            "model_class": "flax.linen.Module",
            "preprocessor": None,
        },
    }
    
    # Required dependencies for each framework
    FRAMEWORK_DEPENDENCIES = {
        "timm": ["torch", "torchvision", "timm", "huggingface_hub"],
        "transformers": ["torch", "transformers", "tokenizers", "huggingface_hub"],
        "torchvision": ["torch", "torchvision"],
        "tensorflow": ["tensorflow", "keras"],
        "jax": ["jax", "jaxlib", "flax"],
    }
    
    def __init__(self):
        """Initialize Library Layer."""
        self._loaded_frameworks: Dict[str, Any] = {}
        self._framework_cache: Dict[str, str] = {}
        self._detect_installed_frameworks()
    
    def _detect_installed_frameworks(self) -> None:
        """Detect which ML frameworks are installed."""
        frameworks = {
            "torch": "torch",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "transformers": "transformers",
            "timm": "timm",
        }
        
        for name, module in frameworks.items():
            try:
                lib = importlib.import_module(module)
                self._loaded_frameworks[name] = {
                    "module": lib,
                    "version": getattr(lib, "__version__", "unknown"),
                }
            except ImportError:
                pass
        
        logger.debug(f"Detected frameworks: {list(self._loaded_frameworks.keys())}")
    
    def detect_framework(self, model_id: str) -> str:
        """
        Detect the framework used by a model.
        
        Args:
            model_id: Hugging Face model identifier
        
        Returns:
            Framework name (timm, transformers, torchvision, etc.)
        """
        # Check cache
        if model_id in self._framework_cache:
            return self._framework_cache[model_id]
        
        try:
            from huggingface_hub import hf_hub_download, HfApi
            
            api = HfApi()
            model_info = api.model_info(model_id)
            
            # Check tags
            tags = model_info.tags or []
            
            # Check for timm
            if "timm" in tags or model_id.startswith("timm/"):
                framework = "timm"
            
            # Check for transformers
            elif any(t in tags for t in ["transformers", "pytorch", "tf"]):
                framework = "transformers"
            
            # Check config file
            else:
                framework = self._detect_from_config(model_id)
            
            self._framework_cache[model_id] = framework
            return framework
            
        except Exception as e:
            logger.warning(f"Could not detect framework for {model_id}: {e}")
            return "transformers"  # Default fallback
    
    def _detect_from_config(self, model_id: str) -> str:
        """Detect framework from model config file."""
        try:
            from huggingface_hub import hf_hub_download
            import json
            
            config_path = hf_hub_download(model_id, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            
            # Check config keys
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(key in config for key in patterns["config_keys"]):
                    return framework
            
        except Exception:
            pass
        
        return "transformers"
    
    def resolve_dependencies(
        self,
        model_id: str,
        auto_install: bool = False,
    ) -> List[str]:
        """
        Resolve and optionally install dependencies for a model.
        
        Args:
            model_id: Model identifier
            auto_install: Automatically install missing dependencies
        
        Returns:
            List of resolved dependencies
        """
        framework = self.detect_framework(model_id)
        required = self.FRAMEWORK_DEPENDENCIES.get(framework, [])
        
        missing = []
        for dep in required:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            if auto_install:
                logger.info(f"Installing missing dependencies: {missing}")
                self._install_packages(missing)
            else:
                logger.warning(f"Missing dependencies for {model_id}: {missing}")
        
        return required
    
    def _install_packages(self, packages: List[str]) -> None:
        """Install packages via pip."""
        for package in packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "-q"
                ])
                logger.info(f"Installed: {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
    
    def get_model_loader(self, framework: str) -> callable:
        """
        Get the appropriate model loader function for a framework.
        
        Args:
            framework: Framework name
        
        Returns:
            Model loader function
        """
        loaders = {
            "timm": self._load_timm_model,
            "transformers": self._load_transformers_model,
            "torchvision": self._load_torchvision_model,
            "tensorflow": self._load_tensorflow_model,
        }
        
        return loaders.get(framework, self._load_transformers_model)
    
    def _load_timm_model(
        self,
        model_id: str,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a timm model."""
        import timm
        
        # Extract model name from HF ID
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        
        model = timm.create_model(model_name, pretrained=True, **kwargs)
        
        # Get preprocessing config
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        
        return model, transform
    
    def _load_transformers_model(
        self,
        model_id: str,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a transformers model."""
        from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
        from transformers import AutoModelForImageClassification
        
        try:
            model = AutoModelForImageClassification.from_pretrained(
                model_id, **kwargs
            )
        except Exception:
            model = AutoModel.from_pretrained(model_id, **kwargs)
        
        try:
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            processor = AutoFeatureExtractor.from_pretrained(model_id)
        
        return model, processor
    
    def _load_torchvision_model(
        self,
        model_id: str,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a torchvision model."""
        import torchvision.models as models
        import torchvision.transforms as transforms
        
        model_name = model_id.split("/")[-1].lower()
        model_fn = getattr(models, model_name, None)
        
        if model_fn is None:
            raise ValueError(f"Unknown torchvision model: {model_name}")
        
        model = model_fn(pretrained=True, **kwargs)
        
        # Standard ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return model, transform
    
    def _load_tensorflow_model(
        self,
        model_id: str,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load a TensorFlow/Keras model."""
        import tensorflow as tf
        
        model = tf.keras.models.load_model(model_id, **kwargs)
        
        # Standard preprocessing
        preprocessor = tf.keras.applications.imagenet_utils.preprocess_input
        
        return model, preprocessor
    
    def get_framework_info(self, framework: str) -> FrameworkInfo:
        """
        Get information about a framework.
        
        Args:
            framework: Framework name
        
        Returns:
            FrameworkInfo object
        """
        if framework not in self._loaded_frameworks:
            raise ValueError(f"Framework not loaded: {framework}")
        
        info = self._loaded_frameworks[framework]
        patterns = self.FRAMEWORK_PATTERNS.get(framework, {})
        
        return FrameworkInfo(
            name=framework,
            version=info["version"],
            model_class=patterns.get("model_class", "unknown"),
            preprocessor_class=patterns.get("preprocessor", "unknown"),
            requires=self.FRAMEWORK_DEPENDENCIES.get(framework, []),
        )
    
    def list_available_frameworks(self) -> List[str]:
        """List all available (installed) frameworks."""
        return list(self._loaded_frameworks.keys())
    
    def __repr__(self) -> str:
        return f"LibraryLayer(frameworks={self.list_available_frameworks()})"
