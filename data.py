"""
Data Layer

Manages datasets, model artifacts, and logging storage for
the ML-ray vulnerability assessment framework.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Iterator
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from loguru import logger


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    num_samples: int
    num_classes: int
    input_shape: Tuple[int, ...]
    split: str


class ImageNetValidationDataset(Dataset):
    """ImageNet-1k validation dataset wrapper."""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize ImageNet validation dataset.
        
        Args:
            root: Root directory containing validation images
            transform: Preprocessing transform
        """
        self.root = Path(root)
        self.transform = transform
        
        # Load image paths and labels
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels."""
        samples = []
        
        # Assume directory structure: root/class_name/image.JPEG
        for class_idx, class_dir in enumerate(sorted(self.root.iterdir())):
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.JPEG"):
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image
        
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DataLayer:
    """
    Data Layer for ML-ray.
    
    Manages storage and access for datasets, model artifacts,
    and execution logs. Provides unified interfaces for loading
    data across different frameworks.
    
    Attributes:
        cache_dir: Root directory for caching
        dataset_dir: Directory for dataset storage
        model_dir: Directory for model artifacts
        log_dir: Directory for execution logs
    """
    
    # Supported datasets
    SUPPORTED_DATASETS = {
        "imagenet-1k": {
            "num_classes": 1000,
            "input_shape": (3, 224, 224),
            "source": "huggingface",
        },
        "imagenet-a": {
            "num_classes": 200,
            "input_shape": (3, 224, 224),
            "source": "huggingface",
        },
        "imagenet-o": {
            "num_classes": 200,
            "input_shape": (3, 224, 224),
            "source": "huggingface",
        },
        "cifar10": {
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "source": "torchvision",
        },
        "cifar100": {
            "num_classes": 100,
            "input_shape": (3, 32, 32),
            "source": "torchvision",
        },
    }
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        dataset_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize Data Layer.
        
        Args:
            cache_dir: Root cache directory
            dataset_dir: Dataset storage directory
            model_dir: Model artifact directory
            log_dir: Log storage directory
        """
        self.cache_dir = Path(cache_dir)
        self.dataset_dir = Path(dataset_dir or self.cache_dir / "datasets")
        self.model_dir = Path(model_dir or self.cache_dir / "models")
        self.log_dir = Path(log_dir or self.cache_dir / "logs")
        
        # Create directories
        for dir_path in [self.dataset_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._dataset_cache: Dict[str, Dataset] = {}
        
        logger.debug(f"Data Layer initialized: cache_dir={self.cache_dir}")
    
    def load_dataset(
        self,
        name: str,
        split: str = "validation",
        transform: Optional[Callable] = None,
    ) -> Dataset:
        """
        Load a dataset.
        
        Args:
            name: Dataset name (e.g., "imagenet-1k")
            split: Dataset split ("train", "validation", "test")
            transform: Preprocessing transform
        
        Returns:
            PyTorch Dataset
        """
        cache_key = f"{name}_{split}"
        
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        logger.info(f"Loading dataset: {name} ({split})")
        
        if name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        
        dataset_info = self.SUPPORTED_DATASETS[name]
        
        if dataset_info["source"] == "huggingface":
            dataset = self._load_huggingface_dataset(name, split, transform)
        elif dataset_info["source"] == "torchvision":
            dataset = self._load_torchvision_dataset(name, split, transform)
        else:
            raise ValueError(f"Unknown dataset source: {dataset_info['source']}")
        
        self._dataset_cache[cache_key] = dataset
        
        return dataset
    
    def _load_huggingface_dataset(
        self,
        name: str,
        split: str,
        transform: Optional[Callable],
    ) -> Dataset:
        """Load dataset from Hugging Face."""
        from datasets import load_dataset as hf_load_dataset
        
        # Map split names
        split_map = {
            "validation": "validation",
            "val": "validation",
            "test": "test",
            "train": "train",
        }
        hf_split = split_map.get(split, split)
        
        # Load dataset
        dataset = hf_load_dataset(
            name,
            split=hf_split,
            cache_dir=str(self.dataset_dir),
        )
        
        # Wrap for PyTorch compatibility
        return HuggingFaceDatasetWrapper(dataset, transform)
    
    def _load_torchvision_dataset(
        self,
        name: str,
        split: str,
        transform: Optional[Callable],
    ) -> Dataset:
        """Load dataset from torchvision."""
        import torchvision.datasets as datasets
        
        is_train = split == "train"
        
        dataset_classes = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }
        
        if name not in dataset_classes:
            raise ValueError(f"Torchvision dataset not found: {name}")
        
        dataset = dataset_classes[name](
            root=str(self.dataset_dir),
            train=is_train,
            download=True,
            transform=transform,
        )
        
        return dataset
    
    def get_dataloader(
        self,
        dataset: str,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        preprocessor: Optional[Callable] = None,
        num_workers: int = 4,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Get a DataLoader for a dataset.
        
        Args:
            dataset: Dataset name or Dataset instance
            batch_size: Batch size
            num_samples: Limit number of samples (None for all)
            preprocessor: Preprocessing function
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data
        
        Returns:
            PyTorch DataLoader
        """
        if isinstance(dataset, str):
            ds = self.load_dataset(dataset, transform=preprocessor)
        else:
            ds = dataset
        
        # Subset if num_samples specified
        if num_samples and num_samples < len(ds):
            indices = np.random.choice(len(ds), num_samples, replace=False)
            ds = Subset(ds, indices)
        
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    def save_model_artifact(
        self,
        model_id: str,
        artifact_name: str,
        data: Any,
    ) -> Path:
        """
        Save a model artifact.
        
        Args:
            model_id: Model identifier
            artifact_name: Artifact name
            data: Data to save
        
        Returns:
            Path to saved artifact
        """
        model_dir = self.model_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = model_dir / artifact_name
        
        if isinstance(data, (dict, list)):
            with open(artifact_path.with_suffix(".json"), "w") as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, torch.Tensor):
            torch.save(data, artifact_path.with_suffix(".pt"))
        elif isinstance(data, np.ndarray):
            np.save(artifact_path.with_suffix(".npy"), data)
        else:
            with open(artifact_path, "wb") as f:
                import pickle
                pickle.dump(data, f)
        
        logger.debug(f"Saved artifact: {artifact_path}")
        return artifact_path
    
    def load_model_artifact(
        self,
        model_id: str,
        artifact_name: str,
    ) -> Any:
        """
        Load a model artifact.
        
        Args:
            model_id: Model identifier
            artifact_name: Artifact name
        
        Returns:
            Loaded artifact data
        """
        model_dir = self.model_dir / model_id.replace("/", "_")
        
        # Try different extensions
        for ext in [".json", ".pt", ".npy", ""]:
            artifact_path = model_dir / f"{artifact_name}{ext}"
            if artifact_path.exists():
                if ext == ".json":
                    with open(artifact_path) as f:
                        return json.load(f)
                elif ext == ".pt":
                    return torch.load(artifact_path)
                elif ext == ".npy":
                    return np.load(artifact_path)
                else:
                    with open(artifact_path, "rb") as f:
                        import pickle
                        return pickle.load(f)
        
        raise FileNotFoundError(f"Artifact not found: {model_id}/{artifact_name}")
    
    def log_experiment(
        self,
        experiment_id: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log experiment data.
        
        Args:
            experiment_id: Unique experiment identifier
            data: Experiment data to log
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            **data,
        }
        
        log_file = self.log_dir / f"{experiment_id}.json"
        
        # Append to existing or create new
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
            logs.append(log_entry)
        else:
            logs = [log_entry]
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    
    def get_dataset_info(self, name: str) -> DatasetInfo:
        """
        Get information about a dataset.
        
        Args:
            name: Dataset name
        
        Returns:
            DatasetInfo object
        """
        if name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        
        info = self.SUPPORTED_DATASETS[name]
        
        return DatasetInfo(
            name=name,
            num_samples=-1,  # Unknown until loaded
            num_classes=info["num_classes"],
            input_shape=info["input_shape"],
            split="validation",
        )
    
    def __repr__(self) -> str:
        return (
            f"DataLayer(dataset_dir={self.dataset_dir}, "
            f"model_dir={self.model_dir}, log_dir={self.log_dir})"
        )


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper to make Hugging Face datasets compatible with PyTorch."""
    
    def __init__(
        self,
        dataset,
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        
        # Handle different column names
        if "image" in item:
            image = item["image"]
        elif "img" in item:
            image = item["img"]
        else:
            image = item[list(item.keys())[0]]
        
        if "label" in item:
            label = item["label"]
        elif "labels" in item:
            label = item["labels"]
        else:
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
