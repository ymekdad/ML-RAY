"""
Model Collectors

Utilities for collecting and cataloging models from public repositories.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from loguru import logger


@dataclass
class ModelMetadata:
    """Model metadata from Hugging Face Hub."""
    model_id: str
    author: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str]
    created_at: Optional[str]
    last_modified: Optional[str]


class HuggingFaceCollector:
    """
    Collector for models from Hugging Face Hub.
    
    Enables programmatic discovery and filtering of public models
    based on various criteria including dataset, task, and popularity.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace Collector.
        
        Args:
            token: Optional HuggingFace API token for private models
        """
        self.token = token
        self._api = None
        logger.debug("HuggingFace Collector initialized")
    
    @property
    def api(self):
        """Lazy-load HuggingFace API."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api
    
    def get_models(
        self,
        dataset: Optional[str] = None,
        task: Optional[str] = None,
        library: Optional[str] = None,
        author: Optional[str] = None,
        min_downloads: int = 0,
        limit: Optional[int] = None,
        sort: str = "downloads",
    ) -> List[str]:
        """
        Get list of model IDs matching criteria.
        
        Args:
            dataset: Filter by training dataset (e.g., "imagenet-1k")
            task: Filter by task (e.g., "image-classification")
            library: Filter by library (e.g., "timm", "transformers")
            author: Filter by author/organization
            min_downloads: Minimum download count
            limit: Maximum number of models to return
            sort: Sort order ("downloads", "likes", "created")
        
        Returns:
            List of model IDs
        """
        logger.info(f"Collecting models: dataset={dataset}, task={task}, "
                   f"library={library}, limit={limit}")
        
        # Build filter kwargs
        filter_kwargs = {}
        
        if task:
            filter_kwargs["task"] = task
        if library:
            filter_kwargs["library"] = library
        if author:
            filter_kwargs["author"] = author
        
        # Fetch models from HuggingFace
        models = list(self.api.list_models(
            sort=sort,
            direction=-1,  # Descending
            **filter_kwargs,
        ))
        
        # Filter by dataset if specified
        if dataset:
            models = [
                m for m in models
                if dataset in (m.tags or []) or
                   (hasattr(m, "cardData") and 
                    m.cardData and 
                    dataset in str(m.cardData.get("datasets", [])))
            ]
        
        # Filter by minimum downloads
        models = [m for m in models if (m.downloads or 0) >= min_downloads]
        
        # Apply limit
        if limit:
            models = models[:limit]
        
        model_ids = [m.modelId for m in models]
        
        logger.info(f"Collected {len(model_ids)} models")
        
        return model_ids
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get detailed metadata for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            ModelMetadata object
        """
        info = self.api.model_info(model_id)
        
        return ModelMetadata(
            model_id=info.modelId,
            author=info.author or "",
            downloads=info.downloads or 0,
            likes=info.likes or 0,
            tags=info.tags or [],
            pipeline_tag=info.pipeline_tag,
            created_at=str(info.created_at) if info.created_at else None,
            last_modified=str(info.last_modified) if info.last_modified else None,
        )
    
    def get_imagenet_models(
        self,
        min_downloads: int = 100,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get models trained on ImageNet-1k.
        
        Args:
            min_downloads: Minimum download count
            limit: Maximum number of models
        
        Returns:
            List of model IDs
        """
        return self.get_models(
            dataset="imagenet-1k",
            task="image-classification",
            min_downloads=min_downloads,
            limit=limit,
        )
    
    def get_qa_models(
        self,
        min_downloads: int = 100,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get Question-Answering models.
        
        Args:
            min_downloads: Minimum download count
            limit: Maximum number of models
        
        Returns:
            List of model IDs
        """
        return self.get_models(
            task="question-answering",
            min_downloads=min_downloads,
            limit=limit,
        )
    
    def get_text_generation_models(
        self,
        min_downloads: int = 1000,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get text generation (LLM) models.
        
        Args:
            min_downloads: Minimum download count
            limit: Maximum number of models
        
        Returns:
            List of model IDs
        """
        return self.get_models(
            task="text-generation",
            min_downloads=min_downloads,
            limit=limit,
        )
    
    def get_top_models(
        self,
        task: Optional[str] = None,
        n: int = 10,
    ) -> List[ModelMetadata]:
        """
        Get top N most downloaded models.
        
        Args:
            task: Optional task filter
            n: Number of models to return
        
        Returns:
            List of ModelMetadata objects
        """
        model_ids = self.get_models(task=task, limit=n, sort="downloads")
        return [self.get_model_metadata(mid) for mid in model_ids]
    
    def export_collection(
        self,
        model_ids: List[str],
        output_path: str,
    ) -> None:
        """
        Export model collection to file.
        
        Args:
            model_ids: List of model IDs
            output_path: Output file path (JSON)
        """
        import json
        
        collection = []
        for model_id in model_ids:
            try:
                metadata = self.get_model_metadata(model_id)
                collection.append({
                    "model_id": metadata.model_id,
                    "author": metadata.author,
                    "downloads": metadata.downloads,
                    "likes": metadata.likes,
                    "tags": metadata.tags,
                    "pipeline_tag": metadata.pipeline_tag,
                })
            except Exception as e:
                logger.warning(f"Failed to get metadata for {model_id}: {e}")
        
        with open(output_path, "w") as f:
            json.dump(collection, f, indent=2)
        
        logger.info(f"Exported {len(collection)} models to {output_path}")
