"""
Hardware Layer

Manages computational resources including GPU and CPU allocation
for efficient model evaluation and attack execution.
"""

import os
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class GPUInfo:
    """GPU device information."""
    index: int
    name: str
    total_memory: int
    free_memory: int
    utilization: float


class HardwareLayer:
    """
    Hardware Layer for ML-ray.
    
    Manages computational resources for efficient model evaluation,
    including GPU allocation, memory management, and mixed precision support.
    
    Attributes:
        device: Primary computation device
        gpu_ids: List of available GPU indices
        mixed_precision: Whether mixed precision is enabled
    """
    
    def __init__(
        self,
        device: str = "auto",
        gpu_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.9,
        mixed_precision: bool = False,
    ):
        """
        Initialize Hardware Layer.
        
        Args:
            device: Device specification ("auto", "cuda", "cuda:X", "cpu")
            gpu_ids: Specific GPU indices to use
            memory_fraction: Fraction of GPU memory to allocate
            mixed_precision: Enable automatic mixed precision
        """
        self.memory_fraction = memory_fraction
        self.mixed_precision = mixed_precision
        
        # Detect and configure device
        self.device = self._resolve_device(device)
        self.gpu_ids = gpu_ids or self._get_available_gpus()
        
        # Configure memory allocation
        if torch.cuda.is_available():
            self._configure_cuda_memory()
        
        logger.info(f"Hardware Layer initialized: device={self.device}")
    
    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve device specification to torch.device.
        
        Args:
            device: Device specification string
        
        Returns:
            torch.device instance
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
                logger.warning("CUDA not available, falling back to CPU")
        
        return torch.device(device)
    
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU indices."""
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    
    def _configure_cuda_memory(self) -> None:
        """Configure CUDA memory allocation."""
        if self.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction,
                device=self.device.index if self.device.type == "cuda" else 0
            )
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
    
    def configure(
        self,
        gpu_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.9,
        mixed_precision: bool = False,
    ) -> None:
        """
        Reconfigure hardware settings.
        
        Args:
            gpu_ids: GPU indices to use
            memory_fraction: Memory allocation fraction
            mixed_precision: Enable mixed precision
        """
        if gpu_ids:
            self.gpu_ids = gpu_ids
            if gpu_ids:
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
        
        self.memory_fraction = memory_fraction
        self.mixed_precision = mixed_precision
        
        if torch.cuda.is_available():
            self._configure_cuda_memory()
        
        logger.info(f"Hardware reconfigured: gpus={self.gpu_ids}, "
                   f"memory_fraction={memory_fraction}")
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """
        Get information about available GPUs.
        
        Returns:
            List of GPUInfo objects
        """
        if not torch.cuda.is_available():
            return []
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            
            gpu_info.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory=total_memory,
                free_memory=free_memory,
                utilization=1.0 - (free_memory / total_memory),
            ))
        
        return gpu_info
    
    def get_optimal_batch_size(
        self,
        model_size_mb: float,
        input_size: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            model_size_mb: Model size in megabytes
            input_size: Input tensor shape (C, H, W)
            dtype: Data type for inputs
        
        Returns:
            Recommended batch size
        """
        if not torch.cuda.is_available():
            return 16  # Conservative CPU default
        
        # Get free memory
        free_memory, _ = torch.cuda.mem_get_info(self.device.index or 0)
        free_memory_mb = free_memory / (1024 ** 2)
        
        # Estimate memory per sample
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        input_memory_mb = (
            bytes_per_element * 
            torch.tensor(input_size).prod().item() / 
            (1024 ** 2)
        )
        
        # Account for gradients and activations (rough estimate: 3x model + input)
        available_for_batch = free_memory_mb - (3 * model_size_mb)
        
        if available_for_batch <= 0:
            return 1
        
        batch_size = int(available_for_batch / (input_memory_mb * 4))  # 4x for safety
        
        return max(1, min(batch_size, 128))
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared")
    
    def memory_stats(self) -> dict:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {"device": str(self.device)}
        
        if torch.cuda.is_available() and self.device.type == "cuda":
            stats.update({
                "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
                "cached_mb": torch.cuda.memory_reserved(self.device) / (1024 ** 2),
                "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            })
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"HardwareLayer(device={self.device}, "
            f"gpu_ids={self.gpu_ids}, "
            f"mixed_precision={self.mixed_precision})"
        )
