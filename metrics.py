"""
Metrics Calculator

Utility functions for computing evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MetricsCalculator:
    """
    Calculator for various ML evaluation metrics.
    
    Includes standard classification metrics and adversarial robustness metrics.
    """
    
    @staticmethod
    def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy."""
        return np.mean(predictions == labels)
    
    @staticmethod
    def precision_recall_f1(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: Optional[int] = None,
        average: str = "macro",
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1-score.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            num_classes: Number of classes
            average: Averaging method ("macro", "micro", "weighted")
        
        Returns:
            Tuple of (precision, recall, f1)
        """
        if num_classes is None:
            num_classes = max(np.max(predictions), np.max(labels)) + 1
        
        precisions = []
        recalls = []
        supports = []
        
        for c in range(num_classes):
            pred_c = predictions == c
            label_c = labels == c
            
            tp = np.sum(pred_c & label_c)
            fp = np.sum(pred_c & ~label_c)
            fn = np.sum(~pred_c & label_c)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            supports.append(np.sum(label_c))
        
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
        elif average == "weighted":
            total_support = sum(supports)
            precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
        else:  # micro
            total_tp = sum(
                np.sum((predictions == c) & (labels == c))
                for c in range(num_classes)
            )
            total_fp = sum(
                np.sum((predictions == c) & (labels != c))
                for c in range(num_classes)
            )
            total_fn = sum(
                np.sum((predictions != c) & (labels == c))
                for c in range(num_classes)
            )
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    @staticmethod
    def attack_success_rate(
        clean_predictions: np.ndarray,
        adversarial_predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute attack success rate.
        
        Attack is successful if clean prediction was correct but
        adversarial prediction is wrong.
        
        Args:
            clean_predictions: Predictions on clean samples
            adversarial_predictions: Predictions on adversarial samples
            labels: True labels
        
        Returns:
            Attack success rate
        """
        clean_correct = clean_predictions == labels
        adv_wrong = adversarial_predictions != labels
        
        successful_attacks = clean_correct & adv_wrong
        
        # Only count samples that were correctly classified initially
        return np.sum(successful_attacks) / np.sum(clean_correct) if np.sum(clean_correct) > 0 else 0
    
    @staticmethod
    def confidence_drop(
        clean_confidences: np.ndarray,
        adversarial_confidences: np.ndarray,
    ) -> float:
        """
        Compute average confidence drop due to adversarial perturbation.
        
        Args:
            clean_confidences: Confidences on clean samples
            adversarial_confidences: Confidences on adversarial samples
        
        Returns:
            Average confidence drop
        """
        return np.mean(clean_confidences - adversarial_confidences)
    
    @staticmethod
    def transfer_success_rate(
        source_adversarial_success: np.ndarray,
        target_adversarial_success: np.ndarray,
    ) -> float:
        """
        Compute transfer attack success rate.
        
        Args:
            source_adversarial_success: Success on source model
            target_adversarial_success: Success on target model
        
        Returns:
            Transfer success rate
        """
        # Transfer success: attack worked on source AND target
        both_success = source_adversarial_success & target_adversarial_success
        
        return np.sum(both_success) / np.sum(source_adversarial_success) \
            if np.sum(source_adversarial_success) > 0 else 0
    
    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        num_bins: int = 15,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            confidences: Model confidence scores
            accuracies: Binary accuracy indicators
            num_bins: Number of calibration bins
        
        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    @staticmethod
    def perturbation_statistics(
        clean_samples: np.ndarray,
        adversarial_samples: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute statistics about adversarial perturbations.
        
        Args:
            clean_samples: Original samples
            adversarial_samples: Perturbed samples
        
        Returns:
            Dictionary with perturbation statistics
        """
        perturbations = adversarial_samples - clean_samples
        
        return {
            "mean_l2_norm": np.mean(np.linalg.norm(
                perturbations.reshape(perturbations.shape[0], -1), axis=1
            )),
            "mean_linf_norm": np.mean(np.max(
                np.abs(perturbations.reshape(perturbations.shape[0], -1)), axis=1
            )),
            "mean_perturbation": np.mean(np.abs(perturbations)),
            "max_perturbation": np.max(np.abs(perturbations)),
        }
