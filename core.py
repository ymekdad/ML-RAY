"""
ML-ray Core Module

Main entry point for the ML-ray vulnerability assessment framework.
Orchestrates all layers and modules for comprehensive security testing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from mlray.layers import (
    HardwareLayer,
    LibraryLayer,
    DataLayer,
    ModelManagementLayer,
    SecurityAssessmentLayer,
    UILayer,
)
from mlray.modules import (
    MLSecurityModule,
    LLMSecurityModule,
    PrivacyExtractionModule,
    PerformanceAssessmentModule,
    VulnerabilityReportingModule,
)
from mlray.utils.metrics import MetricsCalculator
from mlray.configs import AttackConfig, load_config


@dataclass
class AssessmentResults:
    """Container for vulnerability assessment results."""
    
    model_id: str
    timestamp: str
    clean_metrics: Dict[str, float] = field(default_factory=dict)
    adversarial_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vulnerability_score: float = 0.0
    attack_success_rates: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> None:
        """Print a summary of assessment results."""
        print(f"\n{'='*60}")
        print(f"Vulnerability Assessment Results: {self.model_id}")
        print(f"{'='*60}")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nClean Performance:")
        for metric, value in self.clean_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nVulnerability Score: {self.vulnerability_score:.2%}")
        print(f"\nAttack Success Rates:")
        for attack, asr in self.attack_success_rates.items():
            print(f"  {attack}: {asr:.2%}")
        print(f"{'='*60}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "clean_metrics": self.clean_metrics,
            "adversarial_metrics": self.adversarial_metrics,
            "vulnerability_score": self.vulnerability_score,
            "attack_success_rates": self.attack_success_rates,
            "recommendations": self.recommendations,
        }


class MLRay:
    """
    ML-ray: Automated Vulnerability Assessment Framework
    
    A multi-layered framework for comprehensive security testing of
    public AI models across diverse architectures and frameworks.
    
    Attributes:
        device: Computation device (cuda/cpu)
        mode: Assessment mode (ml/llm/qa)
        cache_dir: Directory for model caching
        config: Configuration settings
    
    Example:
        >>> scanner = MLRay(device="cuda")
        >>> scanner.load_model("microsoft/resnet-50")
        >>> results = scanner.assess(attacks=["fgsm", "pgd"])
        >>> results.summary()
    """
    
    def __init__(
        self,
        device: str = "auto",
        mode: str = "ml",
        cache_dir: str = "./cache",
        config_path: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize ML-ray framework.
        
        Args:
            device: Computation device ("auto", "cuda", "cuda:0", "cpu")
            mode: Assessment mode ("ml", "llm", "qa")
            cache_dir: Directory for caching models and data
            config_path: Path to configuration file
            log_level: Logging level
        """
        self._setup_logging(log_level)
        logger.info("Initializing ML-ray vulnerability assessment framework")
        
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize layers
        self.hardware_layer = HardwareLayer(device=device)
        self.device = self.hardware_layer.device
        
        self.library_layer = LibraryLayer()
        self.data_layer = DataLayer(cache_dir=self.cache_dir)
        self.model_management = ModelManagementLayer(
            cache_dir=self.cache_dir,
            device=self.device
        )
        self.security_layer = SecurityAssessmentLayer(device=self.device)
        self.ui_layer = UILayer()
        
        # Initialize modules
        self.ml_security = MLSecurityModule(device=self.device)
        self.llm_security = LLMSecurityModule(device=self.device)
        self.privacy_module = PrivacyExtractionModule(device=self.device)
        self.performance_module = PerformanceAssessmentModule()
        self.reporting_module = VulnerabilityReportingModule()
        
        # Load configuration
        self.config = load_config(config_path) if config_path else AttackConfig()
        
        # State
        self.model = None
        self.model_id = None
        self.model_info = None
        self.preprocessor = None
        
        logger.info(f"ML-ray initialized on device: {self.device}")
    
    def _setup_logging(self, level: str) -> None:
        """Configure logging."""
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
        )
    
    def load_model(
        self,
        model_id: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Load a model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model identifier (e.g., "microsoft/resnet-50")
            revision: Specific model revision/commit
            trust_remote_code: Allow execution of remote code
        """
        logger.info(f"Loading model: {model_id}")
        
        # Detect framework
        framework = self.library_layer.detect_framework(model_id)
        logger.info(f"Detected framework: {framework}")
        
        # Resolve dependencies
        self.library_layer.resolve_dependencies(model_id)
        
        # Load model through model management layer
        self.model, self.model_info = self.model_management.load_model(
            model_id=model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        
        self.model_id = model_id
        
        # Setup preprocessor
        self.preprocessor = self.model_management.get_preprocessor(model_id)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        self.model_management.log_model_info(model_id, self.model_info)
        
        logger.info(f"Model loaded successfully: {model_id}")
    
    def assess(
        self,
        attacks: List[Union[str, Any]],
        dataset: str = "imagenet-1k",
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        save_adversarial: bool = False,
    ) -> AssessmentResults:
        """
        Run vulnerability assessment on loaded model.
        
        Args:
            attacks: List of attack names or attack instances
            dataset: Dataset identifier for evaluation
            batch_size: Batch size for evaluation
            num_samples: Number of samples to evaluate (None for full dataset)
            save_adversarial: Whether to save generated adversarial examples
        
        Returns:
            AssessmentResults containing vulnerability metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Starting vulnerability assessment for {self.model_id}")
        logger.info(f"Attacks: {attacks}")
        
        # Load dataset
        dataloader = self.data_layer.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            preprocessor=self.preprocessor,
        )
        
        # Evaluate clean performance
        logger.info("Evaluating clean performance...")
        clean_metrics = self.performance_module.evaluate_clean(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
        )
        
        # Run attacks
        adversarial_metrics = {}
        attack_success_rates = {}
        
        for attack in tqdm(attacks, desc="Running attacks"):
            attack_name = attack if isinstance(attack, str) else attack.__class__.__name__
            logger.info(f"Executing attack: {attack_name}")
            
            results = self.ml_security.run_attack(
                model=self.model,
                attack=attack,
                dataloader=dataloader,
                config=self.config,
                save_adversarial=save_adversarial,
            )
            
            adversarial_metrics[attack_name] = results["metrics"]
            attack_success_rates[attack_name] = results["attack_success_rate"]
        
        # Calculate overall vulnerability score
        vulnerability_score = np.mean(list(attack_success_rates.values()))
        
        # Generate recommendations
        recommendations = self.reporting_module.generate_recommendations(
            clean_metrics=clean_metrics,
            adversarial_metrics=adversarial_metrics,
            attack_success_rates=attack_success_rates,
        )
        
        results = AssessmentResults(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            clean_metrics=clean_metrics,
            adversarial_metrics=adversarial_metrics,
            vulnerability_score=vulnerability_score,
            attack_success_rates=attack_success_rates,
            recommendations=recommendations,
        )
        
        logger.info(f"Assessment complete. Vulnerability score: {vulnerability_score:.2%}")
        
        return results
    
    def assess_llm(
        self,
        attacks: List[Union[str, Any]],
        num_prompts: int = 100,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AssessmentResults:
        """
        Run LLM-specific vulnerability assessment.
        
        Args:
            attacks: List of LLM attack names or instances
            num_prompts: Number of test prompts per attack
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
        
        Returns:
            AssessmentResults for LLM vulnerabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Starting LLM vulnerability assessment for {self.model_id}")
        
        adversarial_metrics = {}
        attack_success_rates = {}
        
        for attack in tqdm(attacks, desc="Running LLM attacks"):
            attack_name = attack if isinstance(attack, str) else attack.__class__.__name__
            logger.info(f"Executing LLM attack: {attack_name}")
            
            results = self.llm_security.run_attack(
                model=self.model,
                tokenizer=self.preprocessor,
                attack=attack,
                num_prompts=num_prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            adversarial_metrics[attack_name] = results["metrics"]
            attack_success_rates[attack_name] = results["attack_success_rate"]
        
        vulnerability_score = np.mean(list(attack_success_rates.values()))
        
        recommendations = self.reporting_module.generate_llm_recommendations(
            attack_success_rates=attack_success_rates,
        )
        
        return AssessmentResults(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            clean_metrics={},
            adversarial_metrics=adversarial_metrics,
            vulnerability_score=vulnerability_score,
            attack_success_rates=attack_success_rates,
            recommendations=recommendations,
        )
    
    def assess_qa(
        self,
        attacks: List[Union[str, Any]],
        test_samples: int = 1000,
    ) -> AssessmentResults:
        """
        Run QA model vulnerability assessment.
        
        Args:
            attacks: List of attack names or instances
            test_samples: Number of test samples
        
        Returns:
            AssessmentResults for QA model vulnerabilities
        """
        return self.assess_llm(attacks=attacks, num_prompts=test_samples)
    
    def full_assessment(
        self,
        dataset: str = "imagenet-1k",
        white_box_attacks: Optional[List[str]] = None,
        black_box_attacks: Optional[List[str]] = None,
        privacy_attacks: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> AssessmentResults:
        """
        Run comprehensive vulnerability assessment across all attack categories.
        
        Args:
            dataset: Dataset for evaluation
            white_box_attacks: List of white-box attacks
            black_box_attacks: List of black-box attacks
            privacy_attacks: List of privacy attacks
            batch_size: Batch size for evaluation
        
        Returns:
            Comprehensive AssessmentResults
        """
        all_attacks = []
        
        if white_box_attacks:
            all_attacks.extend(white_box_attacks)
        if black_box_attacks:
            all_attacks.extend(black_box_attacks)
        
        # Run standard assessment
        results = self.assess(
            attacks=all_attacks,
            dataset=dataset,
            batch_size=batch_size,
        )
        
        # Run privacy assessment if specified
        if privacy_attacks:
            privacy_results = self.privacy_assessment(attacks=privacy_attacks)
            results.adversarial_metrics.update(privacy_results.adversarial_metrics)
            results.attack_success_rates.update(privacy_results.attack_success_rates)
            results.vulnerability_score = np.mean(
                list(results.attack_success_rates.values())
            )
        
        return results
    
    def privacy_assessment(
        self,
        attacks: List[str],
        num_queries: int = 600,
    ) -> AssessmentResults:
        """
        Run privacy and extraction attack assessment.
        
        Args:
            attacks: List of privacy attack names
            num_queries: Number of queries for extraction attacks
        
        Returns:
            AssessmentResults for privacy vulnerabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Starting privacy assessment for {self.model_id}")
        
        adversarial_metrics = {}
        attack_success_rates = {}
        
        for attack in attacks:
            logger.info(f"Executing privacy attack: {attack}")
            
            results = self.privacy_module.run_attack(
                model=self.model,
                attack=attack,
                num_queries=num_queries,
                device=self.device,
            )
            
            adversarial_metrics[attack] = results["metrics"]
            attack_success_rates[attack] = results["success_rate"]
        
        return AssessmentResults(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            clean_metrics={},
            adversarial_metrics=adversarial_metrics,
            vulnerability_score=np.mean(list(attack_success_rates.values())),
            attack_success_rates=attack_success_rates,
            recommendations=[],
        )
    
    def transfer_assessment(
        self,
        source_model: str,
        target_models: List[str],
        attack: Any,
        dataset: str = "imagenet-1k",
        batch_size: int = 32,
        num_samples: int = 1000,
    ) -> Dict[str, AssessmentResults]:
        """
        Assess adversarial transferability across models.
        
        Args:
            source_model: Model ID for generating adversarial examples
            target_models: List of target model IDs
            attack: Attack instance or name
            dataset: Dataset for evaluation
            batch_size: Batch size
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary mapping target model IDs to AssessmentResults
        """
        logger.info(f"Starting transfer attack assessment")
        logger.info(f"Source model: {source_model}")
        logger.info(f"Target models: {target_models}")
        
        # Load source model and generate adversarial examples
        self.load_model(source_model)
        
        dataloader = self.data_layer.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            preprocessor=self.preprocessor,
        )
        
        # Generate adversarial examples from source model
        adversarial_samples = self.ml_security.generate_adversarial(
            model=self.model,
            attack=attack,
            dataloader=dataloader,
            config=self.config,
        )
        
        # Evaluate on target models
        results = {}
        
        for target_id in target_models:
            logger.info(f"Evaluating transfer to: {target_id}")
            
            self.load_model(target_id)
            
            transfer_results = self.ml_security.evaluate_transfer(
                model=self.model,
                adversarial_samples=adversarial_samples,
                device=self.device,
            )
            
            results[target_id] = AssessmentResults(
                model_id=target_id,
                timestamp=datetime.now().isoformat(),
                clean_metrics=transfer_results.get("clean_metrics", {}),
                adversarial_metrics={"transfer": transfer_results["metrics"]},
                vulnerability_score=transfer_results["transfer_success_rate"],
                attack_success_rates={"transfer": transfer_results["transfer_success_rate"]},
                recommendations=[],
            )
        
        return results
    
    def compare_results(self, results: Dict[str, AssessmentResults]) -> None:
        """
        Compare vulnerability results across multiple models.
        
        Args:
            results: Dictionary mapping model IDs to AssessmentResults
        """
        print(f"\n{'='*80}")
        print("Model Vulnerability Comparison")
        print(f"{'='*80}")
        print(f"{'Model':<40} {'Vulnerability Score':<20} {'Max ASR':<15}")
        print(f"{'-'*80}")
        
        for model_id, result in sorted(
            results.items(),
            key=lambda x: x[1].vulnerability_score,
            reverse=True
        ):
            max_asr = max(result.attack_success_rates.values()) if result.attack_success_rates else 0
            print(f"{model_id:<40} {result.vulnerability_score:<20.2%} {max_asr:<15.2%}")
        
        print(f"{'='*80}\n")
    
    def save_results(
        self,
        results: AssessmentResults,
        output_path: str,
    ) -> None:
        """
        Save assessment results to file.
        
        Args:
            results: AssessmentResults to save
            output_path: Output file path (JSON)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    def generate_report(
        self,
        results: AssessmentResults,
        output_path: str,
        format: str = "pdf",
    ) -> None:
        """
        Generate vulnerability assessment report.
        
        Args:
            results: AssessmentResults to report
            output_path: Output file path
            format: Report format ("pdf", "html", "json")
        """
        self.reporting_module.generate_report(
            results=results,
            output_path=output_path,
            format=format,
        )
        logger.info(f"Report generated: {output_path}")
    
    def log_error(self, model_id: str, error: str) -> None:
        """Log an error for a model."""
        logger.error(f"Error for {model_id}: {error}")
        self.model_management.log_error(model_id, error)
