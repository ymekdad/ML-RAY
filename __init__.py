"""
ML-ray: Large-Scale Automated Vulnerability Assessment of Public AI Models

A multi-layered framework for automated security testing of public AI models
hosted on platforms such as Hugging Face.
"""

__version__ = "1.0.0"
__author__ = "ML-ray Authors"

from mlray.core import MLRay
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

__all__ = [
    "MLRay",
    "HardwareLayer",
    "LibraryLayer",
    "DataLayer",
    "ModelManagementLayer",
    "SecurityAssessmentLayer",
    "UILayer",
    "MLSecurityModule",
    "LLMSecurityModule",
    "PrivacyExtractionModule",
    "PerformanceAssessmentModule",
    "VulnerabilityReportingModule",
]
