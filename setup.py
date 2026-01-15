#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML-ray: Large-Scale Automated Vulnerability Assessment of Public AI Models
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ml-ray",
    version="1.0.0",
    author="ML-ray Authors",
    author_email="mlray@example.com",
    description="Automated vulnerability assessment framework for public AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlray-security/ml-ray",
    project_urls={
        "Bug Tracker": "https://github.com/mlray-security/ml-ray/issues",
        "Documentation": "https://github.com/mlray-security/ml-ray/docs",
        "Source Code": "https://github.com/mlray-security/ml-ray",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.25.0",
        "datasets>=2.8.0",
        "huggingface-hub>=0.12.0",
        "timm>=0.6.12",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "pandas>=1.4.0",
        "pyyaml>=6.0",
        "Pillow>=9.0.0",
        "loguru>=0.6.0",
        "adversarial-robustness-toolbox>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "llm": [
            "accelerate>=0.15.0",
            "evaluate>=0.4.0",
        ],
        "privacy": [
            "opacus>=1.3.0",
        ],
        "reporting": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "reportlab>=3.6.0",
            "jinja2>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlray=mlray.cli:main",
            "mlray-collect=scripts.collect_models:main",
            "mlray-assess=scripts.run_assessment:main",
            "mlray-report=scripts.generate_reports:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mlray": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "security",
        "adversarial",
        "vulnerability",
        "assessment",
        "deep learning",
        "hugging face",
        "llm",
        "robustness",
    ],
)
