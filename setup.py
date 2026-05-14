"""Setup script for the Driver Behavior Analytics & Crash Prediction System."""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="driver-behavior-analytics",
    version="2.1.0",
    author="Jay Guwalani",
    author_email="guwalanijj@gmail.com",
    description=(
        "Real-time vehicle safety monitoring with YOLOv7, ensemble ML, "
        "and survival analysis for crash prediction."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jay-guwalani/Driver-Behavior-Analytics-Crash-Prediction-System",
    packages=find_packages(exclude=("tests", "tests.*", "notebooks")),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "dba-infer=scripts.run_inference:main",
            "dba-api=src.api.inference_api:run",
        ],
    },
)
