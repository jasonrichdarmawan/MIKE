from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent

setup(
    name="transformers_eap",
    version="0.0.1",
    description="EAP helpers and graph utilities",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Jason Rich Darmawan",
    author_email= "63768126+jasonrichdarmawan@users.noreply.github.com",
    packages=find_packages(include=["transformers_eap", "transformers_eap.*"]),
    install_requires=[
        # add runtime deps here, e.g. "torch>=1.12", "transformers>=4.0"
    ],
    python_requires=">=3.10",
)