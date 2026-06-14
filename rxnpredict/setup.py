from setuptools import setup, find_packages
setup(
    name="rxnpredict",
    version="1.0.0",
    author="Dr. Mushtaq Ali",
    author_email="info@dream2europe.com",
    description="Transformer retrosynthesis prediction with SMILES augmentation on USPTO-50K",
    packages=find_packages(),
    python_requires=">=3.8",
)
