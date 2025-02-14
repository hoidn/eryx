from setuptools import setup, find_packages

setup(
    name="eryx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=2.0.0",
        "pytest",
    ],
    python_requires=">=3.8",
)
