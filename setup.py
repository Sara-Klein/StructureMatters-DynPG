from setuptools import setup, find_packages

setup(
    name='dynamic-policy-gradient',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # list your package dependencies here
        # e.g., 'numpy', 'pandas>=1.0.0'
        "numpy",
        "torch>=2.0.1",
        "scipy",
        "gymnasium",
        "matplotlib",
        "absl-py"
    ],
)