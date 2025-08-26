from setuptools import setup, find_packages

setup(
    name="posterior_nco",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "matplotlib"],
    python_requires=">=3.9",
)