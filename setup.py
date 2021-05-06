from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()


install_requires = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"]

setup(
    name="nemea-fet",
    version="0.1.1",
    author="Daniel Uhříček",
    author_email="daniel.uhricek@gypri.cz",
    url="https://github.com/danieluhricek/nemea-fet",
    description="Nemea Feature Exploration Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    packages=find_packages(),
)
