from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ml-project-template",
    packages=find_packages(),
    description="Template Repository for ML Projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    url="https://github.com/dialexa/ml-project-template.git",
    python_requires=">=3.8",
)
