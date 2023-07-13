import os

from setuptools import setup, find_packages


def readme() -> str:
    """Utility function to read the README.md.

    Used for the `long_description`. It's nice, because now
    1) we have a top level README file and
    2) it's easier to type in the README file than to put a raw string in below.

    Args:
        nothing

    Returns:
        String of readed README.md file.
    """
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="solubility_model",
    version="0.1.0",
    author="Jon Swain",
    author_email="jonswain123@gmail.com",
    description="A ML model to predict solubility of small organic compounds, trained on data provided by the AI4SD summer school 2022.",
    python_requires=">=3",
    url="",
    packages=find_packages(),
    long_description=readme(),
)
