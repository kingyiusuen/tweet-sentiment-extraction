from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).parent


with open(BASE_DIR / "requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


dev_packages = [
    "black==21.7b0",
    "flake8==3.9.2",
    "isort==5.9.3",
]


setup(
    name="tweet-sentiment-extraction",
    version="0.1.0",
    license="MIT",
    description="Extract words that support a tweet's sentiment.",
    author="King Yiu Suen",
    author_email="kingyiusuen@gmail.com",
    url="https://github.com/kingyiusuen/tweet-sentiment-extraction/",
    keywords=[
        "machine-learning",
        "deep-learning",
        "artificial-intelligence",
        "neural-network",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "dev": dev_packages,
    },
)
