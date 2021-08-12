import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)


def unzip(zip_file):
    with zipfile.ZipFile(zip_file, "r") as f:
        f.extractall()


def main():
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=checkpoint_dir / "sentiment_analyer")
    AutoTokenizer.from_pretrained(model_name, cache_dir=checkpoint_dir / "sentiment_analyer")

    url = "https://github.com/kingyiusuen/tweet-sentiment-extraction/releases/download/v0.1/sentiment_phrase_extractor.zip"  # noqa: E501
    sentiment_phrase_extractor_filepath = checkpoint_dir / "sentiment_phrase_extractor.zip"
    download_url(url, sentiment_phrase_extractor_filepath)
    unzip(sentiment_phrase_extractor_filepath)
    os.remove(sentiment_phrase_extractor_filepath)


if __name__ == "__main__":
    main()
