from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer


class SentimentAnalyzer:
    """Predicts the sentiment of a text."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", cache_dir=checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model.eval()
        self.labels = ["positive", "neutral", "negative"]

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**model_inputs)
        logits = outputs["logits"].squeeze(0)
        probs = torch.softmax(logits)
        return {label: prob.item() for label, prob in zip(self.labels, probs)}


class SentimentPhraseExtractor:
    """Predicts the phrase from the text that exemplifies the provided sentiment."""

    def __init__(self, checkpoint_dir: Union[Path, str]):
        self.model = AutoModel.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(self, sentiment: str, text: str) -> str:
        model_inputs = self.tokenizer(sentiment, text, return_tensors="pt")
        outputs = self.model(**model_inputs)
        pred = post_process(
            self.tokenizer,
            model_inputs["input_ids"],
            outputs["start_logits"],
            outputs["end_logits"],
        )[0]
        return pred


def post_process(
    tokenizer: PreTrainedTokenizer,
    batch_input_ids: Tensor,
    start_logits: Tensor,
    end_logits: Tensor,
) -> List[str]:
    """Extract support phrases based on predicted starting and ending positions."""
    # Get the most likely beginning and end of answers
    start_ids = torch.argmax(start_logits, dim=1)
    end_ids = torch.argmax(end_logits, dim=1) + 1
    # Fetch the tokens from the identified start and stop values, convert those tokens to a string
    answers = []
    batch_input_ids = batch_input_ids.tolist()
    batch_size = len(batch_input_ids)
    for i in range(batch_size):
        input_ids, start_id, end_id = batch_input_ids[i], start_ids[i], end_ids[i]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[start_id:end_id])
        answer = tokenizer.convert_tokens_to_string(tokens)
        answers.append(answer)
    return answers
