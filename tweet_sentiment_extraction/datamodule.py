from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from transformers import PreTrainedTokenizer


def train_val_split(
    dataset: Dataset,
    val_percent: float,
    random_state: int = 42,
) -> Tuple[Subset, Subset]:
    """Split a dataset into two."""
    num_samples = len(dataset)
    val_size = int(num_samples * val_percent)
    train_size = num_samples - val_size
    generator = torch.Generator().manual_seed(random_state)
    return random_split(dataset, [train_size, val_size], generator)


class CSVData(Dataset):
    def __init__(self, filepath: Union[Path, str]):
        self.df = pd.read_csv(filepath)
        self.df.dropna(inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.df.iloc[idx].to_dict()


class DataModule:
    """Describes the process of data preparation. Inspired by Pytorch Lightning."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        val_percent: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        self.tokenizer = tokenizer
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.datasets: Dict[str, Union[Dataset, Subset]] = {}

    @property
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "data"

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_val_dataset = CSVData(self.data_dir / "train.csv")
            train_dataset, val_dataset = train_val_split(train_val_dataset, self.val_percent)
            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset

        if stage in ("predict", None):
            # I didn't call this a test dataset, because a test dataset would have ground truth labels
            predict_dataset = CSVData(self.data_dir / "test.csv")
            self.datasets["predict"] = predict_dataset

    def collate_fn(self, batch: Dict[Any, Any]) -> Dict[str, Any]:
        """Convert text to token ids, create attention masks, pad sequences."""
        raw_inputs = {key: [d[key] for d in batch] for key in batch[0].keys()}
        has_ground_truth = "selected_text" in raw_inputs
        model_inputs = self.tokenizer(
            raw_inputs["sentiment"],
            raw_inputs["text"],
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=has_ground_truth,
        )
        if has_ground_truth:
            offsets_mapping = model_inputs.pop("offset_mapping")
            start_positions, end_positions = find_start_and_end_positions(
                raw_inputs["text"],
                raw_inputs["selected_text"],
                offsets_mapping,
            )
            model_inputs["start_positions"] = start_positions
            model_inputs["end_positions"] = end_positions
        return {"raw_inputs": raw_inputs, "model_inputs": model_inputs}

    def get_dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            shuffle=split == "train",  # Only shuffle the training dataset
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )


def find_start_and_end_positions(
    text: List[str],
    selected_text: List[str],
    offset_mapping: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Find the start and end positions of selected_text in tokenized text."""
    batch_size = len(text)
    start_positions = torch.zeros(batch_size, dtype=torch.long)
    end_positions = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        # Find start and end positions on raw text
        start_position_raw = text[i].find(selected_text[i])
        end_position_raw = start_position_raw + len(selected_text[i])
        mask = np.full(len(text[i]), False, dtype=bool)
        mask[start_position_raw : end_position_raw + 1] = True
        # Find start and end postisions after tokenization
        # The tokenized sequence has a pattern of: [CLS] sentiment [SEP] text [SEP]
        target_idx = []
        for j, (offset1, offset2) in enumerate(offset_mapping[i]):
            if j < 3:  # Skip [CLS], sentiment, [SEP]
                continue
            if any(mask[offset1:offset2]):
                target_idx.append(j)
        start_positions[i] = target_idx[0]
        end_positions[i] = target_idx[-1]
    return start_positions, end_positions
