from typing import Dict, List, Union

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, PreTrainedModel, PreTrainedTokenizer, get_scheduler

from .models import post_process


class Trainer:
    """Defines model training logics."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: str,
        num_epochs: int,
        lr: float,
        warmup_percent: float,
        checkpoint_dir: str,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.warmup_percent = warmup_percent
        self.checkpoint_dir = checkpoint_dir

    def fit(
        self,
        model: PreTrainedModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """Fit a model."""
        self.model = model
        self.dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        self.on_fit_start()
        for self.epoch in range(self.num_epochs):
            for phase in ["train", "val"]:
                self.on_epoch_start(phase)
                progress_bar = tqdm(self.dataloaders_dict[phase], leave=False, desc=phase)
                for batch in self.dataloaders_dict[phase]:
                    batch["model_inputs"] = self.to_device(batch["model_inputs"])
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss = self.common_step(batch)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                    else:
                        with torch.no_grad():
                            self.common_step(batch)
                    progress_bar.update(1)
                    progress_bar.set_postfix({k: v.curr_batch_avg for k, v in self.metrics.items()})
                self.on_epoch_end(phase)

    def on_fit_start(self) -> None:
        """Called before fitting starts."""
        self.model.to(self.device)
        num_training_steps = len(self.dataloaders_dict["train"]) * self.num_epochs
        self.configure_optimizer_and_scheduler(num_training_steps)
        self.metrics = {"loss": MetricTracker(), "jaccard": MetricTracker()}
        self.best_val_loss = float("inf")
        self.tokenizer.save_pretrained(self.checkpoint_dir)

    def on_epoch_start(self, phase: str) -> None:
        """Called at the beginning of each training/validation epoch."""
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        for metric in self.metrics.values():
            metric.reset()

    def on_epoch_end(self, phase: str) -> None:
        """Called at the end of each training/validation epoch."""
        epoch_metrics = {k: v.compute() for k, v in self.metrics.items()}
        print(
            f"Epoch {self.epoch + 1}/{self.num_epochs} | {phase.capitalize():^5} | "
            f"Loss: {epoch_metrics['loss']:.4f} | Jaccard: {epoch_metrics['jaccard']:.4f}"
        )

        if phase == "val" and epoch_metrics["loss"] < self.best_val_loss:
            self.best_val_loss = epoch_metrics["loss"]
            self.model.save_pretrained(self.checkpoint_dir)

    def common_step(self, batch) -> Tensor:
        """Step shared by training and validation epochs."""
        outputs = self.model(**batch["model_inputs"])
        loss = outputs["loss"]
        preds = post_process(
            self.tokenizer,
            batch["model_inputs"]["input_ids"],
            outputs["start_logits"],
            outputs["end_logits"],
        )
        jaccard_index = jaccard(preds, batch["raw_inputs"]["selected_text"])
        batch_size = len(batch["model_inputs"]["input_ids"])
        self.metrics["loss"].update(loss.item(), batch_size)
        self.metrics["jaccard"].update(jaccard_index, batch_size)
        return loss

    @torch.no_grad()
    def predict(self, model: PreTrainedModel, predict_dataloader: DataLoader) -> None:
        """Make predictions and output the predictions to a csv file."""
        model.eval()
        textID = []
        preds = []
        for batch in tqdm(predict_dataloader, leave=False):
            batch["model_inputs"] = self.to_device(batch["model_inputs"])
            outputs = model(**batch["model_inputs"])
            textID += batch["raw_inputs"]["textID"]
            preds += post_process(
                self.tokenizer,
                batch["model_inputs"]["input_ids"],
                outputs["start_logits"],
                outputs["end_logits"],
            )
        df = pd.DataFrame({"textID": textID, "selected_text": preds})
        df.to_csv("submission.csv", index=False)

    def to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Move tensors in batch to GPU (if available)."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def configure_optimizer_and_scheduler(self, num_training_steps) -> None:
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(num_training_steps * self.warmup_percent),
            num_training_steps=num_training_steps,
        )


class MetricTracker:
    """Keep track of the running sum of a metric, so that the metric can be aggregated at the end of an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg = 0
        self.total_val = 0
        self.num_samples = 0

    def update(self, val: Union[float, int], batch_size: int) -> None:
        self.curr_batch_avg = val
        self.total_val += val * batch_size
        self.num_samples += batch_size

    def compute(self) -> float:
        return self.total_val / self.num_samples


def jaccard(preds: List[str], targets: List[str]) -> float:
    """Implementation of the word-level Jaccard index.

    Reference:
    https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation
    """
    correct = 0.0
    for str1, str2 in zip(preds, targets):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        correct += len(c) / (len(a) + len(b) - len(c))
    return correct / len(preds)
