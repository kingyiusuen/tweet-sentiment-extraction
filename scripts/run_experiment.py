from argparse import ArgumentParser

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from tweet_sentiment_extraction import DataModule, Trainer


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepset/roberta-base-squad2")
    parser.add_argument("--val_percent", type=float, default=0.1, help="Percentage of train set used for validation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Percentage of steps used to warmup")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sentiment_phrase_extractor")
    parser.add_argument("--do_fit", default=False, action="store_true")
    parser.add_argument("--do_predict", default=False, action="store_true")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if not args.do_fit and not args.do_predict:
        raise RuntimeError("Use at least one of --do_fit (for training) and --do_predict (for inference).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = DataModule(
        tokenizer=tokenizer,
        val_percent=args.val_percent,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device != "cpu",
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        warmup_percent=args.warmup_percent,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.do_fit:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        datamodule.setup("fit")
        train_dataloader = datamodule.get_dataloader("train")
        val_dataloader = datamodule.get_dataloader("val")
        trainer.fit(model, train_dataloader, val_dataloader)

    if args.do_predict:
        model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint_dir)
        datamodule.setup("predict")
        predict_dataloader = datamodule.get_dataloader("predict")
        trainer.predict(model, predict_dataloader)


if __name__ == "__main__":
    main()
