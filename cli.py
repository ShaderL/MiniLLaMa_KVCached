import argparse

import yaml
import torch

from train import train
from generate import generate_from_checkpoint


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_nested_value(config, path, value):
    current = config

    for key in path[:-1]:
        current = current.setdefault(key, {})

    current[path[-1]] = value


def train_command(args):
    config = load_config(args.config)

    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs

    if args.max_stories is not None:
        config["dataset"]["sampling"]["max_stories"] = args.max_stories

    if args.max_train_samples is not None:
        config["dataset"]["sampling"]["max_train_samples"] = (
            args.max_train_samples
        )

    if args.device is not None:
        config["runtime"]["device"] = args.device

    if args.seed is not None:
        config["runtime"]["seed"] = args.seed

    train(config)


def generate_command(args):
    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text = generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tokenizer_name=args.tokenizer,
    )

    print()
    print(text)


def build_parser():
    parser = argparse.ArgumentParser(
        description="MiniLLaMA command line interface"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train MiniLLaMA",
    )

    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config",
    )

    train_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name",
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )

    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )

    train_parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Override maximum number of stories",
    )

    train_parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override maximum number of training samples",
    )

    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda or cpu",
    )

    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    train_parser.set_defaults(
        func=train_command
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text",
    )

    generate_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mini_llama.pth",
        help="Path to model checkpoint",
    )

    generate_parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt text",
    )

    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens",
    )

    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )

    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )

    generate_parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name",
    )

    generate_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda or cpu",
    )

    generate_parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable sampling",
    )

    generate_parser.set_defaults(
        func=generate_command
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

