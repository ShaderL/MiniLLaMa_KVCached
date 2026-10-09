import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model_miniLLaMa import ModelConfig, MiniLLaMA


class TokenBlockDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        block_size,
        text_column="text",
    ):
        self.samples = []

        for item in tqdm(dataset, desc="Tokenizing"):
            text = item[text_column]

            if not text or not text.strip():
                continue

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            # 一个完整训练样本至少需要 block_size + 1 个 token
            if len(token_ids) < block_size + 1:
                continue

            # 按 block_size + 1 切分
            for start in range(
                0,
                len(token_ids) - block_size,
                block_size,
            ):
                chunk = token_ids[start:start + block_size + 1]

                if len(chunk) < block_size + 1:
                    continue

                x = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                )

                self.samples.append((x, y))

        print(f"Created {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def set_seed(seed):
    """
    保证 random / torch 的随机实验尽可能可复现。
    """

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_dataset(config):
    """
    根据 config 加载数据集。

    当前第一版支持：
        - Hugging Face datasets
        - TinyStories
    """

    dataset_config = config["dataset"]

    dataset_name = dataset_config["name"]
    split = dataset_config.get("split", "train")
    text_column = dataset_config.get("text_column", "text")

    if dataset_name == "tinystories":
        hf_name = "roneneldan/TinyStories"
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    print(f"Loading dataset: {hf_name}")
    print(f"Split: {split}")

    dataset = load_dataset(
        hf_name,
        split=split,
    )


    sampling_config = dataset_config.get(
        "sampling",
        {},
    )

    strategy = sampling_config.get(
        "strategy",
        "first",
    )

    max_stories = sampling_config.get(
        "max_stories",
        None,
    )

    seed = sampling_config.get(
        "seed",
        42,
    )

    if max_stories is not None and max_stories < len(dataset):

        if strategy == "first":
            print(
                f"Using first {max_stories} stories."
            )

            dataset = dataset.select(
                range(max_stories)
            )

        elif strategy == "random":
            print(
                f"Randomly selecting "
                f"{max_stories} stories "
                f"(seed={seed})."
            )

            dataset = dataset.shuffle(
                seed=seed
            ).select(
                range(max_stories)
            )

        else:
            raise ValueError(
                f"Unknown sampling strategy: {strategy}"
            )

    print(f"Stories used: {len(dataset)}")

    return dataset, text_column


def build_model(config, vocab_size):
    """
    根据 config 创建 MiniLLaMA。
    """

    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=config["model"]["block_size"],
        num_layers=config["model"]["num_layers"],
        d_model=config["model"]["d_model"],
        d_ffn=config["model"]["d_ffn"],
        num_heads=config["model"]["num_heads"],
        rms_norm_eps=config["model"].get(
            "rms_norm_eps",
            1e-6,
        ),
    )

    model = MiniLLaMA(model_config)

    return model, model_config


def save_checkpoint(
    model,
    model_config,
    training_config,
    optimizer,
    epoch,
    loss,
    path,
):
    """
    保存训练 checkpoint。

    保存：
        - 模型参数
        - 模型结构配置
        - 训练配置
        - optimizer 状态
        - 当前 epoch
        - 当前 loss

    这样 generate.py 可以仅凭 checkpoint
    重建模型。
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "model_config": {
            "vocab_size": model_config.vocab_size,
            "block_size": model_config.block_size,
            "num_layers": model_config.num_layers,
            "d_model": model_config.d_model,
            "d_ffn": model_config.d_ffn,
            "num_heads": model_config.num_heads,
            "rms_norm_eps": model_config.rms_norm_eps,
        },

        "training_config": training_config,

        "optimizer_state_dict": optimizer.state_dict(),

        "epoch": epoch,

        "loss": loss,
    }

    torch.save(
        checkpoint,
        path,
    )

    print(f"Checkpoint saved to: {path}")


def train(config):
    """
    完整训练流程。

    config:
        从 YAML 读取的 Python dict。
    """

    runtime_config = config.get(
        "runtime",
        {},
    )

    seed = runtime_config.get(
        "seed",
        42,
    )

    set_seed(seed)

    device = runtime_config.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    device = torch.device(device)

    print("=" * 60)
    print("MiniLLaMA Training")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Seed: {seed}")

    tokenizer_name = config.get(
        "tokenizer",
        "gpt2",
    )

    print(f"Tokenizer: {tokenizer_name}")

    tokenizer = GPT2TokenizerFast.from_pretrained(
        tokenizer_name
    )

    # GPT-2 没有 pad token
    # 当前训练不需要 padding，因此这里只记录 vocab size。
    vocab_size = tokenizer.vocab_size

    print(f"Vocabulary size: {vocab_size}")

    dataset, text_column = load_training_dataset(
        config
    )

    block_size = config["model"]["block_size"]

    train_dataset = TokenBlockDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        block_size=block_size,
        text_column=text_column,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training samples were created. "
            "Try increasing max_stories or "
            "checking block_size."
        )

    max_train_samples = config["dataset"] \
        .get("sampling", {}) \
        .get("max_train_samples", None)

    if (
        max_train_samples is not None
        and max_train_samples < len(train_dataset)
    ):
        print(
            f"Limiting training samples "
            f"to {max_train_samples}."
        )

        train_dataset = torch.utils.data.Subset(
            train_dataset,
            range(max_train_samples),
        )

    print(
        f"Final training samples: "
        f"{len(train_dataset)}"
    )

    training_config = config["training"]

    batch_size = training_config.get(
        "batch_size",
        4,
    )

    num_workers = training_config.get(
        "num_workers",
        0,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Number of batches: {len(dataloader)}"
    )

    model, model_config = build_model(
        config,
        vocab_size,
    )

    model = model.to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{num_parameters:,}"
    )

    criterion = nn.CrossEntropyLoss()

    learning_rate = training_config.get(
        "learning_rate",
        3e-4,
    )

    weight_decay = training_config.get(
        "weight_decay",
        0.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_epochs = training_config.get(
        "num_epochs",
        1,
    )

    log_interval = training_config.get(
        "log_interval",
        100,
    )

    output_config = config.get(
        "output",
        {},
    )

    checkpoint_path = output_config.get(
        "checkpoint",
        "checkpoints/mini_llama.pth",
    )

    model.train()

    for epoch in range(num_epochs):

        print()
        print(
            f"===== Epoch "
            f"{epoch + 1}/{num_epochs} ====="
        )

        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
        )

        for step, (x, y) in enumerate(
            progress_bar
        ):

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            logits = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                y.reshape(-1),
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loss_value = loss.item()

            total_loss += loss_value

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}"
            )

            if (
                log_interval > 0
                and (step + 1) % log_interval == 0
            ):
                avg_loss = (
                    total_loss / (step + 1)
                )

                print(
                    f"Step "
                    f"{step + 1}/{len(dataloader)} "
                    f"| "
                    f"Loss: {loss_value:.4f} "
                    f"| "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        average_loss = (
            total_loss / len(dataloader)
        )

        print()
        print(
            f"Epoch {epoch + 1} finished "
            f"| Average Loss: "
            f"{average_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            model_config=model_config,
            training_config=config,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=average_loss,
            path=checkpoint_path,
        )

    print()
    print("=" * 60)
    print("Training finished.")
    print("=" * 60)

    return model

if __name__ == "__main__":
    raise RuntimeError(
        "Please run training through cli.py."
    )

