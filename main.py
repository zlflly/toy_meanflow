import torch

from toy_meanflow.codebook import FixedGaussianCodebook
from toy_meanflow.config import DataConfig
from toy_meanflow.path import build_linear_path
from toy_meanflow.data import (
    BlockDataset,
    TokenBuffer,
    build_dataloader,
)
from toy_meanflow.tokenizer import ByteTokenizer


def main() -> None:
    data_config = DataConfig(
        seq_len=8,
        num_workers=0,
    )

    tokenizer = ByteTokenizer()

    buffer = TokenBuffer(
        block_size=data_config.seq_len,
    )

    texts = [
        "Hello MeanFlow!",
        "This is a toy project.",
        "We are learning PyTorch.",
    ]

    all_blocks: list[list[int]] = []

    for text in texts:
        token_ids = tokenizer.encode(text)
        new_blocks = buffer.add(token_ids)
        all_blocks.extend(new_blocks)

    dataset = BlockDataset(all_blocks)

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    codebook = FixedGaussianCodebook(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        seed=42,
    )

    token_batch = next(iter(dataloader))
    """
    dataloader本身是一个DataLoader对象，不能直接取值,iter把它变成一共迭代器
    next用一次，就是提取迭代器中的下一个元素，而dataloader每次产生的是一个batch（足够数量的block）
    """
    continuous_batch = codebook.encode(token_batch)

    print("Token batch:")
    print(token_batch)

    print("\nToken batch shape:")
    print(token_batch.shape)

    print("\nContinuous batch shape:")
    print(continuous_batch.shape)

    print("\nContinuous batch dtype:")
    print(continuous_batch.dtype)

    print("\nFirst token ID:")
    print(token_batch[0, 0])

    print("\nIts continuous vector:")
    print(continuous_batch[0, 0])

    # --- 测试 build_linear_path ---
    clean = continuous_batch

    noise = torch.randn_like(clean)

    t = torch.tensor(
        [0.0, 1.0],
        dtype=clean.dtype,
    )

    z_t, velocity = build_linear_path(
        clean=clean,
        noise=noise,
        t=t,
    )

    print("\nClean shape:")
    print(clean.shape)

    print("\nNoise shape:")
    print(noise.shape)

    print("\nNoisy path shape:")
    print(z_t.shape)

    print("\nVelocity shape:")
    print(velocity.shape)

    print("\nAt t=0, z_t equals clean:")
    print(torch.allclose(z_t[0], clean[0]))

    print("\nAt t=1, z_t equals noise:")
    print(torch.allclose(z_t[1], noise[1]))


if __name__ == "__main__":
    main()