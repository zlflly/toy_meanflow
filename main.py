from toy_meanflow.codebook import FixedGaussianCodebook
from toy_meanflow.config import DataConfig
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


if __name__ == "__main__":
    main()