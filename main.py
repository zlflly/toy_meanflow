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

    print("Number of samples:")
    print(len(dataset))

    print("\nNumber of batches:")
    print(len(dataloader))

    for batch_index, token_batch in enumerate(dataloader):
        print(f"\nBatch {batch_index}:")
        print(token_batch)

        print("Shape:")
        print(token_batch.shape)

        print("Dtype:")
        print(token_batch.dtype)

        print("Decoded samples:")

        for sample in token_batch:
            print(repr(tokenizer.decode(sample)))


if __name__ == "__main__":
    main()