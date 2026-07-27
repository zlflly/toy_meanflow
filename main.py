from toy_meanflow.config import DataConfig
from toy_meanflow.data import split_into_blocks
from toy_meanflow.tokenizer import ByteTokenizer


def main() -> None:
    data_config = DataConfig(seq_len=8)
    tokenizer = ByteTokenizer()

    text = "Hello MeanFlow!"
    token_ids = tokenizer.encode(text)

    blocks = split_into_blocks(
        token_ids=token_ids,
        block_size=data_config.seq_len,
    )

    print("Original text:")
    print(text)

    print("\nAll token IDs:")
    print(token_ids)

    print("\nSequence length:")
    print(data_config.seq_len)

    print("\nBlocks:")
    for index, block in enumerate(blocks):
        print(f"block {index}: {block}")
        print(f"decoded: {tokenizer.decode(block)}")


if __name__ == "__main__":
    main()