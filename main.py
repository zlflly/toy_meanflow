from toy_meanflow.config import DataConfig
from toy_meanflow.data import BlockDataset, TokenBuffer
from toy_meanflow.tokenizer import ByteTokenizer


def main() -> None:
    data_config = DataConfig(seq_len=8)
    tokenizer = ByteTokenizer()

    buffer = TokenBuffer(
        block_size=data_config.seq_len,
    )

    first_tokens = tokenizer.encode(
        "Hello",
        add_eos=False,
    )

    first_blocks = buffer.add(first_tokens)

    second_tokens = tokenizer.encode(
        " MeanFlow!",
        add_eos=True,
    )

    second_blocks = buffer.add(second_tokens)

    all_blocks = first_blocks + second_blocks

    dataset = BlockDataset(all_blocks)

    print("Number of samples:")
    print(len(dataset))

    print("\nBlock size:")
    print(dataset.block_size)

    first_sample = dataset[0]

    print("\nFirst sample:")
    print(first_sample)

    print("\nFirst sample type:")
    print(type(first_sample))

    print("\nFirst sample dtype:")
    print(first_sample.dtype)

    print("\nFirst sample shape:")
    print(first_sample.shape)

    print("\nDecoded first sample:")
    print(tokenizer.decode(first_sample))


if __name__ == "__main__":
    main()