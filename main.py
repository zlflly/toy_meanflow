import torch

from toy_meanflow.codebook import FixedGaussianCodebook
from toy_meanflow.config import DataConfig
from toy_meanflow.path import build_linear_path
from toy_meanflow.time_sampler import UniformTimeSampler
from toy_meanflow.model import TinyVelocityModel
from toy_meanflow.objective import flow_matching_loss
from toy_meanflow.data import (
    BlockDataset,
    TokenBuffer,
    build_dataloader,
)
from toy_meanflow.tokenizer import ByteTokenizer
torch.manual_seed(42)


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

    time_sampler = UniformTimeSampler()

    random_t = time_sampler.sample(
        batch_size=clean.shape[0],
        device=clean.device,
        dtype=clean.dtype,
    )
    random_z_t, random_velocity = build_linear_path(
        clean=clean,
        noise=noise,
        t=random_t
    ) 

    model = TinyVelocityModel(
        data_dim=continuous_batch.shape[-1],
        hidden_dim=64,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    time_sampler = UniformTimeSampler()

    parameter_before = (
        model.input_projection.weight
        .detach()
        .clone()
    )

    num_steps = 10000

    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True) # 每一步都先清空梯度

        loss = flow_matching_loss(
            model=model,
            clean=continuous_batch,
            time_sampler=time_sampler,
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(
                f"step={step:03d} "
                f"loss={loss.item():.6f}" # loss.item()把零维的pytorch张量变成普通的python浮点数
            )

    parameter_after = (
    model.input_projection.weight
    .detach()
    .clone()
    )

    parameters_changed = not torch.equal(
        parameter_before,
        parameter_after,
    )

    print("\nTraining loss:")
    print(loss.item())

    print("\nParameters changed:")
    print(parameters_changed)

    parameter_change = (
    parameter_after - parameter_before
    ).abs().mean()

    print("\nMean parameter change:")
    print(parameter_change.item())


if __name__ == "__main__":
    main()