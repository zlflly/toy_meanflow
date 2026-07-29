import torch

from toy_meanflow.codebook import FixedGaussianCodebook
from toy_meanflow.config import DataConfig
from toy_meanflow.path import build_linear_path
from toy_meanflow.time_sampler import (UniformTimeSampler, UniformTimePairSampler)
from toy_meanflow.model import TinyMeanFlowModel
from toy_meanflow.objective import flow_matching_loss

from toy_meanflow.data import (
    BlockDataset,
    TokenBuffer,
    build_dataloader,
)
from toy_meanflow.tokenizer import ByteTokenizer
torch.manual_seed(42)


def main() -> None:
    # --- 数据管线 ---
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

    token_batch = next(iter(dataloader)) # 现在实际进入continuous_batch当前的只有8个block当中的前两个
    """
    dataloader本身是一个DataLoader对象，不能直接取值,iter把它变成一共迭代器
    next用一次，就是提取迭代器中的下一个元素，而dataloader每次产生的是一个batch（足够数量的block）
    """
    continuous_batch = codebook.encode(token_batch)

    # --- 模型与优化器 ---
    model = TinyMeanFlowModel(
        data_dim=continuous_batch.shape[-1],
        hidden_dim=64,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # --- 训练循环（暂停：等 objective 支持时间对 (r, t) 后恢复）---
    # time_sampler = UniformTimeSampler()
    #
    # parameter_before = (
    #     model.input_projection.weight
    #     .detach()
    #     .clone()
    # )
    #
    # fixed_noise = torch.randn_like(
    #     continuous_batch
    # )
    #
    # fixed_t = torch.tensor(
    #     [0.25, 0.75],
    #     device=continuous_batch.device,
    #     dtype=continuous_batch.dtype,
    # )
    #
    # num_steps = 100
    #
    # for step in range(1, num_steps + 1):
    #     optimizer.zero_grad(set_to_none=True) # 每一步都先清空梯度
    #
    #     loss = flow_matching_loss(
    #         model=model,
    #         clean=continuous_batch,
    #         time_sampler=time_sampler,
    #         noise=fixed_noise,
    #         t=fixed_t,
    #     )
    #
    #     loss.backward()
    #
    #     grad_norm = torch.nn.utils.clip_grad_norm_(
    #         model.parameters(),
    #         max_norm=1.0
    #     )
    #
    #     optimizer.step()
    #
    #     if step == 1 or step % 10 == 0:
    #         print(
    #             f"step={step:03d} "
    #             f"loss={loss.item():.6f}" # loss.item()把零维的pytorch张量变成普通的python浮点数
    #         )
    #
    # parameter_after = (
    #     model.input_projection.weight
    #     .detach()
    #     .clone()
    # )
    #
    # parameters_changed = not torch.equal(
    #     parameter_before,
    #     parameter_after,
    # )
    #
    # print("\nTraining loss:")
    # print(loss.item())
    #
    # print("\nParameters changed:")
    # print(parameters_changed)
    #
    # parameter_change = (
    #     parameter_after - parameter_before
    # ).abs().mean()
    #
    # print("\nMean parameter change:")
    # print(parameter_change.item())

    # --- 测试 MeanFlow 时间对前向传播 ---
    pair_sampler = UniformTimePairSampler(
        non_equal_ratio=0.75,
    )

    r, t = pair_sampler.sample(
        batch_size=continuous_batch.shape[0],
        device=continuous_batch.device,
        dtype=continuous_batch.dtype,
    )

    noise = torch.randn_like(
        continuous_batch
    )

    z_t, velocity = build_linear_path(
        clean=continuous_batch,
        noise=noise,
        t=t,
    )

    predicted_average_velocity = model(
        z_t=z_t,
        r=r,
        t=t,
    )

    print("z_t shape:")
    print(z_t.shape)

    print("\nr shape:")
    print(r.shape)

    print("\nt shape:")
    print(t.shape)

    print("\nInterval lengths:")
    print(t - r)

    print("\nPredicted average velocity shape:")
    print(predicted_average_velocity.shape)

    print("\nOutput shape matches z_t:")
    print(
        predicted_average_velocity.shape
        == z_t.shape
    )


if __name__ == "__main__":
    main()
