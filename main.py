import torch

from toy_meanflow.codebook import FixedGaussianCodebook
from toy_meanflow.config import DataConfig
from toy_meanflow.path import build_linear_path
from toy_meanflow.time_sampler import (UniformTimeSampler, UniformTimePairSampler)
from toy_meanflow.model import TinyMeanFlowModel
from toy_meanflow.objective import (build_meanflow_target, model_time_derivative, meanflow_loss)
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

    pair_sampler = UniformTimePairSampler(
        non_equal_ratio=0.75,
    )

    num_steps = 2000

    for step in range(1, num_steps+1):
        optimizer.zero_grad(set_to_none=True)

        loss = meanflow_loss(
            model=model,
            clean=continuous_batch,
            pair_sampler=pair_sampler,
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        ) # 梯度裁剪，

        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(
                f"step={step:03d} "
                f"loss={loss.item():.6f} "
                f"grad={grad_norm.item():.6f}"
            )
    """
    随机采样 noise
        ↓
    随机采样 r、t
        ↓
    构造 z_t
        ↓
    JVP 得到 prediction 和 du_dt
        ↓
    构造 MeanFlow target
        ↓
    target.detach()
        ↓
    计算 MSE
    """

if __name__ == "__main__":
    main()

"""
step=001 loss=2.118796 grad=0.876317
step=020 loss=2.038393 grad=0.822287
step=040 loss=1.758342 grad=0.633901
step=060 loss=1.739841 grad=0.742398
step=080 loss=1.769289 grad=0.616378
step=100 loss=2.007261 grad=0.872968
step=120 loss=1.773544 grad=0.640151
step=140 loss=1.890278 grad=0.695220
step=160 loss=1.719754 grad=0.717078
step=180 loss=1.658942 grad=0.743942
step=200 loss=1.455982 grad=0.587683 在1.5以上震荡非常厉害
"""