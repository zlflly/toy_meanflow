import torch
from torch import nn

from toy_meanflow.path import build_linear_path
from toy_meanflow.time_sampler import UniformTimeSampler

def flow_matching_loss(
    model: nn.Module,
    clean: torch.Tensor, # 这已经是经过码本编码过的连续向量训练数据了
    time_sampler: UniformTimeSampler,
    noise: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> torch.tensor:
    """这里传入的 model 只是说明是 nn.Module 对象，它其实传过来的是 TinyVelocityModel 的实例"""
    if noise is None:
        noise = torch.randn_like(clean)

    if t is None:
        t = time_sampler.sample(
            batch_size=clean.shape[0],
            device=clean.device,
            dtype=clean.dtype,
        )

    z_t, target_velocity = build_linear_path(
        clean=clean,
        noise=noise,
        t=t,
    )

    predicted_velocity = model(
        z_t=z_t,
        t = t,
    )

    loss = (
        predicted_velocity - target_velocity
    ).square().mean() # 计算均方误差，先平方，再取平均

    return loss