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
) -> torch.Tensor:
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

def model_time_derivative(
    model: nn.Module,
    z_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """目的是用一次前向传播，同时拿到模型输出u和du/dt"""
    def model_function(
        current_z_t: torch.Tensor, # current_z_t的current前缀指的是这是一个模型内部的变量，跟外界的防止混淆
        current_r: torch.Tensor,
        current_t: torch.Tensor,
    ) -> torch.Tensor:
        # torch.func.jvp要求第一个参数必须是一个纯函数，
        return model(
            z_t = current_z_t,
            r=current_r,
            t=current_t,
        )

    prediction, du_dt = torch.func.jvp( # 返回一个二元数组，一个是普通的前向输出u，一个是方向导数，du/dt
        model_function,
        primals=(z_t, r, t),
        tangents=(
            velocity,
            torch.zeros_like(r),
            torch.ones_like(t),
        ),
    )

    return prediction, du_dt