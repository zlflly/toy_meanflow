import torch

def build_linear_path(
        clean: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor, # 每个位置都是一个时间标量，例如t = torch.tensor([0.0, 1.0])，表示第0个样本构造的时间t=0，第1个样本构造的t=1
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")

    if t.ndim != 1:
        raise ValueError("t must be a 1D tensor")

    if t.shape[0] != clean.shape[0]: # 表示t[i]对应第i个样本的所有特征维度使用同一个 t
        """
        例如
        clean.shape = (3, 4)
        noise.shape = (3, 4)
        t = torch.tensor([0.0, 0.5, 1.0])
        表示第0个样本，每个特征维度对应的时间都是t
        """
        raise ValueError("t must contain one value per batch sample")

    # t 的形状从 (batch,) 变成 (batch, 1, 1, ...) 便于广播
    # 例如 clean.shape=(3,4) 时 t_view.shape=(3,1)，广播时 t[0] 扩展为 [0.0, 0.0, 0.0, 0.0]
    t_view = t.reshape(
        t.shape[0],
        *([1]*(clean.ndim - 1)),
    )

    z_t = (1.0 - t_view) * clean + t_view *noise

    velocity = noise - clean # 在线性路径中国，这个速度是常量

    return z_t, velocity