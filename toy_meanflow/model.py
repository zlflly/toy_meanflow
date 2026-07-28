import torch
from torch import nn

class TinyVelocityModel(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        if data_dim <= 0:
            raise ValueError("data_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.input_projection = nn.Linear(
            data_dim,
            hidden_dim,
        )
        # 把每一个token的连续向量的维度，从data_dim扩展到hidden_dim
        

        self.time_projection = nn.Linear(
            1,
            hidden_dim,
        )
        # 把变量时间t扩展到hidden_dim，之后会把这个时间向量加到序列的每个位置上

        self.output_projection = nn.Linear(
            hidden_dim,
            data_dim,
        ) # 把投影重新映射回data_dim

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.ndim != 3:
            raise ValueError("z_t must have shape [batch, length, dim]")

        if t.ndim != 1:
            raise ValueError("t must have shape [batch]")

        if z_t.shape[0] != t.shape[0]:
            raise ValueError("z_t and t must have the same batch size")

        hidden = self.input_projection(z_t)
        time_hidden = self.time_projection(t[:, None])
        # t[:, None]之前，t的形状是[B](假如)，t[:, None]之后，t的形状就是[B, 1]

        hidden = hidden + time_hidden[:, None, :]
        # time_hidden的形状是[B, H]，那么time_hidden[:, None, :]的形状就是 [B, 1, H]
        hidden = torch.tanh(hidden)

        velocity = self.output_projection(hidden)

        return velocity
