import torch
from torch import nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int, # 那么每个注意力头处理 hidden_dim // num_heads
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_projection = nn.Linear( # 第一个参数表示输入维度，第二参数是输出维度
            hidden_dim,
            3 * hidden_dim,
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, length, hidden_dim]")

        batch_size, sequence_length, hidden_dim = hidden.shape # 把hidden的形状，分别赋值给前面三个变量

        if hidden_dim != self.hidden_dim:
            raise ValueError("the last dimension does not match hidden_dim")

        qkv = self.qkv_projection(hidden)

        query, key, value = qkv.chunk( # 意思是沿着最后一个维度平均切成三份
            chunks=3,
            dim=-1,
        )

        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        query = query.transpose(1, 2) # 指的是把索引为1和2的维度交换，也就是从 [B, L, heads, head_dim] 变成 [B, heads, L, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # [B, heads, L, head_dim]

        attention_scores = (
            query @ key.transpose(-2, -1) # 指的是把最后两个维度交换
        ) / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attended = attention_weights @ value
        # attended：[B, heads, L, head_dim]

        attended = attended.transpose(1, 2)

        attended = attended.reshape(
            batch_size,
            sequence_length,
            self.hidden_dim,
        )

        output = self.output_projection(attended)

        return output


class TinyMeanFlowModel(nn.Module):
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
            2,
            hidden_dim,
        )
        # 把变量时间对(t, r)扩展到hidden_dim，其实输入的是(t, t-r)，之后会把这个时间向量加到序列的每个位置上

        self.output_projection = nn.Linear(
            hidden_dim,
            data_dim,
        ) # 把投影重新映射回data_dim

    def forward(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.ndim != 3:
            raise ValueError("z_t must have shape [batch, length, dim]")

        if t.ndim != 1 or r.ndim != 1:
            raise ValueError("t and r must have shape [batch]")

        batch_size = z_t.shape[0]

        if r.shape[0] != batch_size:
            raise ValueError("z_t and r must have the same batch size")

        if t.shape[0] != batch_size:
            raise ValueError("z_t and t must have the same batch size")

        if torch.any(r > t): # r > t 比较的结果是布尔张量
            raise ValueError("r must be less than or equal to t")

        hidden = self.input_projection(z_t)

        interval_length = t - r
        time_features = torch.stack([t, interval_length], dim = 1) # stack是在新的维度上进行拼接，能够保留每个样本有两个时间特征的结构，如果是cat，那么就是在原有的维度上进行拼接，就损失了这个结构
        time_hidden = self.time_projection(time_features)

        hidden = hidden + time_hidden[:, None, :]
        # time_hidden的形状是[B, H]，那么time_hidden[:, None, :]的形状就是 [B, 1, H]
        hidden = torch.tanh(hidden)

        average_velocity = self.output_projection(hidden)

        return average_velocity
