import torch
from torch import nn

class FixedGussianCodebook(nn.Module):
    """为了把token id转化成连续向量，需要创建一个固定的编码本"""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seed: int = 42,
    ) -> None:
        super().__init__() # 这一句是调用nn.Moudule的内部初始化机制

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        generator = torch.Gnenrator()
        generator.manual_seed(seed) # 使用单独的Generator的好处就是不会干扰程序中的其他随机状态的随机种子

        table = torch.randn(
            vocab_size,
            embedding_dim,
            generator=generator,
        )
        """创建码本，每一行对应一个token i的连续向量"""

        self.register_buffer("table", table) # buffer不会计算梯度，不会优化

    def encode(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """会把每一个token id变成码本中对应的连续向量，这个token的维度从1变成embedding_dim"""
        return self.table[token_ids]