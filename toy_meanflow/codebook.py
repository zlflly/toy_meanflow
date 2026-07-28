import torch
from torch import nn

class FixedGaussianCodebook(nn.Module):
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

        generator = torch.Generator()
        generator.manual_seed(seed) # 使用单独的Generator的好处就是不会干扰程序中的其他随机状态的随机种子

        table = torch.randn(
            vocab_size,
            embedding_dim,
            generator=generator,
        )
        """创建码本，每一行对应一个token i的连续向量"""

        self.embedding_dim = embedding_dim
        self.register_buffer("table", table) # buffer不会计算梯度，不会优化

    def encode(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """会把每一个token id变成码本中对应的连续向量，这个token的维度从1变成embedding_dim"""
        return self.table[token_ids]

    def decode(
        self,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """使用最近邻搜索"""
        if vectors.shape[-1] != self.embedding_dim:
            raise ValueError("the last dimension must equal embedding_dim")

        original_shape = vectors.shape[:-1] # 只需要去掉最后一个维度，就可以得到原来的维度
        """[2, 8, embedding_dim]
            original_shape = [2, 8]
        """

        flat_vectors = vectors.reshape(
            -1, # -1表示让pytorch自己判断
            self.embedding_dim,
        ) # reshape，相当于把几个batch的待解码的向量拼在一起，最后一个维度是embedding_dim，第一个维度就是要解码的向量的个数

        distances = torch.cdist( # torch.cdist() 会计算每个待解码向量与每个码本向量之间的欧氏距离
            flat_vectors,
            self.table,
        )
        '''输出的形状是[batch_size*seq_len, vocab_size]（分别是两个二维向量的第一个维度），也就是[i, j] 表示第i个连续向量，与第j个token表示的连续向量之间的欧式距离'''

        token_ids = distances.argmin(dim=1) # 取最小值对应的tokne_id

        return token_ids.reshape(original_shape)
