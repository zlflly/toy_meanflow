import torch

class UniformTimeSampler: # 封装成类，便于之后替换采样策略
    def sample(
        self,
        batch_size: int, # 之后需要生成batch_size个时间，一个样本一个
        device: torch.device, # 时间张量和模型数据处于同一设备
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        t = torch.rand(
            batch_size,
            device=device,
            dtype=dtype,
        )

        return t

class UniformTimePairSampler:
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """这里随机采样两个时间点，然后排序，小的作为r"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        pair = torch.rand(
            batch_size,
            2,
            device=device,
            dtype=dtype,
        )

        r = pair.min(dim=1).values
        t = pair.max(dim=1).values

        return r, t