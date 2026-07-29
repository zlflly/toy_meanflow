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
    """现在初始化允许传入参数`non_equal_ratio`，传入所有的时间对当中，`r≠t`的比例，默认是0.75"""
    def __init__(
        self,
        non_equal_ratio:float = 0.75,
    ) -> None:
        if not 0.0 <= non_equal_ratio <= 1.0:
            raise ValueError("non_equal_ratio must be between 0 and 1")

        self.non_equal_ratio = non_equal_ratio

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

        keep_interval = ( # 这个函数的含义就是是否保留这个`r≠t`的时间区间
            torch.rand(
                batch_size,
                device=device,
            ) # 这里是生成和batch_size大小相同的01之间的随机数，如果这个数小于`non_equal_ratio`，那么就保留，否则这一组的r更换成t
            < self.non_equal_ratio
        )

        r = torch.where( # 这个函数的意思是，把所有为false的位置的r，用t来替换
            keep_interval,
            r,
            t,
        )

        return r, t