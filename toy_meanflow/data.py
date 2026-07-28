import torch
from torch.utils.data import Dataset, DataLoader

from collections.abc import Sequence

def split_into_blocks(
        token_ids: Sequence[int], # sequence允许[]和()两种格式
        block_size: int, # 表示一个小块的token数，最后一个长度不够的会被丢掉
) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_complete_blocks = len(token_ids) // block_size
    usable_length = num_complete_blocks*block_size

    blocks: list[list[int]] = []
    # 外层list包含多个训练样本
    # 内层list是一个固定长度的token序列

    for start in range(0, usable_length, block_size): # 遍历每一个block的起始索引
        end = start + block_size # 计算相应的结束位置
        block = list(token_ids[start:end]) # 切片之后，再存入block这个嵌套列表
        blocks.append(block)

    return blocks

class TokenBuffer:
    """创建一个buffer容器，传入block大小，
    使用的时候，传入编码好的token id list，如果满足`block`的大小后，返回满足的block list
    """
    def __init__(self, block_size: int ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self._token_ids: list[int] = [] # 用来保存还没组成完整block的token
        # 开头的下划线表示这是类的内部数据

    def add( 
        self,
        token_ids: Sequence[int],
    ) -> list[list[int]]:
        """向缓冲区内加一段新的token，如果已经组成完整的blocks，就返回"""
        self._token_ids.extend(token_ids) # 不用append，append会形成嵌套列表

        blocks: list[list[int]] = [] # 用来收集完整的block

        while len(self._token_ids) >= self.block_size:
            block = self._token_ids[:self.block_size] # 取出前 block_size 个 token
            blocks.append(block)

            del self._token_ids[:self.block_size] # 在缓冲区中删除这次组成block的tokens

        return blocks

    @property # 可以把一个类内函数的访问方式变成像属性一样
    def remaining_token_ids(self) -> list[int]:
        return self._token_ids.copy()

class BlockDataset(Dataset): 
    """其实就是把输入的dataset存下来（list[list[int]]，并且赋予了查看长度和索引的功能
    """

    def __init__(
        self,
        blocks:list[list[int]], # 接受外部输入
    ) -> None:
        if len(blocks) == 0:
            raise ValueError("block must not be empty")

        block_size = len(blocks[0])
        """
        blocks = [
        [72, 101, 108, 108, 111, 32, 77, 101],   # block 0
        [97, 110, 70, 108, 111, 119, 33, 256],    # block 1
        ]

        len(blocks[0])就是其中一个block的长度
        """

        if block_size == 0:
            raise ValueError("block must not be empty")

        for block in blocks: # 确保所有的block等长
            # 因为后面要根据block创建batch [batch_size, seq_len]
            if len(block) != block_size:
                raise ValueError(
                    "all blocks must have the same length"
                ) 

        self._blocks = blocks # 把所有的blocks在dataset当中存下来
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self._blocks) # 这里起始存的是block的块数，之后构建的batch，也是返回这个

    def __getitem__(
        self,
        index:int # 表示取的是第几个block
    ) -> torch.Tensor:
        block = self._blocks[index]

        return torch.tensor(
            block,
            dtype=torch.long, # 转化成张量，64 位整数
        )

def build_dataloader(
    dataset: BlockDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """把 Dataset 包装成一个能按 batch 取数据的 DataLoader。

    给一个装了很多样本的 Dataset，再加上"一次取几条"的规则，
    返回一个 DataLoader，之后可以用 for 循环一个一个 batch 地取数据。

    参数：
        dataset:已经装好所有样本的 Dataset，比如 BlockDataset。它能告诉你"总共有多少条数据"和"第 i 条数据是什么"。
        batch_size: 每个 batch 装几条样本。比如 batch_size=2，就表示每次从 Dataset 里取 2 条，拼成一个 batch。
        shuffle: 是否打乱顺序。True 表示每轮取数据前随机打乱，False 表示按顺序取。
        num_workers: 额外开几个子进程帮忙读数据。0 表示只用主进程，>0 可以加快速度，但也更耗内存。

    返回：
        一个 DataLoader 对象，可以用 for 循环遍历，每次取出一个 batch。
        例如 for batch in dataloader: 每次的 batch 形状是 (batch_size, seq_len)。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return DataLoader( # 下面都是Dataloader方法需要补充的参数
        dataset = dataset,
        batch_size = batch_size, # 表示一次取出多少条样本
        shuffle = shuffle, # 决定每轮遍历是否打乱样本顺序
        num_workers = num_workers, # 表示额外使用多少个进程读取数据
        drop_last = False, # 表示不丢弃尾端，最后一个不完整的batch（也就是说不满足batch_size）也会保留
    )
    """
    这个的作用是，生成索引列表，然后利用BLockDataset当中的索引方法，一条一条索引出batch
    返回一个按照索引调整好的batch，（因为有shuffle的情况），后面也方便使用iter和next方法
    """
