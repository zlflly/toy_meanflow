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
    def __init__(self, block_size: int ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self._token_ids: list[int] = [] # 用来保存还没组成完整block的token
        # 开头的下划线表示这是类的内部数据

    def add( # 向缓冲区内加一段新的token，如果已经组成完整的blocks，就返回
        self,
        token_ids: Sequence[int],
    ) -> list[list[int]]:
        self._token_ids.extend(token_ids) # 不用append，append会形成嵌套列表

        blocks: list[list[int]] = []

        while len(self._token_ids) >= self.block_size:
            block = self._token_ids[:self.block_size] # 取出前 block_size 个 token
            block.append(block)

            del self._token_ids[:self.block_size]

        return blocks

    @property
    def remaining_token_ids(self) -> list[int]:
        return self._token_ids.copy()