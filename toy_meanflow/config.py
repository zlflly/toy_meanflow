from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_name: str = "stas/openwabtext-10k" 
    split:str = "train"
    seq_len: int = 64 # 这个之后就是data当中的block_size
    shuffle_buffer: int = 2000
    num_workers: int = 0 # 表示使用多少个额外进程来读取数据
    seed: int = 42

