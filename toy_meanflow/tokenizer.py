from collections.abc import Iterable

class ByteTokenizer:
    eos_id: int = 256
    vocab_size: int = 257

    def encode(
            self,
            text: str,
            add_eos: bool = True,
    ) -> list[int]:
        token_ids = list(text.encode("utf-8"))

        if add_eos: # 默认add_eos = True
            token_ids.append(self.eos_id)

        return token_ids

    def decode(
            self,
            token_ids: Iterable[int],
            stop_at_eos: bool = True,
    ) -> str:
        byte_values: list[int] = []

        for token_id in token_ids:
            token_id = int(token_id)

            if stop_at_eos and token_id == self.eos_id:
                break

            if 0<=token_id <=255:
                byte_values.append(token_id)

        return bytes(byte_values).decode(
            "utf-8",
            errors="replace"
        )