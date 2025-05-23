import random

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config_synth_flow.base import BasePipeline


class TextSplitter(BasePipeline):
    def post_init(
        self,
        tokenizer_path: str = "Qwen/Qwen2.5-32B-Instruct",
        text_col: str = "text",
        output_col: str = "text_list",
        chunk_size: int | list[int] = 1000,
        chunk_overlap: int | list[int] = 200,
        separators: list[str] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.chunk_size = chunk_size if isinstance(chunk_size, list) else [chunk_size]
        self.chunk_overlap = chunk_overlap if isinstance(chunk_overlap, list) else [chunk_overlap]
        self.separators = separators if separators else ["\n\n", "\n", "。", "."]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size[0],
            chunk_overlap=self.chunk_overlap[0],
            separators=self.separators,
            # keep_separator=True,
            length_function=self.get_length,
        )
        self.text_col = text_col
        self.output_col = output_col

    def get_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def run_each(self, dct: dict) -> dict:
        if len(self.chunk_size) > 1:
            chunk_size = random.choice(self.chunk_size)
            chunk_overlap = random.choice(self.chunk_overlap)
            self.splitter._chunk_overlap = chunk_overlap
            self.splitter._chunk_size = chunk_size
        dct[self.output_col] = self.splitter.split_text(dct[self.text_col])
        return dct
