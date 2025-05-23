from transformers import AutoTokenizer

from config_synth_flow.base import BasePipeline


class TokenCounter(BasePipeline):
    def post_init(
        self,
        text_col: str = "text",
        tokenizer_path: str = "Qwen/QwQ-32B",
        output_col: str = "token_count",
        tokenized_col: str = None,
    ) -> None:
        self.text_col = text_col
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.output_col = output_col
        self.tokenized_col = tokenized_col

    def run_each(self, dct: dict) -> dict:
        text = dct[self.text_col]
        tokens = self.tokenizer.tokenize(text)
        dct[self.output_col] = len(tokens)
        if self.tokenized_col is not None:
            dct[self.tokenized_col] = tokens
        return dct
