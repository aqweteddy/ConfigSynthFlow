from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from ...base import BasePipeline


class TextChunker(BasePipeline):
    def __post_init__(
        self,
        text_lambda_col: str,
        tokenizer_path: str,
        output_col: str = '_chunked',
        chunk_size: tuple[int, int] = (300, 1000),
        overlap: int = 50,
    ):
        self.text_lambda_col = eval(text_lambda_col)
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

        self.splitter = TextSplitter.from_huggingface_tokenizer(
            self.tokenizer, (chunk_size[0], chunk_size[1]), overlap=overlap
        )
        
        self.output_col = output_col

    def run_each(self, dct: dict) -> dict:
        text = self.text_lambda_col(dct)
        if isinstance(text, list):
            result = []
            for t in text:
                chunks = self.splitter.chunks(t)
                result.extend(chunks)
        else:
            result = self.splitter.chunks(text)
        return {**dct, self.output_col: result}
