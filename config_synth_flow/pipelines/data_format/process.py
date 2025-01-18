from ...base import BasePipeline

class ListFlatter(BasePipeline):
    def __post_init__(self, text_col: str):
        self.text_col = text_col
    
    def run_each(self, dct: dict):
        assert isinstance(dct[self.text_col], list)
        text_list = dct.pop(self.text_col)
        for text in text_list:
            if isinstance(text, str):
                yield {**dct, self.text_col: text}
            elif isinstance(text, dict):
                yield {**dct, **text}
            else:
                raise ValueError(f"Unsupported type: {type(text)}")
