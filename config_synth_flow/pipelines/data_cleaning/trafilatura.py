from trafilatura import extract
from concurrent.futures import ProcessPoolExecutor
import ftfy
from ...base import BasePipeline, DictsGenerator


class TrafilaturaPipeline(BasePipeline):
    required_packages = ["trafilatura"]

    def __post_init__(
        self,
        html_col: str = "text",
        output_col: str = "text",
        num_proc: int = 10,
    ):
        self.html_col = html_col
        self.output_col = output_col
        self.num_proc = num_proc

    def run_each(self, dct: dict) -> dict:
        text = extract(
            dct[self.html_col],
            fast=True,
            output_format="markdown",
            target_language="zh",
            include_tables=True,
            include_comments=False,
            favor_precision=True,
            include_images=False,
            deduplicate=True,
        ) or ''
        if ftfy.is_bad(text):
            text = ftfy.fix_text(text)
            if ftfy.is_bad(text):
                text = ''
        dct[self.output_col] = text
        return dct

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        with ProcessPoolExecutor(self.num_proc) as executor:
            yield from executor.map(self.run_each, dcts)
