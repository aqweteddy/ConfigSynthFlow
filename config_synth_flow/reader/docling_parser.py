from .base import BaseReader
import ftfy

from concurrent.futures import ProcessPoolExecutor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import InputFormat
from pathlib import Path
import os


class DoclingReader(BaseReader):
    required_packages = ["docling", "ftfy"]

    def __post_init__(
        self,
        data_path: str,
        num_thread: int = 8,
        pdf_ocr: bool = False,
        num_proc: int = 4,
        batch_size: int = 1000,
        fix_encoding: bool = True,
        doc_format: list[str] = ["xlsx", "docx", "pdf", "pptx", "md", "html", "image"],
        debug: bool = False,
    ):
        self.data_path = Path(data_path)
        self.num_proc = num_proc
        self.fix_encoding = fix_encoding
        self.debug = debug
        self.batch_size = batch_size

        pdf_option = PdfPipelineOptions(do_ocr=pdf_ocr)
        self.doc_converter_mapper = [
            DocumentConverter(
                allowed_formats=doc_format,
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_option)
                },
            )
            for _ in range(num_proc)
        ]
        self.num_thread = num_thread
        os.environ["OMP_NUM_THREADS"] = str(num_thread)

    def _process(self, files: list[Path], idx: int) -> list[dict]:
        result = []
        for doc in self.doc_converter_mapper[idx].convert_all(
            files, raises_on_error=False
        ):
            text = self.get_text(doc)

            if self.fix_encoding and ftfy.is_bad(text):
                text = ftfy.fix_text(text)

            if self.fix_encoding and ftfy.is_bad(doc.input.file.as_posix()):
                file_path = ftfy.fix_text(doc.input.file.as_posix())
            else:
                file_path = doc.input.file.as_posix()

            result.append({"text": text, "file_path": file_path})
        return result

    def mp_run(self, file_list: list[Path]):
        cnt = 0
        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            for i in range(0, len(file_list), self.batch_size):
                flist = file_list[i : i + self.batch_size]
                results = list(executor.map(
                    self._process,
                    [file_list[j :: self.num_proc] for j in range(self.num_proc)],
                    range(self.num_proc),
                ))
                cnt += len(flist)
                yield from [item for sublist in results for item in sublist]
                self.logger.info(f"Processed {cnt} files.")

    def get_text(self, file: ConversionResult):
        return file.document.export_to_markdown()

    def read(self):
        self.logger.info(f"Reading documents from {self.data_path}")
        files = [f for f in self.data_path.rglob("*") if f.is_file()][:10000]

        if self.debug:
            files = files[:5]

        self.logger.info(f"There are {len(files)} files in the directory.")
        cnt = 0

        for file in self.mp_run(files):
            cnt += 1
            yield file

        self.logger.info(f"Finished reading {cnt} documents.")
