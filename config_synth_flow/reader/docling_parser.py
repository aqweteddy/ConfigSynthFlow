from .base import BaseReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.datamodel.document import ConversionResult
from docling.datamodel.base_models import InputFormat
from pathlib import Path
import os


class DoclingReader(BaseReader):
    required_packages = ["docling"]
    def __post_init__(self, 
                      data_path: str,
                      num_thread: int = 10,
                      pdf_ocr: bool = False,
                      doc_format: list[str] = ["xlsx", 'docx', 'pdf', 'pptx', 
                                               'md', 'html',  'image'],
                      callback_lambda_funcs: list[str] = None
    ):
        self.data_path = Path(data_path)
        pdf_option = PdfPipelineOptions(
            do_ocr=pdf_ocr
        )
        self.doc_converter = DocumentConverter(
            allowed_formats=doc_format,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_option
                )
            }
        )
        self.num_thread = num_thread
        os.environ['OMP_NUM_THREADS'] = str(num_thread)
        
    def get_text(self, file: ConversionResult):
        return file.document.export_to_markdown(strict_text=True)
    
    def read(self):
        self.logger.info(f"Reading documents from {self.data_path}")
        files = [f for f in self.data_path.rglob("*") if f.is_file()]
        self.logger.info(f"There are {len(files)} files in the directory.")
        cnt = 0
        
        for file in self.doc_converter.convert_all(files, raises_on_error=False):
            cnt += 1
            yield {
                "text": self.get_text(file),
                "file_path": file.input.file.as_posix(),
            }
            if cnt % 100 == 0:
                self.logger.info(f"Parsed {cnt} documents.")
        
        self.logger.info(f"Finished reading {cnt} documents.")
