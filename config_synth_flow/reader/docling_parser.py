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
    required_packages = ["docling"]
    def __post_init__(self, 
                      data_path: str,
                      num_thread: int = 8,
                      pdf_ocr: bool = False,
                      num_proc: int = 4,
                      fix_encoding: bool = True,
                      doc_format: list[str] = ["xlsx", 'docx', 'pdf', 'pptx', 
                                               'md', 'html',  'image'],
    ):
        self.data_path = Path(data_path)
        self.num_proc = num_proc
        self.fix_encoding = fix_encoding
        
        pdf_option = PdfPipelineOptions(
            do_ocr=pdf_ocr
        )
        self.doc_converter_mapper = [DocumentConverter(
            allowed_formats=doc_format,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_option
                )
            }
        ) for _ in range(num_proc)]
        self.num_thread = num_thread
        os.environ['OMP_NUM_THREADS'] = str(num_thread)
    
    def mp_run(self, file_list: list[Path]):
        def __process(files: list[Path], idx: int) -> list[dict]:
            cnt = 0
            result = []
            for doc in self.doc_converter_mapper[idx].convert_all(files):
                cnt += 1
                if cnt % 100 == 0:
                    self.logger.info(f"Processes {idx} | Parsed {cnt} documents.")
                text = self.get_text(doc)
                
                if self.fix_encoding and ftfy.is_bad(text):
                    text = ftfy.fix_text(text)
                
                result.append({
                    "text": text,
                    "file_path": doc.input.file.as_posix(),
                })
            return result
        
        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            results = executor.map(__process, file_list, self.num_proc)
        
        return [item for sublist in results for item in sublist]
            
    def get_text(self, file: ConversionResult):
        return file.document.export_to_markdown()
    
    def read(self):
        self.logger.info(f"Reading documents from {self.data_path}")
        files = [f for f in self.data_path.rglob("*") if f.is_file()]
        self.logger.info(f"There are {len(files)} files in the directory.")
        cnt = 0
        
        for file in self.mp_run(files):
            cnt += 1
            yield file
        
        self.logger.info(f"Finished reading {cnt} documents.")
