from .base import BaseReader
import ftfy
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
from multiprocessing import current_process

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )

    from docling.datamodel.document import ConversionResult
    from docling.datamodel.base_models import InputFormat
except ImportError:
    pass


class DoclingReader(BaseReader):
    required_packages = ["docling", "ftfy"]

    def __post_init__(
        self,
        data_path: str,
        num_thread: int = 2,
        pdf_ocr: bool = False,
        num_proc: int = 4,
        batch_size: int = 1000,
        fix_encoding: bool = True,
        doc_format: list[str] = ["xlsx", "docx", "pdf", "pptx", "md", "image"],
        debug: bool = False,
    ):
        self.data_path = Path(data_path)
        self.num_proc = num_proc
        self.fix_encoding = fix_encoding
        self.debug = debug
        self.batch_size = batch_size
        self.doc_format = doc_format

        self.debug = 5 if debug is True else debug

        self._pdf_option = PdfPipelineOptions(do_ocr=pdf_ocr, num_threads=num_thread)
        self.doc_postfix = [f".{f}" for f in self.doc_format if f != "image"]
        if "image" in self.doc_format:
            self.doc_postfix += [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        
        self._doc_converter_mapper = {}
        self.num_thread = num_thread
        os.environ["OMP_NUM_THREADS"] = str(num_thread)
    
    def get_doc_converter(self, idx: int) -> "DocumentConverter":
        if idx not in self._doc_converter_mapper:
            self._doc_converter_mapper[idx] = DocumentConverter(
                allowed_formats=self.doc_format,
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=self._pdf_option)
                },
            )
        return self._doc_converter_mapper[idx]
        

    def _process(self, file: Path) -> dict | None:
        process_name = current_process().name
        proc_id = int(process_name.split('-')[-1]) - 1
        
        if not file.is_file() or file.suffix not in self.doc_postfix:
            return
        
        try:
            doc = self.get_doc_converter(proc_id).convert(file, raises_on_error=False)
            text = self.get_text(doc)
        except Exception as e:
            self.logger.error(f"Error processing {file}: {e}")
            return
        
        if self.fix_encoding and ftfy.is_bad(text):
            text = ftfy.fix_text(text)
            if ftfy.is_bad(text):
                text = ""

        if self.fix_encoding and ftfy.is_bad(doc.input.file.as_posix()):
            file_path = ftfy.fix_text(doc.input.file.as_posix())
        else:
            file_path = doc.input.file.as_posix()

        if text:
            return {"text": text, "file_path": file_path}
        return None

    def mp_run(self, file_list: Sequence[Path]):
        cnt = 0

        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            result_iter = executor.map(self._process, file_list)
            
            for result in result_iter:
                cnt += 1
                if cnt % 1000 == 0:
                    self.logger.info(f"Processed {cnt} files.")
                yield result
                
            
        self.logger.info(f"Finished processing {cnt} files.")

    def get_text(self, file: ConversionResult):
        return file.document.export_to_markdown()

    def read(self):
        self.logger.info(f"Reading documents from {self.data_path}")
        files = self.data_path.rglob("*")
        cnt = 0

        for file in self.mp_run(files):
            if file:
                yield file
                cnt += 1
            
            if cnt % 1000 == 0:
                self.logger.info(f"Read {cnt} documents.")
            
            if self.debug and cnt >= self.debug:
                self.logger.info(f"Debug mode: Read {cnt} documents.")
                break
        self.logger.info(f"Finished reading {cnt} documents.")
