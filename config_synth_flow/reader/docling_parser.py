import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import current_process
from pathlib import Path

import ftfy

from .base import BaseReader

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError:
    pass


class DoclingReader(BaseReader):
    required_packages = ["docling", "ftfy"]

    def post_init(
        self,
        data_path: str = None,
        file_list: str = None,
        num_thread: int = 2,
        pdf_ocr: bool = False,
        num_proc: int = 4,
        batch_size: int = 1000,
        fix_encoding: bool = True,
        resume: bool = False,
        doc_format: list[str] = ["xlsx", "docx", "pdf", "pptx", "md", "image"],
        debug: bool = False,
    ):
        """
        Initialize the DoclingReader.

        Args:
            data_path (str): Path to the data directory.
            file_list (str): Path to the file list.
            num_thread (int): Number of threads for PDF OCR. Defaults to 2.
            pdf_ocr (bool): Whether to perform OCR on PDFs. Defaults to False.
            num_proc (int): Number of processes for multiprocessing. Defaults to 4.
            batch_size (int): Batch size for processing. Defaults to 1000.
            fix_encoding (bool): Whether to fix text encoding issues. Defaults to True.
            doc_format (list[str]): List of document formats to process. Defaults to ["xlsx", "docx", "pdf", "pptx", "md", "image"].
            debug (bool): Debug mode flag. Defaults to False.
        """
        super().post_init(resume=resume)
        self.data_path = Path(data_path) if data_path else None
        self.file_list = Path(file_list) if file_list else None
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
        """
        Get the DocumentConverter for the given process index.

        Args:
            idx (int): Process index.

        Returns:
            DocumentConverter: The DocumentConverter instance.
        """
        if idx not in self._doc_converter_mapper:
            self._doc_converter_mapper[idx] = DocumentConverter(
                allowed_formats=self.doc_format,
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=self._pdf_option)
                },
            )
        return self._doc_converter_mapper[idx]

    def _process(self, file: Path) -> dict | None:
        """
        Process a single file.

        Args:
            file (Path): Path to the file.

        Returns:
            dict | None: Processed file information or None if processing fails.
        """
        process_name = current_process().name
        proc_id = int(process_name.split("-")[-1]) - 1

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
        """
        Run the multiprocessing pipeline.

        Args:
            file_list (Sequence[Path]): List of file paths to process.

        Yields:
            dict: Processed file information.
        """
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
        """
        Extract text from the ConversionResult.

        Args:
            file (ConversionResult): The ConversionResult object.

        Returns:
            str: Extracted text.
        """
        return file.document.export_to_markdown()

    def read(self):
        """
        Read documents from the data path.

        Yields:
            dict: Processed file information.
        """
        if self.data_path:
            self.logger.info(f"Reading documents from {self.data_path}")
            files = self.data_path.rglob("*")
            cnt = 0
        elif self.file_list:
            self.logger.info(f"Reading documents from {self.file_list}")
            with open(self.file_list) as f:
                files = [Path(line.strip()) for line in f]
            cnt = 0
            self.logger.info(f"Found {len(files)} files in the list.")

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
