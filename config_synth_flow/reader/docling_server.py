import time
from pathlib import Path
from typing import Any

import httpx

from config_synth_flow.base import BaseReader

DEFAULT_KWARGS = {
    "from_formats": ["docx", "pptx", "html", "image", "pdf", "asciidoc", "md", "xlsx"],
    "to_formats": ["md"],
    # "image_export_mode": "placeholder",
    "do_ocr": False,
    "force_ocr": False,
    "ocr_engine": "rapidocr",
    "ocr_lang": ["english", "chinese"],
    "pdf_backend": "dlparse_v4",
    "table_mode": "fast",
    "abort_on_error": False,
    "return_as_file": False,
}


class DoclingServerReader(BaseReader):
    required_packages: list[str] = ["httpx"]

    def post_init(
        self,
        url: str,
        data_path: str,
        text_col: str = "text",
        kwargs: dict[str, Any] = None,
        resume: bool = False,
    ):
        """
        https://github.com/docling-project/docling-serve/blob/main/docs/usage.md
        """
        super().post_init(resume=resume)
        self.upload_url = url.strip("/") + "/v1alpha/convert/file/async"
        self.status_url = url.strip("/") + "/v1alpha/status/poll"
        self.fetch_url = url.strip("/") + "/v1alpha/result"
        self.text_col = text_col
        self.data_path = data_path

        self.kwargs = {**DEFAULT_KWARGS, **(kwargs or {})}

    def read(self):
        files = list(Path(self.data_path).glob("*"))
        files.sort()
        self.logger.info(f"There are {len(files)} files to read.")

        tasks = []
        for file in files:
            tasks.append(self.submit(file))

        req_cnt = 0
        while len(tasks) > 0:
            for task in tasks:
                req_cnt += 1
                status = self.get_status(task)
                if status == "success":
                    tasks.remove(task)
                    text = self.fetch(task["task_id"])
                    dct = {"hash_id": self.get_unique_id(text), "text": text}
                    self.logger.info(f"Processed {file.name}.")
                    yield dct
                elif status == "failure" or not status:
                    self.logger.error(f"Task {task['task_id']} failed.")
                    tasks.remove(task)

                if req_cnt % 10 == 0:
                    time.sleep(10)
            time.sleep(10)

    def submit(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            files = {
                "files": (file_path.name, f, "application/pdf")
            }
            with httpx.Client() as client:
                response = client.post(self.upload_url, data=self.kwargs, files=files)
                return response.json()

    def get_status(self, task: dict) -> bool:
        response = httpx.get(f"{self.status_url}/{task['task_id']}")
        task = response.json()
        return task.get("task_status")

    def fetch(self, task_id: str) -> str:
        response = httpx.get(f"{self.fetch_url}/{task_id}")
        return response.json()["result"]
