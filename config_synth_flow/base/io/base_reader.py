from hashlib import md5
from pathlib import Path

from ..pipeline import BasePipeline, DictsGenerator


class BaseReader(BasePipeline):
    resume: bool = False
    writer_output_path: Path | None = None

    def post_init(self, resume: bool = False):
        self.resume = resume
        self._tmp_save_processed_ids = []
        self._unique_ids = set()

    def read(self) -> DictsGenerator: ...

    def __call__(self) -> DictsGenerator:
        skipped_cnt = 0
        for dct in self.read():
            if "hash_id" in dct:
                id = dct["hash_id"]
            else:
                id = self.get_unique_id(dct)
                dct["hash_id"] = id

            if self.resume and id in self._unique_ids:
                skipped_cnt += 1
                continue

            self.add_ids(id)
            yield dct

    def add_ids(self, id: str) -> None:
        self._tmp_save_processed_ids.append(id)
        self._unique_ids.add(id)

        if len(self._tmp_save_processed_ids) >= 1000 and self.writer_output_path is not None:
            with open(self.writer_output_path / "processed_ids.txt", "a") as f:
                f.write("\n".join(self._tmp_save_processed_ids) + "\n")
            self._tmp_save_processed_ids = []

    @staticmethod
    def get_unique_id(obj: dict) -> str:
        """
        Generate a unique id for the given object.

        Args:
            obj (dict): Object to generate id for.

        Returns:
            str: Unique id.
        """
        items = sorted(obj.items(), key=lambda x: x[0])
        return md5(str(items).encode()).hexdigest()

    def __del__(self):
        """
        Destructor to save any remaining processed ids.
        """
        if len(self._tmp_save_processed_ids) > 0 and self.writer_output_path is not None:
            with open(self.writer_output_path / "processed_ids.txt", "a") as f:
                f.write("\n".join(self._tmp_save_processed_ids) + "\n")
            self._tmp_save_processed_ids = []

    def set_writer_output_path(self, writer_output_path: Path | str | None = None) -> None:
        self.writer_output_path = writer_output_path
        if self.writer_output_path is None:
            return

        if isinstance(self.writer_output_path, str):
            self.writer_output_path = Path(self.writer_output_path)
        self.writer_output_path.mkdir(parents=True, exist_ok=True)

        if self.resume and (self.writer_output_path / "processed_ids.txt").exists():
            with open(self.writer_output_path / "processed_ids.txt") as f:
                self._unique_ids = set(line.strip() for line in f.readlines())
                self.logger.info(f"Resuming from {len(self._unique_ids)} samples")
        else:
            self.logger.info("Starting from scratch")
            self._unique_ids = set()
            with open(self.writer_output_path / "processed_ids.txt", "w") as f:
                pass
