from hashlib import md5
from pathlib import Path

from ..base import BasePipeline, DictsGenerator


class BaseReader(BasePipeline):
    """
    Base class for the reader.
    Read the dataset and generate the `dict[str, Any]` for the pipeline.

    override the `read` method to read the dataset. You can also override the `get_unique_id` method to generate the unique id for each sample.

    Notice that __call__ method is implemented to generate the unique id for each sample.

    property:
    - resume: bool: Resume the previous run. Default to False. When True, it will load the processed dataset and skip the processed samples.
    - writer_output_path: Path: Output path of the writer. This is used to check the processed dataset.
    """

    resume: bool = False
    writer_output_path: Path = None

    def post_init(self, resume: bool = False):
        """
        Initialize the BaseReader.

        Args:
            resume (bool): Whether to resume the previous run. Defaults to False.
        """
        self.resume = resume
        self._tmp_save_processed_ids = []

    def __call__(self) -> DictsGenerator:
        """
        Implicitly call the `read` method, generate the unique id for each sample and yield the sample.

        Yields:
            dict: Sample with unique id.
        """
        skipped_cnt = 0
        for dct in self.read():
            if "hash_id" in dct:
                id = dct["hash_id"]
            else:
                id = self.get_unique_id(dct)
                dct["hash_id"] = id

            if self.resume and id in self.unique_id_set:
                skipped_cnt += 1
            else:
                self.add_id(id)
                yield dct

    def add_id(self, id: str) -> None:
        """
        Add a unique id to the set of processed ids.

        Args:
            id (str): Unique id to add.
        """
        self.unique_id_set.add(id)
        self._tmp_save_processed_ids.append(id)
        if len(self.unique_id_set) % 100 == 0:
            with open(self.writer_output_path / "processed_ids.txt", "a") as f:
                f.write("\n".join(self._tmp_save_processed_ids) + "\n")
            self._tmp_save_processed_ids = []

    def read(self) -> DictsGenerator:
        """
        Implement the read method to read the dataset.

        Yields:
            dict: Sample data.
        """
        ...

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

    def set_writer_output_path(self, output_path: str | Path | None) -> None:
        """
        Set the output path of the writer. This is used to check the processed dataset.

        Args:
            output_path (str | Path | None): Output path of the writer.
        """
        output_path = output_path or "./.tmp/"

        if isinstance(output_path, str):
            output_path = Path(output_path)

        self.writer_output_path = output_path
        if self.resume and (output_path / "processed_ids.txt").exists():
            with open(output_path / "processed_ids.txt") as f:
                self.unique_id_set = set(f.read().split("\n"))
                self.logger.info(
                    f"already processed {len(self.unique_id_set)} samples."
                )
        else:
            self.unique_id_set = set()

        output_path.mkdir(parents=True, exist_ok=True)

        if not self.resume:
            text_id_fp = open(output_path / "processed_ids.txt", "w")
            text_id_fp.close()

    def __del__(self):
        """
        Destructor to save any remaining processed ids.
        """
        if len(self._tmp_save_processed_ids) > 0:
            with open(self.writer_output_path / "processed_ids.txt", "a") as f:
                f.write("\n".join(self._tmp_save_processed_ids) + "\n")
            self._tmp_save_processed_ids = []
