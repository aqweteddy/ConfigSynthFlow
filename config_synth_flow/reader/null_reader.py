import uuid

from config_synth_flow.base import BaseReader, DictsGenerator


class NullReader(BaseReader):
    def post_init(
        self,
        num_data: int = 10,
    ):
        super().post_init()
        self.num_data = num_data

    def read(self) -> DictsGenerator:
        for _ in range(self.num_data):
            yield {"hash_id": str(uuid.uuid4())}
