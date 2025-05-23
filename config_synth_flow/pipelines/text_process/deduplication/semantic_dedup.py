import numpy as np
from litellm import embedding
from semhash import SemHash
from semhash.utils import Encoder
from tqdm import tqdm

from config_synth_flow.base import BasePipeline, DictsGenerator


class LiteLLMEmbedding(Encoder):
    def __init__(self, litellm_emb_kwargs: dict, max_tokens: int = 2048, batch_size: int = 100):
        self.litellm_emb_kwargs = litellm_emb_kwargs
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def encode(self, text: str | list[str]) -> list[float]:
        if isinstance(text, str):
            text = [text]

        result = []
        for i in tqdm(range(0, len(text), self.batch_size), desc="Embedding"):
            text_batch = text[i : i + self.batch_size]
            text_batch = [t[: self.max_tokens] for t in text_batch]
            res = embedding(input=text_batch, **self.litellm_emb_kwargs)
            emb = np.array([e["embedding"] for e in res.data])
            result.append(emb)
        return np.concatenate(result, axis=0)


class SemanticDeduplicator(BasePipeline):
    required_packages = ["semhash"]

    def post_init(
        self,
        litellm_emb_kwargs: dict,
        text_col: str,
        threshold: float = 0.9,
        use_ann: bool = True,
    ) -> None:
        self.encoder = LiteLLMEmbedding(litellm_emb_kwargs)
        self.text_col = text_col
        self.use_ann = use_ann
        self.threshold = threshold

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        self.logger.info("Start Semantic Deduplication")
        ds = list(tqdm(dcts, desc="Get Text"))
        semhash = SemHash.from_records(
            records=ds,
            columns=[self.text_col],
            model=self.encoder,
            use_ann=self.use_ann,
        )
        dedup_results = semhash.self_deduplicate(threshold=self.threshold)

        self.logger.info(f"Exact duplicate ratio: {dedup_results.exact_duplicate_ratio}")
        self.logger.info(f"Duplicated ratio: {dedup_results.duplicate_ratio}")
        self.logger.info(f"Removed {len(dedup_results.duplicates)} samples")

        dedup_items = dedup_results.deduplicated

        return dedup_items
