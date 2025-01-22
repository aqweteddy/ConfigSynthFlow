from ...base import BasePipeline, DictsGenerator
from .infinity import InfinityApiEmbedder
from typing import Generator
from qdrant_client import QdrantClient, models


class QdrantApiRetriever(BasePipeline):
    required_packages = ["qdrant_client"]
    
    def __post_init__(
        self,
        embedder: InfinityApiEmbedder,
        api_base: str,
        api_port: int = 6333,
        collection_name: str = "default",
        qdrant_text_key: str = "text",
        k_range: tuple[int, int] = (0, 5),
        similarity_threshold: float = 0.4,
        output_col: str = "_retrieved",
        exact_deduplication: bool = True,
        timeout: int = 20,
    ):
        """Retreive documents from QdrantDB

        Args:
            embedder (InfinityApiEmbedder): Embedder pipeline to get embeddings. Must have `get_embeddings` method.
            api_base (str): QdrantDB API base URL
            api_port (int, optional): QdrantDB API port. Defaults to 6333.
            collection_name (str, optional): QdrantDB collection name. Defaults to "default".
            qdrant_text_key (str, optional): Key to get text from QdrantDB payload. Defaults to "text".
            k_range (tuple[int, int], optional): Range of retrieved documents. Defaults to [0, 5).
            similarity_threshold (float, optional): Minimum similarity score to retrieve. Defaults to 0.5.
            output_col (str, optional): Output column name. Defaults to "_retrieved".
            timeout (int, optional): API request timeout. Defaults to 20.
        """
        self.embedder = embedder
        self.host = api_base.strip("/")
        self.collection_name = collection_name
        self.k_range = k_range
        self.similarity_threshold = similarity_threshold
        self.output_col = output_col
        self.qdrant_text_key = qdrant_text_key
        self.timeout = timeout
        self.exact_deduplication = exact_deduplication

        self.qdrant_cli = QdrantClient(url=self.host, port=api_port)

    def __chunk_batch(self, dcts: DictsGenerator, batch_size: int = None):  # type: ignore
        batch = []
        batch_size = batch_size or self.embedder.batch_size
        for i, dct in enumerate(dcts):
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = []
            batch.append(dct)

        if batch:
            yield batch

    def dedup_text_content(self, text_list: list[dict]) -> Generator[dict, None, None]:
        used_texts = set()
        result = []
        for text in text_list:
            if text[self.qdrant_text_key] not in used_texts:
                used_texts.add(text[self.qdrant_text_key])
                result.append(text)
        return result
    
    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        for chunked_dcts in self.__chunk_batch(dcts):
            queries = [self.embedder.query_lambda_col(d) for d in chunked_dcts]
            embeddings = self.embedder.get_embeddings(queries)
            
            reqs = [
                models.SearchRequest(
                    vector=emb,
                    limit=self.k_range[1],
                    score_threshold=self.similarity_threshold,
                    with_payload=True,
                )
                for emb in embeddings
            ]

            retrieved_list = self.qdrant_cli.search_batch(
                collection_name=self.collection_name,
                requests=reqs,
                timeout=self.timeout,
            )

            for dct, retrieved in zip(chunked_dcts, retrieved_list):
                output = [
                    {'text': r.payload[self.qdrant_text_key], 
                     'score': r.score, 
                     'collection': self.collection_name} 
                    for r in retrieved
                ][self.k_range[0] : self.k_range[1]]
                if self.output_col in dct:
                    dct[self.output_col].extend(output)
                else:
                    dct[self.output_col] = output
                if self.exact_deduplication:
                    dct[self.output_col] = self.dedup_text_content(dct[self.output_col])
                yield dct
