from qdrant_client import QdrantClient, models

from config_synth_flow.base import AsyncBasePipeline, DictsGenerator

from ..embedding_reranker import InfinityApiEmbedder


class QdrantApiRetriever(AsyncBasePipeline):
    required_packages: list[str] = ["qdrant_client"]

    def post_init(
        self,
        embedder: InfinityApiEmbedder,
        api_base: str,
        api_port: int = 6333,
        collection_name: str = "default",
        qdrant_text_key: str = "text",
        topk_range: tuple[int, int] = (0, 5),
        similarity_threshold: float = 0.5,
        output_col: str = "qdrant_retrieved",
        exact_dedup: bool = True,
        timeout: int = 120,
    ) -> None:
        self.embedder = embedder
        self.host = api_base.strip("/")
        self.api_base = api_base
        self.api_port = api_port
        self.topk_range = topk_range
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name
        self.qdrant_cli = QdrantClient(url=self.host, port=api_port)
        self.timeout = timeout
        self.output_col = output_col
        self.qdrant_text_key = qdrant_text_key
        self.exact_dedup = exact_dedup

        self._exist_collection = False

    def batch_insert(
        self,
        dcts: DictsGenerator,
        collection_name: str = None,
    ):
        """Batch insert documents into QdrantDB.

        Args:
            dcts (DictsGenerator): Generator of dictionaries to insert.
            collection_name (str, optional): QdrantDB collection name. Defaults to None.
        """
        payloads = [
            {
                "id": d.get("id", None),
                self.qdrant_text_key: d[self.qdrant_text_key],
                "metadata": d.get("metadata", {}),
            }
            for d in dcts
        ]

        embeds = self.embedder.get_embeddings([self.embedder.query_lambda_col(d) for d in dcts])
        if not self._exist_collection:
            try:
                self.logger.info(f"Creating collection {collection_name}")
                self.qdrant_cli.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=len(embeds[0]), distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0, default_segment_number=2
                    ),
                    # quantization_config=models.ScalarQuantization(
                    #     scalar=models.ScalarQuantizationConfig(
                    #         type=models.ScalarType.INT8,
                    #         always_ram=True,
                    #     ),
                    # ),
                    shard_number=4,
                )
            except Exception as _:
                pass
            self._exist_collection = True

        self.qdrant_cli.upload_collection(
            collection_name=collection_name,
            vectors=embeds,
            payload=payloads,
        )

    async def run_each(self, dct: dict) -> dict:
        embed_dct = await self.embedder.run_each(dct)
        embed = embed_dct[self.embedder.output_col]
        retrieved = self.qdrant_cli.search(
            collection_name=self.collection_name,
            query_vector=embed,
            limit=self.topk_range[1],
            score_threshold=self.similarity_threshold,
            timeout=self.timeout,
        )

        output = [
            {
                "id": r.payload.pop("id", None) or r.id,
                "text": r.payload.pop(self.qdrant_text_key),
                "score": r.score,
                "metadata": r.payload.get("metadata", None),
            }
            for r in retrieved
        ][self.topk_range[0] : self.topk_range[1]]

        if self.exact_dedup:
            new_output = []
            used = set()
            for r in output:
                if r["id"] not in used:
                    new_output.append(r)
                    used.add(r["text"])
            output = new_output

        if self.output_col in dct and isinstance(dct[self.output_col], list):
            dct[self.output_col].extend(output)
        else:
            dct[self.output_col] = output
        return dct
