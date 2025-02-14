import asyncio

from aiohttp import ClientSession

from ...base import AsyncBasePipeline


class InfinityApiReranker(AsyncBasePipeline):
    def post_init(
        self,
        api_base: str,
        model_name: str,
        k_range: tuple[int, int] = (1, 5),
        similarity_threshold: float = 0.5,
        query_lambda_col: str = 'lambda x: x["query"]',  # lambda function to extract query str from the input dict
        document_lambda_col: str = 'lambda x: x["documents"]',  # lambda function to extract documents list[str] from the input dict
        metadata_lambda_col: str = None,  # lambda function to extract metadata dict from the input dict
        output_col: str = "reranked",
        timeout: int = 120,
        num_concurrent: int = 4,
    ) -> None:
        """Pipeline to rerank documents using Infinity API.

        Args:
            api_base (str): Infinity API base URL.
            model_name (str): Model name.
            k_range (tuple[int, int], optional): Range of reranked documents. Defaults to (1, 5).
            similarity_threshold (float, optional): Minimum similarity score to rerank. Defaults to 0.5.
            query_lambda_col (str, optional): Lambda function to extract query str from the input dict. Defaults to 'lambda x: x["query"]'.
            document_lambda_col (str, optional): Lambda function to extract documents (list[str]) from the input dict. Defaults to 'lambda x: x["documents"]'.
            metadata_lambda_col (str, optional): Lambda function to extract metadata dict from the input dict. Defaults to None.
            output_col (str, optional): Output column name. Defaults to "reranked".
            timeout (int, optional): API request timeout. Defaults to 120.
            num_concurrent (int, optional): Number of concurrent requests. Defaults to 4.
        """

        self.host = api_base.strip("/")
        self.model_name = model_name
        self.query_lambda_col = eval(query_lambda_col)
        self.document_lambda_col = eval(document_lambda_col)
        self.metadata_lambda_col = (
            eval(metadata_lambda_col) if metadata_lambda_col else None
        )
        self.output_col = output_col
        self.k_range = k_range
        self.similarity_threshold = similarity_threshold
        self.timeout = timeout
        self.num_concurrent = num_concurrent

        if not self.host.startswith("http"):
            self.host = "http://" + self.host

        if not self.host.endswith("/rerank"):
            self.host += "/rerank"

    async def apost(self, query: str, docs: list[str]) -> list[float]:
        """Send a POST request to the Infinity API.

        Args:
            query (str): Query string.
            docs (list[str]): List of documents.

        Returns:
            list[float]: List of relevance scores.
        """
        if len(docs) == 0:
            return []
        docs = [d[:1500] for d in docs]
        params = {
            "query": query,
            "documents": docs,
            "top_n": 10000,
            "model": self.model_name,
            "raw_scores": False,
            "return_documents": False,
        }

        async with ClientSession() as session:
            async with session.post(
                self.host, json=params, timeout=self.timeout
            ) as response:
                res = await response.json()
                result = sorted(res["results"], key=lambda x: x["index"])
        return [r["relevance_score"] for r in result]

    async def run_each(self, dct: dict) -> dict:
        """Rerank documents for each dictionary.

        Args:
            dct (dict): Input dictionary.

        Returns:
            dict: Dictionary with reranked documents.
        """
        query = self.query_lambda_col(dct)
        docs = self.document_lambda_col(dct)

        scores = await self.apost(query, docs)

        if self.metadata_lambda_col:
            metadata_list = self.metadata_lambda_col(dct)
            result = [
                {"text": doc, "score": score, "metadata": metadata}
                for doc, score, metadata in zip(docs, scores, metadata_list)
            ]
        else:
            result = [{"text": doc, "score": score} for doc, score in zip(docs, scores)]
        result = filter(lambda x: x["score"] > self.similarity_threshold, result)
        result = sorted(result, key=lambda x: x["score"], reverse=True)[
            self.k_range[0] : self.k_range[1]
        ]
        dct[self.output_col] = result
        return dct


class InfinityApiEmbedder(AsyncBasePipeline):
    def post_init(
        self,
        api_base: str,
        model_name: str,
        batch_size: int = 32,
        query_lambda_col: str = 'lambda x: x["query"]',
        output_col: str = "_embeddings",
    ):
        """Pipeline to get embeddings from Infinity API.

        Args:
            api_base (str): Infinity API base URL.
            model_name (str): Model name.
            batch_size (int, optional): Batch size. Defaults to 32.
            query_lambda_col (str, optional): Lambda function to extract query str from the input dict. Defaults to 'lambda x: x["query"]'.
            output_col (str, optional): Output column name. Defaults to "_embeddings".
        """

        self.host = api_base.strip("/")
        self.model_name = model_name
        self.query_lambda_col = eval(query_lambda_col)
        self.batch_size = batch_size
        self.output_col = output_col

        if not self.host.startswith("http"):
            self.host = "http://" + self.host

        if not self.host.endswith("/embeddings"):
            self.host += "/embeddings"

    async def apost(self, query: list[str]) -> list[float]:
        """Send a POST request to the Infinity API.

        Args:
            query (list[str]): List of query strings.

        Returns:
            list[float]: List of embeddings.
        """
        if len(query) == 0:
            return []
        params = {
            "input": query,
            "model": self.model_name,
        }

        async with ClientSession() as session:
            async with session.post(self.host, json=params) as response:
                res = await response.json()
                res = res["data"]
                res = sorted(res, key=lambda x: x["index"])

        return [r["embedding"] for r in res]

    async def run_each(self, dct: dict) -> dict:
        """Get embeddings for each dictionary.

        Args:
            dct (dict): Input dictionary.

        Returns:
            dict: Dictionary with embeddings.
        """
        query = self.query_lambda_col(dct)
        embeddings = await self.apost(query)
        dct[self.output_col] = embeddings
        return dct

    def get_embeddings(self, query: list[str]) -> list[float]:
        """Get embeddings for a list of queries.

        Args:
            query (list[str]): List of query strings.

        Returns:
            list[float]: List of embeddings.
        """
        return asyncio.run(self.apost(query))
