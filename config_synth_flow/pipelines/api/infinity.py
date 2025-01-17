from ...base import BasePipeline, DictsGenerator
import requests


class InfinityApiReranker(BasePipeline):
    def __init__(
        self,
        api_base: str,
        model_name: str,
        k_range: tuple[int, int] = (1, 5),
        similarity_threshold: float = 0.5,
        query_lambda_col: str = 'lambda x: x["query"]', # lambda function to extract query str from the input dict
        document_lambda_col: str = 'lambda x: x["documents"]', # lambda function to extract documents list[str] from the input dict
        output_col: str = "reranked",
    ) -> None:
        """Pipeline to rerank documents using Infinity API.
        
        Args:
            api_base (str): Infinity API base URL
            model_name (str): Model name
            k_range (tuple[int, int], optional): Range of reranked documents. Defaults to (1, 5).
            similarity_threshold (float, optional): Minimum similarity score to rerank. Defaults to 0.5.
            query_lambda_col (str, optional): Lambda function to extract query str from the input dict. Defaults to 'lambda x: x["query"]'.
            document_lambda_col (str, optional): Lambda function to extract documents (list[str]) from the input dict. Defaults to 'lambda x: x["documents"]'.
            output_col (str, optional): Output column name. Defaults to "reranked".
        """
        
        self.host = api_base.strip('/')
        self.model_name = model_name
        self.query_lambda_col = eval(query_lambda_col)
        self.document_lambda_col = eval(document_lambda_col)
        self.output_col = output_col
        self.k_range = k_range
        self.similarity_threshold = similarity_threshold
        
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        
        if not self.host.endswith('/rerank'):
            self.host += '/rerank'

    def _post(self, query: str, docs: list[str]) -> list[float]:
        docs = [d[:1500] for d in docs]
        params = {
            "query": query,
            "documents": docs,
            "top_n": 10000,
            "model": self.model_name,
            "raw_scores": False,
            "return_documents": False,
        }

        url = self.host.strip("/")

        if not url.endswith("/rerank"):
            url += "/rerank"

        res = requests.post(
            url,
            json=params,
            timeout=20,
        ).json()
        
        result = sorted(res['result'], key=lambda x: x['index'])
        
        return [r["relevance_score"] for r in result]

    def run_each(self, dct: dict) -> dict:
        query = self.query_lambda_col(dct)
        docs = self.document_lambda_col(dct)
        scores = self._post(query, docs)
        result = [{"text": doc, "score": score} for doc, score in zip(docs, scores)]
        result = filter(lambda x: x["score"] > self.similarity_threshold, result)
        result = sorted(result, key=lambda x: x["score"], reverse=True)[self.k_range[0]:self.k_range[1]]
        dct[self.output_col] = result
        
        return dct


class InfinityApiEmbedder(BasePipeline):
    def __post_init__(self, 
        api_base: str,
        model_name: str,
        batch_size: int = 32,
        query_lambda_col: str = 'lambda x: x["query"]',
        output_col: str = "_embeddings",
    ):
        """Pipeline to get embeddings from Infinity API.
        
        Args:
            api_base (str): Infinity API base URL
            model_name (str): Model name
            batch_size (int, optional): Batch size. Defaults to 32.
            query_lambda_col (str, optional): Lambda function to extract query str from the input dict. Defaults to 'lambda x: x["query"]'.
            output_col (str, optional): Output column name. Defaults to "_embeddings".
        """
        self.host = api_base.strip('/')
        self.model_name = model_name
        self.query_lambda_col = eval(query_lambda_col)
        self.batch_size = batch_size
        self.output_col = output_col
        
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        
        if not self.host.endswith('/embeddings'):
            self.host += '/embeddings'
    
    def _post(self, query: list[str]) -> list[float]:
        params = {
            "input": query,
            "model": self.model_name,
        }

        res = requests.post(
            self.host,
            json=params,
            timeout=20,
        ).json()['data']
        res = sorted(res, key=lambda x: x['index'])
        res = [r['embedding'] for r in res]
        
        return res
    
    def get_embeddings(self, query_list: list[str]) -> list[list[float]]:
        return self._post(query_list)
    
    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        batch_dcts = []
        
        for dct in dcts:
            batch_dcts.append(dct)
            if len(batch_dcts) == self.batch_size:
                queries = [self.query_lambda_col(d) for d in batch_dcts]
                embeddings = self.get_embeddings(queries)
                for dct, emb in zip(batch_dcts, embeddings):
                    dct[self.output_col] = emb
                    yield dct
                batch_dcts = []
        
        embeddings = self.get_embeddings([self.query_lambda_col(d) for d in batch_dcts])
        for dct, emb in zip(batch_dcts, embeddings):
            dct[self.output_col] = emb
            yield dct