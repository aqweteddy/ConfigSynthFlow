import asyncio
import re
from typing import Literal
from openai import AsyncOpenAI
import numpy as np
from httpx import AsyncClient, Limits
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opencc import OpenCC
from tokenizers import Tokenizer

from config_synth_flow.base import AsyncBasePipeline


class InfinitySlidingWindowEduClassifier(AsyncBasePipeline):
    required_packages = ["httpx", "langchain_text_splitters", "opencc", "tokenizers"]

    def post_init(
        self,
        base_url: str,
        model_path: str,
        text_col: str = "text",
        output_col: str = "text",
        score_col: str = "score",
        threshold: float = 3.5,
        to_cn: bool = False,
        chunk_size: int = 500,
        sliding_window_size: int = 100,
    ) -> None:
        self.base_url = base_url.strip("/")
        self.model_path = model_path
        self.tokenizer = Tokenizer.from_pretrained(model_path)
        self.text_col = text_col
        self.output_col = output_col
        self.score_col = score_col
        self.threshold = threshold
        self.to_cn = to_cn
        self.chunk_size = chunk_size
        self.sliding_window_size = sliding_window_size
        self.spliter = RecursiveCharacterTextSplitter(
            ["\n\n", "。", "\n", "？", "！", "……", "…", "...", "."],
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=self.get_length,
            keep_separator=True,
        )
        if to_cn:
            self.converter = OpenCC("t2s")

    def get_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    async def request_infinity(self, req: dict) -> list[float]:
        score = -1
        async with AsyncClient(verify=False) as client:
            for _ in range(3):
                try:
                    response = await client.post(f"{self.host}/rerank", json=req)
                    resp_list = response.json()["results"]
                    resp_list = sorted(resp_list, key=lambda x: x["index"])
                    score = resp_list[0]["relevance_score"]
                    break
                except Exception as e:
                    await asyncio.sleep(1)
        return score

    async def get_score_list(self, chunks: list[str]) -> list[float]:
        scores, req_text = [], []
        for chunk in chunks:
            if self.to_cn:
                chunk = self.converter.convert(chunk)
            req_text.append(chunk)
            req_text = req_text[-self.sliding_window_size :]

            if len(req_text) < self.sliding_window_size:
                continue

            req = {
                "model": self.model_path,
                "query": "\n".join(req_text),
                "documents": [""],
                "raw_scores": True,
            }
            score = await self.request_infinity(req)
            scores.append(score)
        return scores

    async def run_each(self, dct: dict):
        text = dct[self.text_col][:50000]
        chunks = self.spliter.split_text(text)
        scores: list[float] = await self.get_score_list(chunks)

        drop_cnt = 0
        for chunk, score in zip(chunks[::-1], scores[::-1]):
            if score < self.threshold or chunk.strip().endswith("..."):
                text = text.replace(chunk, "")
                drop_cnt += 1
            elif score == -1:
                continue
            else:
                break
        scores = scores[:-drop_cnt]
        scores = [score for score in scores if score > 0]
        if len(scores) == 0:
            return None
        dct[self.score_col] = sum(scores) / len(scores)
        dct[self.output_col] = text.strip()
        return dct


class EduClassifierTailRemover(AsyncBasePipeline):
    def post_init(
        self,
        base_url: str,
        text_col: str = "text",
        model_path: str = "ibm-granite/granite-embedding-107m-multilingual",
        threshold: float = 0.6,
        output_col: str = "text",
        chunk_size: int = 500,
        min_tail_line_length: int = 10,
    ):
        self.host = base_url.strip("/")
        self.text_col = text_col
        self.model_path = model_path
        self.threshold = threshold
        self.tokenizer = Tokenizer.from_pretrained(model_path)
        self.min_tail_line_length = min_tail_line_length
        self.ouput_col = output_col
        self.splitter = RecursiveCharacterTextSplitter(
            ["\n\n", "\n", "。", "？", "！", "，", "……", "…", "...", "."],
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=self.get_length,
            keep_separator="end",
        )

    def get_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def remove_short_tail(self, text: str) -> str:
        rev_text_lines = text.split("\n")[::-1]
        rev_text_lines = [r for r in rev_text_lines if len(r.strip()) > 0]
        for r in rev_text_lines:
            if self.get_length(r) > self.min_tail_line_length and not r.endswith("..."):
                break
            text = text.replace(r, "")
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    async def get_embedding(self, text: str) -> list[float]:
        req = {
            "model": self.model_path,
            "input": text,
        }
        async with AsyncOpenAI(
            base_url=f"{self.host}/embeddings",
        ) as client:
            response = await client.post(f"{self.host}/embeddings", json=req)
        return np.array(response.json()["data"][0]["embedding"])

    @staticmethod
    def cosine_similarity(a: np.array, b: np.array) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def remove_not_important_tail(self, text: str) -> str:
        text_chunk = self.splitter.split_text(text)
        if len(text_chunk) == 0:
            return text
        embed_text = text_chunk[0]
        if len(text_chunk) > 1:
            embed_text += text_chunk[1]
        text_emb = await self.get_embedding(embed_text)
        # text_emb = await self.get_embedding(text[:500])
        text_chunk = text_chunk[1:]
        text_chunk = [t for t in text.split("\n") if t.strip()][-7:]

        for chunk in text_chunk[::-1]:
            chunk_emb = await self.get_embedding(chunk)
            if self.cosine_similarity(text_emb, chunk_emb) <= self.threshold:
                text = text.replace(chunk, "")
            else:
                break
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    async def run_each(self, dct: dict):
        text = dct[self.text_col]

        text = self.remove_short_tail(text)
        if text.count("...") > 4:
            return None
        text = await self.remove_not_important_tail(text)

        if len(text) == 0:
            return None
        text = self.remove_short_tail(text)

        dct[self.ouput_col] = text.strip()
        return dct
