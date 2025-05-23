import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

from rensa import RMinHash
from tokenizers import Tokenizer
from tqdm import tqdm

from config_synth_flow.base import BasePipeline, DictsGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LSH:
    def __init__(self, num_perm: int, threshold: float, num_bands: int):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        self.band_size = num_perm // num_bands
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]

    def _hash_band(self, band: list[int]) -> int:
        return hash(tuple(band))

    def insert(self, key: int, digest) -> None:
        for i in range(self.num_bands):
            start = i * self.band_size
            end = start + self.band_size
            band_hash = self._hash_band(digest[start:end])
            self.hash_tables[i][band_hash].append(key)

    def query(self, digest) -> list[int]:
        candidates = set()
        for i in range(self.num_bands):
            start = i * self.band_size
            end = start + self.band_size
            band_hash = self._hash_band(digest[start:end])
            candidates.update(self.hash_tables[i].get(band_hash, []))
        return list(candidates)

    def is_similar(self, minhash1: "RMinHash", minhash2: "RMinHash") -> bool:
        return minhash1.jaccard(minhash2) >= self.threshold


def rensa_minhash(token_list: list[str], num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(token_list)
    return m.digest()


def get_dedup_index(
    tokens_list: list[list[str]], num_perm=128, theshold=0.8, num_bands=16, num_proc=1
) -> set[int]:
    lsh = LSH(num_perm, theshold, num_bands)
    unique_text_indices = set()

    with ProcessPoolExecutor(num_proc) as executor:
        for idx, digest in enumerate(
            tqdm(
                executor.map(rensa_minhash, tokens_list),
                total=len(tokens_list),
                desc="MinHashing",
            )
        ):
            similar = lsh.query(digest)
            if not similar:
                unique_text_indices.add(idx)
                lsh.insert(idx, digest)

    return unique_text_indices


class MinHashDeduplicator(BasePipeline):
    required_packages = ["rensa", "tokenizers"]

    def post_init(
        self,
        num_perm: int = 64,
        text_col: str = "text",
        threshold: float = 0.8,
        dedup_level: Literal["sentence", "token"] = "sentence",
        sentence_split_regex: str = "\n|。|、|，|；|：|？",
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        token_col: str = "_tokenized",
        num_proc: int = 1,
    ):
        """
        Deduplicate the dataset using MinHash. There are two deduplication levels: sentence and token.
        - sentence: Split the text into sentences with `sentence_split_regex` and deduplicate at the sentence level.
        - token: Tokenize the text with `tokenizer_name` and deduplicate at the token level.

        Args:
            num_perm (int, optional): Number of permutation for MinHash. Defaults to 64.
            text_col (str, optional): Text column name. Defaults to "text".
            dedup_level (Literal["sentence", "token"], optional): Deduplication level. Defaults to "sentence".
            sentence_split_regex (str, optional): Regex to split the text into sentences. Defaults to "\n|。".
            tokenizer_name (str, optional): Tokenizer name. Defaults to "Qwen/Qwen2.5-0.5B".
            token_col (str, optional): Tokenized column name. Defaults to "_tokenized".
            num_proc (int, optional): Number of processes to use. Defaults to 1.
        """
        self.num_perm = num_perm
        self.text_col = text_col
        self.token_col = token_col
        self.dedup_level = dedup_level
        self.threshold = threshold
        self.sent_split_regex = re.compile(sentence_split_regex, flags=re.UNICODE | re.MULTILINE)
        try:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            self.logger.info(
                f"Failed to load tokenizer from file: {e}, trying to load from hugginface."
            )
            self.tokenizer = Tokenizer.from_file(tokenizer_name)
        self.num_proc = num_proc

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    def sentence_split(self, text: str) -> list[str]:
        sentences = self.sent_split_regex.split(text)
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                result.append(sentence)
        return result

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        dcts = list(tqdm(dcts, desc="Initiating Deduplication"))
        text_list = [dct[self.text_col] for dct in dcts]

        if self.token_col not in dcts[0]:
            if self.dedup_level == "token":
                with ProcessPoolExecutor(self.num_proc) as executor:
                    tokens_list = list(
                        tqdm(
                            executor.map(self.tokenize, text_list, chunksize=100),
                            total=len(text_list),
                            desc="Tokenizing",
                        )
                    )
            elif self.dedup_level == "sentence":
                self.logger.info(
                    f"Splitting text into sentences using regex: {self.sent_split_regex}"
                )
                tokens_list = [
                    self.sentence_split(text) for text in tqdm(text_list, desc="Splitting")
                ]
        else:
            tokens_list = [dct[self.token_col] for dct in dcts]
        ids = get_dedup_index(
            tokens_list, self.num_perm, theshold=self.threshold, num_proc=self.num_proc
        )
        self.logger.info(f"Original size: {len(dcts)} | Deduplicated size: {len(ids)}")

        for idx in ids:
            yield dcts[idx]
