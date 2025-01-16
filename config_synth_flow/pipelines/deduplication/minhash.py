from ...base import BasePipeline, DictsGenerator
from rensa import RMinHash
from typing import Literal
from tokenizers import Tokenizer
import re
from tqdm import tqdm

def rensa_minhash(token_list: list[str], num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(token_list)
    return m


def get_dedup_index(tokens_list: list[list[str]], num_perm=128):
    unique_hashes = set()
    deduplicated_indices = []
    
    for idx, tokens in tqdm(enumerate(tokens_list), total=len(tokens_list), desc="Deduplicating"):
        minhash = rensa_minhash(tokens, num_perm)
        hash_tuple = tuple(minhash.digest())
        
        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)
    
    return deduplicated_indices

class MinHashDeduplication(BasePipeline):
    required_packages = ["rensa", "tokenizers"]
    
    def __post_init__(self, 
                      num_perm: int = 64,
                      text_col: str = "text",
                      dedup_level: Literal["sentence", "token"] = "sentence",
                      tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
                      token_col: str = "_tokenized",
                      ):
        self.num_perm = num_perm
        self.text_col = text_col
        self.token_col = token_col
        self.dedup_level = dedup_level
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    
    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        dcts = list(dcts)
        text_list = [dct[self.text_col] for dct in dcts]
        
        if self.token_col not in dcts[0]:
            if self.dedup_level == "token":
                tokens_list = [ls.tokens for ls in self.tokenizer.encode_batch(text_list)]
            elif self.dedup_level == "sentence":    
                tokens_list = [re.split('\n|。', text) for text in text_list]
        else:
            
            tokens_list = [dct[self.token_col] for dct in dcts]
        ids = get_dedup_index(tokens_list, self.num_perm)
        self.logger.info(f"Original size: {len(dcts)} | Deduplicated size: {len(ids)}")
        
        for idx in ids:
            yield dcts[idx]
        