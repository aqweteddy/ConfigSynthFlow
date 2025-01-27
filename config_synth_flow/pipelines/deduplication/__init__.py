"""
This package provides various deduplication pipelines.
"""

from .minhash import MinHashDeduplication
from .set import SetExactMatch

__all__ = [MinHashDeduplication, SetExactMatch]
