"""
Common types used across mixins.
"""

from collections.abc import Generator
from typing import Any, Union

DictsGenerator = Union[list[dict[str, Any]], Generator[dict[str, Any], None, None]]
