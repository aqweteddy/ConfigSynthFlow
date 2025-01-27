from .base import BaseAgent
from .magpie import MagpieBasedOnTextAgent
from .multiturn_qa import MultiTurnGenerationAgent
from .QA_refinement import EditAgent, QARefinementAgent, ValidAgent
from .self_instruct import DocSelfInstructAgent

__all__ = [
    "BaseAgent",
    "MagpieBasedOnTextAgent",
    "DocSelfInstructAgent",
    "QARefinementAgent",
    "ValidAgent",
    "EditAgent",
    "MultiTurnGenerationAgent",
]
