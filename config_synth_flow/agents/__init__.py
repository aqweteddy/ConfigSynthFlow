from .base import BaseAgent
from .benchmark import EditAgent, QARefinementAgent, ValidAgent
from .self_instruct import DocSelfInstructAgent
# from .magpie.text_magpie import MagpieBasedOnTextAgent
# from .multiturn_qa import MultiTurnGenerationAgent

# # from .QA_refinement import EditAgent, QARefinementAgent, ValidAgent
# from .self_instruct import DocSelfInstructAgent

__all__ = [
    "BaseAgent",
    "EditAgent",
    # "MagpieBasedOnTextAgent",
    "DocSelfInstructAgent",
    # "QARefinementAgent",
    # "ValidAgent",
    # "EditAgent",
    # "MultiTurnGenerationAgent",
]
