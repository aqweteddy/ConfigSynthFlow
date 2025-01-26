from .base import BaseAgent
import copy
from .QA_refinement import QARefinementAgent

class MultiTurnGenerationAgent(BaseAgent):
    def __post_init__(
        self,
        qa_refinement_agent: QARefinementAgent,
        messages_col: str = "messages",
        output_col: str = "messages",
        num_extended_turns: int = 3
    ):
        self.qa_refinement_agent = qa_refinement_agent
        self.num_extended_turns = num_extended_turns
        self.messages_col = messages_col
        self.output_col = output_col
    
    async def run_agent(self, dct: dict) -> list[dict]:
        messages: list[dict] =copy.deepcopy(dct[self.messages_col])
        for _ in range(self.num_extended_turns):
            output = await self.qa_refinement_agent.run_agent(messages, dct)
            messages.extend(output["messages"])
        return messages