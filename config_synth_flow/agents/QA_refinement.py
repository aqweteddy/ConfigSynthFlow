from .base import BaseAgent
from pydantic import BaseModel
from jinja2 import Template
from .constant import DEFAULT_EDIT_QA_TEMPLATE, DEFAULT_VALID_QA_TEMPLATE


class QueryResponseItem(BaseModel):
    query: str
    response: str


class ValidResponseItem(BaseModel):
    reason: str
    suggestion: str
    score: float


def messages_to_text(messages: list[dict[str, str]]) -> str:
    text = ""
    for message in messages:
        text += f"{message['role'].upper()}:\n"
        text += f"{message['content']}\n---\n"

    return text


class EditAgent(BaseAgent):
    def __post_init__(
        self,
        model: str = "gpt-4o-mini",
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        edit_template: str = DEFAULT_EDIT_QA_TEMPLATE,
        system_prompt: str = "你是一個擅長改進對話的助理，請根據建議，對對話進行改進，並嚴格遵守輸出格式。",
    ):
        super().__post_init__(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
        )
        self.edit_template = Template(edit_template)
        self.system_prompt = system_prompt
        self.gen_kwargs["response_format"] = QueryResponseItem

    def run_agent(
        self, messages: list[dict[str, str]], dct: dict, suggestion: str
    ) -> QueryResponseItem:
        dct["mes_str"] = messages_to_text(messages)
        dct["suggestion"] = suggestion
        prompt = []
        if self.system_prompt:
            prompt.append({"role": "system", "content": self.system_prompt})
        prompt.append({"role": "user", "content": self.edit_template.render(**dct)})
        dct.pop("mes_str")
        dct.pop("suggestion")
        return self.chat(prompt, "user")


class ValidAgent(BaseAgent):
    def __post_init__(
        self,
        model: str = "gpt-4o-mini",
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        validation_template: str = DEFAULT_VALID_QA_TEMPLATE,
        system_prompt: str = "你是一個對話品質評估專家，請根據對話內容，對對話進行評估，並嚴格遵守輸出格式。",
    ):
        super().__post_init__(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
        )
        self.validation_template = Template(validation_template)
        self.system_prompt = system_prompt
        self.gen_kwargs["response_format"] = ValidResponseItem

    def run_agent(self, messages: list[dict[str, str]], dct: dict) -> ValidResponseItem:
        prompt = []
        mes_str = messages_to_text(messages)
        if self.system_prompt:
            prompt.append({"role": "system", "content": self.system_prompt})
        dct["mes_str"] = mes_str
        prompt.append(
            {"role": "user", "content": self.validation_template.render(**dct)}
        )
        dct.pop("mes_str")
        return self.chat(prompt)


class QARefinementAgent(BaseAgent):
    def __post_init__(
        self,
        edit_agent: EditAgent,
        valid_agent: ValidAgent,
        max_rounds: int = 3,
        early_stop_criteria_lambda: str = None,
        messages_col: str = "messages",
        output_col: str = "messages",
        history_col: str = None,
        # mode: Literal["refine", "multiturn"] = "refine",  # multiturn or refine
    ):
        """
        QA Refinement Agent

        Args:
            edit_agent (BaseAgent): The agent that will refine the input. `edit_agent.run_agent` output should be a dict with the keys "query" and "answer"
            validation_agent (BaseAgent): The agent that will validate the input. `validantion_agent.run_agent` output should be a dict with the keys "reason" and "score"
            max_rounds (int, optional): The maximum number of rounds. Defaults to 3.
            early_stop_criteria_lambda (str, optional): The early stop criteria. Defaults to None.
            text_col (str, optional): The text column. Defaults to "text".
            output_col (str, optional): The output column. Defaults to "messages".
            metadata_col (str, optional): The metadata column. Defaults to "metadata".
            history_col (str, optional): The history column. Defaults to "history".
            mode (Literal["refine", "multiturn"], optional): The mode of the agent. Defaults to "refine".
        """
        self.edit_agent = edit_agent
        self.validation_agent = valid_agent
        self.max_rounds = max_rounds
        self.output_col = output_col
        self.messages_col = messages_col
        self.early_stop_criteria_lambda = eval(early_stop_criteria_lambda) or (
            lambda x: x.score > 3
        )
        self.history_col = history_col

    async def run_agent(self, messages: list[dict[str, str]], dct: dict) -> dict:
        history, result = [], []

        valid_res: ValidResponseItem = await self.validation_agent.run_agent(messages, dct)
        history.append({"role": "validation", "content": valid_res.model_dump()})
        if self.early_stop_criteria_lambda(valid_res):
            return {"history": history, "messages": messages}

        for i in range(self.max_rounds):
            # Refinement
            res: QueryResponseItem = await self.edit_agent.run_agent(messages, dct)
            history.append({"role": "refinement", "content": res.model_dump()})
            history.append({"role": "user", "content": res.query})
            history.append({"role": "assistant", "content": res.response})

            # Validation
            messages = [
                {"role": "user", "content": res.query},
                {"role": "assistant", "content": res.response},
            ]
            
            valid_res: ValidResponseItem = await self.validation_agent.run_agent(messages, dct)
            history.append({"role": "validation", "content": valid_res.model_dump()})


            if self.early_stop_criteria_lambda(valid_res):
                break

        return {"history": history, "messages": result}

    async def run_each(self, dct: dict) -> dict:
        output = await self.run_agent(dct[self.messages_col], dct)
        dct[self.output_col] = output["messages"]
        if self.history_col:
            dct[self.history_col] = output["history"]
        return dct
