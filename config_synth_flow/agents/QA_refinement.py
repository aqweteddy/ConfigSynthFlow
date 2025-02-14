# import json
# from typing import Literal

# from jinja2 import Template
# from pydantic import BaseModel

# from .base import BaseAgent
# from .constant import DEFAULT_EDIT_QA_TEMPLATE, DEFAULT_VALID_QA_TEMPLATE


# class QueryResponseItem(BaseModel):
#     query: str
#     response: str


# class ValidResponseItem(BaseModel):
#     reason: str
#     suggestion: str
#     score: float


# def messages_to_text(messages: list[dict[str, str]]) -> str:
#     text = ""
#     for message in messages:
#         text += f"{message['role'].upper()}:\n"
#         text += f"{message['content']}\n---\n"

#     return text


# class EditAgent(BaseAgent):
#     def post_init(
#         self,
#         model: str = "gpt-4o-mini",
#         openai_kwargs: dict = None,
#         gen_kwargs: dict = None,
#         edit_template: str | Literal["edit", "evolve"] = "edit",
#         system_prompt: str = "你是一個擅長改進對話的助理，請根據建議，對對話進行改進，並嚴格遵守輸出格式。",
#     ):
#         super().post_init(
#             model=model,
#             openai_kwargs=openai_kwargs,
#             gen_kwargs=gen_kwargs,
#         )

#         if edit_template in ["edit", "evolve"]:
#             edit_template = DEFAULT_EDIT_QA_TEMPLATE[edit_template]

#         self.edit_template = Template(edit_template)

#         self.system_prompt = system_prompt
#         self.gen_kwargs["response_format"] = QueryResponseItem

#     def run_agent(
#         self, messages: list[dict[str, str]], dct: dict, suggestion: str
#     ) -> QueryResponseItem:
#         dct["mes_str"] = messages_to_text(messages)
#         dct["suggestion"] = suggestion
#         prompt = []
#         if self.system_prompt:
#             prompt.append({"role": "system", "content": self.system_prompt})
#         prompt.append({"role": "user", "content": self.edit_template.render(**dct)})
#         dct.pop("mes_str")
#         dct.pop("suggestion")
#         return self.chat(prompt)


# class ValidAgent(BaseAgent):
#     def post_init(
#         self,
#         model: str = "gpt-4o-mini",
#         openai_kwargs: dict = None,
#         gen_kwargs: dict = None,
#         validation_template: str | Literal["validate", "evolve"] = "validate",
#         system_prompt: str = "你是一個對話品質評估專家，請根據對話內容，對對話進行評估，並嚴格遵守輸出格式。",
#     ):
#         super().post_init(
#             model=model,
#             openai_kwargs=openai_kwargs,
#             gen_kwargs=gen_kwargs,
#         )

#         if validation_template in DEFAULT_VALID_QA_TEMPLATE:
#             validation_template = DEFAULT_VALID_QA_TEMPLATE[validation_template]
#         elif r"{{" not in validation_template:
#             self.logger.warning(
#                 "There is no jinja2 template in the validation_template. Please check if it is correct."
#             )

#         self.validation_template = Template(validation_template)
#         self.system_prompt = system_prompt
#         self.gen_kwargs["response_format"] = ValidResponseItem

#     async def run_agent(
#         self, messages: list[dict[str, str]], dct: dict
#     ) -> ValidResponseItem:
#         prompt = []
#         mes_str = messages_to_text(messages)
#         if self.system_prompt:
#             prompt.append({"role": "system", "content": self.system_prompt})
#         dct["mes_str"] = mes_str
#         prompt.append(
#             {"role": "user", "content": self.validation_template.render(**dct)}
#         )
#         dct.pop("mes_str")
#         return await self.chat(prompt)


# class QARefinementAgent(BaseAgent):
#     """
#     There are two sub-agents in this agent: edit_agent and validation_agent.
#     The edit_agent refines the input and the validation_agent validates the input.
#     `QARefinementAgent` will run above agents in a loop until the `early_stop_criteria_lambda` or `max_round` is met.
#     """

#     def post_init(
#         self,
#         edit_agent: EditAgent,
#         valid_agent: ValidAgent,
#         max_rounds: int = 3,
#         early_stop_criteria_lambda: str = None,
#         messages_col: str = "messages",
#         output_col: str = "messages",
#         history_col: str = None,
#     ):
#         """
#         Initialize the QARefinementAgent. The agent will refine the input until the `early_stop_criteria` or `max_round` is met.

#         Args:
#             edit_agent (BaseAgent): The agent that will refine the input. `edit_agent.run_agent` output should be a dict with the keys "query" and "answer"
#             validation_agent (BaseAgent): The agent that will validate the input. `validantion_agent.run_agent` output should be a dict with the keys "reason" and "score"
#             max_rounds (int, optional): The maximum number of rounds. Defaults to 3.
#             early_stop_criteria_lambda (str, optional): The early stop criteria. Defaults to None.
#             text_col (str, optional): The text column. Defaults to "text".
#             output_col (str, optional): The output column. Defaults to "messages".
#             metadata_col (str, optional): The metadata column. Defaults to "metadata".
#             history_col (str, optional): The history column. Defaults to "history".
#         """
#         self.edit_agent = edit_agent
#         self.validation_agent = valid_agent
#         self.max_rounds = max_rounds
#         self.output_col = output_col
#         self.messages_col = messages_col
#         self.early_stop_criteria_lambda = eval(early_stop_criteria_lambda) or (
#             lambda x: x.score > 3
#         )
#         self.history_col = history_col

#     async def run_agent(self, messages: list[dict[str, str]], dct: dict) -> dict:
#         history = []

#         valid_res: ValidResponseItem = await self.validation_agent.run_agent(
#             messages, dct
#         )
#         history.append({"role": "validation", "content": valid_res.model_dump()})

#         if self.early_stop_criteria_lambda(valid_res):
#             return {"history": history, "messages": messages}

#         for i in range(self.max_rounds):
#             # Refinement
#             res: QueryResponseItem = await self.edit_agent.run_agent(
#                 messages, dct, valid_res.suggestion
#             )
#             history.append({"role": "refinement", "content": res.model_dump()})
#             history.append({"role": "user", "content": res.query})
#             history.append({"role": "assistant", "content": res.response})

#             # Validation
#             messages = [
#                 {"role": "user", "content": res.query},
#                 {"role": "assistant", "content": res.response},
#             ]

#             valid_res: ValidResponseItem = await self.validation_agent.run_agent(
#                 messages, dct
#             )
#             history.append({"role": "validation", "content": valid_res.model_dump()})

#             if self.early_stop_criteria_lambda(valid_res):
#                 break
#         return {"history": history, "messages": messages}

#     async def run_each(self, dct: dict) -> dict:
#         output = await self.run_agent(dct[self.messages_col], dct)
#         dct[self.output_col] = output["messages"]
#         if self.history_col:
#             dct[self.history_col] = json.dumps(output["history"], ensure_ascii=False)
#         return dct
