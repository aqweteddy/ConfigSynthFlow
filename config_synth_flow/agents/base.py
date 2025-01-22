from ..pipelines.api import AsyncOpenAIChat
from openai.types.chat import ChatCompletion
from typing import Any
from pydantic import BaseModel


class BaseAgent(AsyncOpenAIChat):
    async def chat(self, messages: list[dict[str, str]]) -> str | BaseModel:
        if 'response_format' in self.gen_kwargs and issubclass(
            self.gen_kwargs["response_format"], BaseModel
        ):
            res: ChatCompletion = await self.openai_client.beta.chat.completions.parse(
                messages=messages, **self.gen_kwargs
            )
            return res.choices[0].message.parsed
        else:
            res: ChatCompletion = await self.openai_client.chat.completions.create(
                messages=messages, **self.gen_kwargs
            )
            return res.choices[0].message.content

    async def run_agent(self, dct: dict) -> dict | str | Any:
        """Define the logic of the agent here

        Args:
            dct (dict): The input dictionary
        Returns:
            output: The output. Can be any type
        """
        raise NotImplementedError

    async def run_each(self, dct: dict) -> dict:
        resp = await self.run_agent(dct)
        dct[self.output_col] = resp
        return dct
