from typing import Any

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from ..pipelines.api import AsyncOpenAIChat


class BaseAgent(AsyncOpenAIChat):
    """
    Base class for all agents.
    - Add a `chat` method to the OpenAI chat completion API.
    - Add a `run_agent` method to define the logic of the agent.
    """

    async def chat(self, messages: list[dict[str, str]]) -> str | BaseModel:
        """
        Asynchronously sends a list of messages to the OpenAI chat completion API and returns the response.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries, where each dictionary contains
                                             keys such as "role" and "content".

        Returns:
            str | BaseModel: The content of the response message as a string, or a parsed BaseModel instance
                             if the "response_format" in `gen_kwargs` is a subclass of BaseModel.

        Raises:
            Any exceptions raised by the OpenAI client during the API call.
        """
        if "response_format" in self.gen_kwargs and issubclass(
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
            output: The output. Can be any type, but recommend a `dictionary` or `string`.
        """
        raise NotImplementedError

    async def run_each(self, dct: dict) -> dict:
        """
        Asynchronously runs the agent, and then update `run_agent` output to `output_col`.
        Args:
            dct (dict): The input dictionary to be processed by the agent.
        Returns:
            dict: The updated dictionary with the agent's response added to the specified output column.
        """

        resp = await self.run_agent(dct)
        dct[self.output_col] = resp
        return dct
