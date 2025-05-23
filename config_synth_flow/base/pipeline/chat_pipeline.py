from typing import Any

from litellm import ModelResponse, TextCompletionResponse, acompletion, atext_completion

from .async_pipeline import AsyncBasePipeline
from .base_pipeline import BasePipeline


class AsyncChatBasePipeline(AsyncBasePipeline):
    def post_init(self, litellm_kwargs: dict[str, Any]) -> None:
        self.litellm_kwargs = litellm_kwargs
        self._chat_cnt = 0

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> ModelResponse:
        self._chat_cnt += 1
        if self._chat_cnt % 100 == 0:
            self.logger.info(f"Chat count: {self._chat_cnt}")

        # merge kwargs with litellm_kwargs
        kwargs = {**self.litellm_kwargs, **kwargs}
        return await acompletion(
            messages=messages,
            **kwargs,
        )

    async def completion(self, prompt: str, **kwargs) -> TextCompletionResponse:
        for k, v in self.litellm_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v

        return await atext_completion(
            prompt=prompt,
            **kwargs,
        )


class BatchChatPipeline(BasePipeline):
    # TODO: Implement batch chat pipeline
    pass
