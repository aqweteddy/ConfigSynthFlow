import copy
import json
from typing import Any

import litellm
from pydantic import BaseModel

from config_synth_flow.base import JudgePipeline, PromptTemplate


class Score(BaseModel):
    score: float


class LlmAsJudge(JudgePipeline):
    GET_SCORE_PROMPT = "Please extract the final score from the following text in JSON format.\n\n## Format\n```json\n{{'score': float}}\n```\n\n## Text\n\n{text}"

    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        judge_template: PromptTemplate,
        system_template: PromptTemplate | None = None,
        parser_litellm_kwargs: dict[str, Any] | None = None,
        messages_col: str = "messages",
        output_col: str = "judge_result",
    ) -> None:
        self.litellm_kwargs = litellm_kwargs
        self.judge_template = judge_template
        self.system_template = system_template
        self.messages_col = messages_col
        self.output_col = output_col
        self.parser_litellm_kwargs = parser_litellm_kwargs or {"model": "gpt-4.1-nano"}

    async def judge(self, dct: dict) -> str:
        judge_mess = []
        if self.system_template:
            judge_mess.append({"role": "system", "content": self.system_template.render(**dct)})
        judge_mess.append({"role": "user", "content": self.judge_template.render(**dct)})
        response = await litellm.acompletion(messages=judge_mess, **self.litellm_kwargs)
        reason = response["choices"][0]["message"]["content"]
        return reason

    async def get_score(self, reason: str) -> float:
        kwargs = copy.deepcopy(self.parser_litellm_kwargs)
        kwargs["response_format"] = Score
        kwargs["messages"] = [
            {"role": "user", "content": self.GET_SCORE_PROMPT.format(text=reason)}
        ]
        resp = await litellm.acompletion(**kwargs)
        score = json.loads(resp.choices[0].message.content)["score"]
        return score

    async def run_each(self, dct: dict):
        result = {}
        result["reason"] = await self.judge(dct)
        result["score"] = await self.get_score(result["reason"])
        dct[self.output_col] = result
        return dct
