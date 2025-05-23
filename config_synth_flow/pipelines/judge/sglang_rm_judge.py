from typing import Any

from httpx import AsyncClient
from transformers import AutoTokenizer

from config_synth_flow.base import JudgePipeline


class SglangRmJudge(JudgePipeline):
    required_packages = ["transformers", "httpx"]

    def post_init(
        self,
        base_url: str,
        tokenizer_path: str,
        messages_col: str = "messages",
        output_col: str = "rm_score",
        judge_foreach_round: bool = True,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.strip("/") + "/classify"
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.messages_col = messages_col
        self.output_col = output_col
        self.timeout = timeout
        self.judge_foreach_round = judge_foreach_round

    async def judge(self, dct: dict) -> float | list[float]:
        messages = dct[self.messages_col]
        prompts, scores = [], []
        if self.judge_foreach_round:
            for i in range(0, len(messages), 2):
                mess = messages[: i + 2]
                prompts.append(self.tokenizer.apply_chat_template(mess, tokenize=False))
        else:
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False))
        async with AsyncClient(timeout=self.timeout) as client:
            scores = []
            for prompt in prompts:
                data = {"model": self.tokenizer_path, "text": prompt}
                res = await client.post(self.base_url, json=data)
                scores.append(res.json()["embedding"][0])
        return scores if self.judge_foreach_round else scores[0]

    async def get_score(self, judge_result: Any) -> float | list[float]:
        return judge_result

    async def run_each(self, dct: dict):
        if not dct[self.messages_col]:
            return

        score = await self.judge(dct)
        dct[self.output_col] = score
        return dct
