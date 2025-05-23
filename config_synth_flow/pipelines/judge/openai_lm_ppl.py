from typing import Any

from config_synth_flow.base import JudgePipeline


class OpenaiLmPplPipeline(JudgePipeline):
    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        messages_col: str = "messages",
        output_col: str = "ppl",
    ) -> None:
        self.litellm_kwargs = litellm_kwargs
        self.messages_col = messages_col
        self.output_col = output_col

    async def run_each(self, dct: dict):
        if not dct[self.messages_col]:
            return

        messages = dct[self.messages_col]
        resp = await self.chat(messages=messages, top_logprobs=1, log_probs=True, max_tokens=1)
        logprobs = resp.choices[0].logprobs

        # for logprob in logprobs:
        #     for token, prob in logprob.tokens:
        # avg_lp = sum(lp_list) / len(lp_list)
        # dct[self.output_col] = math.exp(-avg_lp)
