import copy
import random
from typing import Any

from config_synth_flow.base import PromptTemplate
from config_synth_flow.base.validator import Validator
from config_synth_flow.pipelines.chat import ChatGenerator


class RejectSampling(ChatGenerator):
    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        search_space_params: dict[str, list[float]],
        num_combinations: int = 10,
        system_template: PromptTemplate | None = None,
        validator_list: list[Validator] | None = None,
        messages_col: str = "messages",
        output_col: str = "output",
        prompt_type_col: str = "_prompt_type",
        max_retries: int = 3,
        force_json_mode: bool = False,
        valid_col: str = "validation_scores",
        ground_as_candidate: bool = False,
        max_retries_if_no_valid_response: int = 3,
    ):
        super().post_init(
            litellm_kwargs=litellm_kwargs,
            output_col=output_col,
            prompt_type_col=prompt_type_col,
            max_retries=max_retries,
            force_json_mode=force_json_mode,
            valid_col=valid_col,
            system_template=system_template,
            validator_list=validator_list,
        )

        self.num_combinations = num_combinations
        self.messages_col = messages_col
        self.search_space_params = search_space_params
        self.ground_as_candidate = ground_as_candidate
        self.max_retries_if_no_valid_response = max_retries_if_no_valid_response

    @property
    def gen_params(self) -> dict[str, float]:
        res = {}
        for k, v in self.search_space_params.items():
            res[k] = random.choice(v)

        return res

    async def reject_sampling(self, messages: list[dict], ground: str | None = None) -> list[str]:
        candidates = []

        for _ in range(self.num_combinations):
            try:
                resp = await self.chat(
                    messages,
                    **self.gen_params,
                )
                for choice in resp.choices:
                    candidates.append(choice.message.content)
            except Exception as e:
                self.logger.warning(f"Error generating candidate: {e}")
                continue

        if ground:
            candidates.append(ground)
        return candidates

    async def validate_samples(
        self,
        dct: dict,
        candidates: list[str],
    ) -> tuple[str, dict[str, float]]:
        dct = copy.deepcopy(dct)
        max_items = None
        max_score = 0

        metric_name = self.validator_list[-1].name
        for candidate in candidates:
            dct[self.output_col].append({"role": "assistant", "content": candidate})
            try:
                res, scores = await self.get_validator_result(dct)
            except Exception as e:
                self.logger.warning(f"Error validating candidate: {e}")
                continue

            if res and scores[metric_name]["score"] > max_score:
                max_score = scores[metric_name]["score"]
                max_items = (candidate, scores)
            dct[self.output_col].pop(-1)
        if max_items:
            return max_items
        else:
            return (None, None)

    async def multiturn_rejection_sampling(
        self,
        dct: dict,
        ground_as_candidate: bool = True,
        system_prompt: str | None = None,
    ) -> tuple[list[dict], list[dict[str, float]]]:
        messages = dct[self.messages_col]
        user_mess = [m["content"] for m in messages if m["role"] == "user"]
        assistant_mess = [m["content"] for m in messages if m["role"] == "assistant"]
        assert len(user_mess) == len(assistant_mess)
        result, scores = [], []
        if system_prompt and messages[0]["role"] != "system":
            result.append({"role": "system", "content": system_prompt})

        for u, ori_ass in zip(user_mess, assistant_mess):
            for _ in range(self.max_retries_if_no_valid_response):
                result.append({"role": "user", "content": u})

                candidates = await self.reject_sampling(
                    result,
                    ground=ori_ass if ground_as_candidate else None,
                )

                dct[self.output_col] = result
                res, score = await self.validate_samples(dct, candidates)
                if res:
                    result.append({"role": "assistant", "content": res})
                    scores.append(score)
                    break
                else:
                    self.logger.warning(
                        f"No valid assistant response found for user message: {u}, retry {_ + 1} times."
                    )
                    result = result[:-1]
            else:
                self.logger.warning(
                    f"No valid assistant response found for user message: {u}, skip this turn."
                )
        return result, scores

    async def run_each(self, dct: dict) -> dict:
        system_prompt = self.system_template.render(**dct) if self.system_template else None
        result, scores = await self.multiturn_rejection_sampling(
            copy.deepcopy(dct),
            self.ground_as_candidate,
            system_prompt=system_prompt,
        )
        dct[self.output_col] = result
        dct[self.valid_col] = scores
        return dct
