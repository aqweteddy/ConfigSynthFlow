from copy import deepcopy
from typing import Any

from transformers import AutoTokenizer

from config_synth_flow.base import PromptTemplate
from config_synth_flow.pipelines.chat import ChatGenerator
from config_synth_flow.base.validator import Validator


class Magpie(ChatGenerator):
    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        tokenizer_path: str,
        system_template: PromptTemplate | list[PromptTemplate] | str,
        max_turns: int = 2,
        validator_list: list[dict] | None = None,
        user_prefix: str = "<|im_start|>user\n",
        assistant_prefix: str = "<|im_start|>assistant\n",
        user_max_tokens: int = 100,
        assistant_max_tokens: int = 3000,
        output_col: str = "output",
        prompt_type_col: str = "_prompt_type",
        valid_col: str = "_validation_scores",
        max_retries: int = 3,
    ) -> None:
        super().post_init(
            litellm_kwargs,
            system_template=system_template,
            output_col=output_col,
            prompt_type_col=prompt_type_col,
            max_retries=max_retries,
        )
        self.max_turns = max_turns
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.user_max_tokens = user_max_tokens
        self.assistant_max_tokens = assistant_max_tokens
        self.validator_list: list[Validator] = []
        self.valid_col = valid_col
        if validator_list:
            for validator in validator_list:
                self.validator_list.append(Validator(**validator))

    def get_magpie_prompt(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        if messages[-1]["role"] in ["system", "assistant"]:
            prompt = prompt + self.user_prefix
        else:
            prompt = prompt + self.assistant_prefix

        return prompt

    async def get_validator_result(self, dct: dict) -> tuple[bool, dict[str, float]]:
        scores = {}
        if self.validator_list:
            for validator in self.validator_list:
                valid, score = await validator.validate(dct, save_judge_result=True)
                scores[validator.name] = score
                if not valid:
                    return False, scores
        return True, scores

    async def get_magpie_one_turn(self, messages: list[dict]) -> list[dict]:
        prompt = self.get_magpie_prompt(messages)
        new_messages = []
        for _ in range(self.max_retries):
            try:
                resp = await self.completion(prompt=prompt, max_tokens=self.user_max_tokens)
            except Exception as e:
                self.logger.warning(f"User message generation error: {e}")
                continue
            if resp.choices[0].finish_reason == "stop":
                new_messages.append(
                    {"role": "user", "content": resp.choices[0].text.strip("\\n").strip()}
                )
                break
        else:
            self.logger.warning(
                f"cannot generate messages after {self.max_retries} retries. Skip this turn."
            )
            return []

        prompt = self.get_magpie_prompt(messages + new_messages)
        for _ in range(self.max_retries):
            try:
                resp = await self.completion(prompt=prompt, max_tokens=self.assistant_max_tokens)
            except Exception as e:
                self.logger.warning(f"Assistant message generation error: {e}")
                continue
            if resp.choices[0].finish_reason == "stop":
                new_messages.append(
                    {"role": "assistant", "content": resp.choices[0].text.strip("\\n").strip()}
                )
                break
        else:
            return []

        return new_messages

    async def run_each(self, dct: dict) -> dict:
        messages, prompt_types = self.get_messages(
            dct
        )  # list[dct[system_prompt]], list[dct[prompt_type]]

        retry_cnt = 0
        valid_scores_list = []
        while len(messages) <= self.max_turns // 2:
            new_messages = await self.get_magpie_one_turn(messages)
            if not new_messages:
                self.logger.warning(
                    f"cannot generate messages after {self.max_retries} retries. Skip this turn."
                )
                continue
            messages.extend(new_messages)

            # validate the generated messages
            cpy_dct = deepcopy(dct)
            cpy_dct[self.output_col] = messages[:]
            valid_res, valid_scores = await self.get_validator_result(cpy_dct)
            if not valid_res:
                messages = messages[:-2]
                retry_cnt += 1
                if retry_cnt > self.max_retries:
                    self.logger.warning(
                        f"Max retries reached. Returning the last valid messages. {messages}"
                    )
                    break
            else:
                retry_cnt = 0
                valid_scores_list.append([valid_scores])

        dct[self.output_col] = messages
        dct[self.prompt_type_col] = prompt_types
        dct[self.valid_col] = valid_scores_list
        return dct
