import random
from copy import deepcopy
from typing import Any

from config_synth_flow.base import PromptTemplate, Validator
from config_synth_flow.pipelines.chat import ChatGenerator


class PromptEvolver(ChatGenerator):
    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        system_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        messages_col: str = "messages",
        output_col: str = "messages",
        prompt_type_col: str = "evolve_types",
        max_retries: int = 2,
        force_json_mode: bool = False,
        validator_list: list[Validator] | None = None,
        valid_col: str = "validation_scores",
        # prompt evolve
        evolve_times: int | list[int] = 3,
        first_prompt_template: list[PromptTemplate] | PromptTemplate | None = None,
        evolve_template: list[PromptTemplate] | PromptTemplate | None = None,
        gen_assistant_resp: bool = False,
        gen_assistant_resp_template: list[PromptTemplate] | PromptTemplate | None = None,
    ) -> None:
        super().post_init(
            litellm_kwargs=litellm_kwargs,
            user_template=None,
            system_template=system_template,
            output_col=output_col,
            prompt_type_col=prompt_type_col,
            max_retries=max_retries,
            force_json_mode=force_json_mode,
            validator_list=validator_list,
            valid_col=valid_col,
        )
        assert evolve_template is not None
        self._system_template = system_template or PromptTemplate(
            template_str="你是一個善於提問的專家助理，請你根據用戶提供的資訊，提出一個 prompt。你回答時直接輸出 prompt，不要輸出任何前後綴、解釋或說明。",
        )
        self.messages_col = messages_col
        self.evolve_template = (
            [evolve_template] if isinstance(evolve_template, PromptTemplate) else evolve_template
        )
        self.first_prompt_template = (
            [first_prompt_template]
            if isinstance(first_prompt_template, PromptTemplate)
            else first_prompt_template
        )
        self.evolve_times = evolve_times if isinstance(evolve_times, list) else [evolve_times]

        self.gen_assistant_resp = gen_assistant_resp
        self.gen_assistant_resp_template = (
            [gen_assistant_resp_template]
            if isinstance(gen_assistant_resp_template, PromptTemplate)
            else gen_assistant_resp_template
        ) or [PromptTemplate(template_str="{{ messages[-1].content }}")]

    async def evolve(self, dct: dict, evolve_times: int) -> tuple[dict, list[dict]]:
        prompt_types = []
        dct = deepcopy(dct)
        for _ in range(evolve_times):
            sys_template = self.system_template
            evol_template = self.get_template(self.evolve_template)
            evol_messages = [
                {"role": "system", "content": sys_template.render(**dct)},
                {"role": "user", "content": evol_template.render(**dct)},
            ]
            resp = await self.chat(evol_messages)
            dct[self.messages_col][-1]["content"] = resp["choices"][0]["message"]["content"]
            prompt_types.append({"role": "user", "prompt_type": evol_template.name})
        return dct[self.messages_col], prompt_types

    async def gen_first_prompt(self, dct: dict) -> dict:
        sys_template = self.get_template(self.system_template)
        first_prompt_template = self.get_template(self.first_prompt_template)
        first_messages = [
            {"role": "system", "content": sys_template.render(dct)},
            {"role": "user", "content": first_prompt_template.render(dct)},
        ]

        resp = await self.chat(first_messages)
        first_prompt = resp["choices"][0]["message"]["content"]
        return {"role": "user", "content": first_prompt}

    async def run_each(self, dct: dict) -> dict:
        if self.messages_col not in dct:
            dct[self.messages_col] = self.gen_first_prompt(dct)

        assert dct[self.messages_col][-1]["role"] == "user"

        for _ in range(self.max_retries):
            copy_dct = deepcopy(dct)

            evoled_messages, prompt_types = await self.evolve(
                copy_dct, random.choice(self.evolve_times)
            )
            copy_dct[self.output_col] = evoled_messages
            copy_dct[self.valid_col] = []
            copy_dct[self.prompt_type_col] = prompt_types

            if self.gen_assistant_resp:
                gen_assistant_resp_template = self.get_template(self.gen_assistant_resp_template)

                assistant_resp_messages = [
                    {"role": "user", "content": gen_assistant_resp_template.render(**copy_dct)},
                ]
                resp = await self.chat(assistant_resp_messages)
                copy_dct[self.output_col].append(
                    {"role": "assistant", "content": resp["choices"][0]["message"]["content"]}
                )
            valid, scores = await self.get_validator_result(copy_dct)
            if valid:
                dct[self.output_col] = copy_dct[self.output_col]
                dct[self.valid_col] = scores
                dct[self.prompt_type_col] = copy_dct[self.prompt_type_col]
                return dct

        return None
