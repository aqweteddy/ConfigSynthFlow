import random
from copy import deepcopy
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_synth_flow.base.prompt_template import PromptTemplate

from .magpie import Magpie


class ContextualMagpie(Magpie):
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
        output_col: str = "messages",
        prompt_type_col: str = "_prompt_type",
        valid_col: str = "validation_scores",
        max_retries: int = 3,
        # contextual magpie
        text_col: str = "text",
        chunk_size: int = 300,
        separators: list[str] = ["\n\n", "。", "？", "！", "……", "…", "..."],
        chunk_as_multiturn: bool = True,
    ) -> None:
        super().post_init(
            litellm_kwargs=litellm_kwargs,
            tokenizer_path=tokenizer_path,
            system_template=system_template,
            max_turns=max_turns,
            validator_list=validator_list,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            user_max_tokens=user_max_tokens,
            assistant_max_tokens=assistant_max_tokens,
            output_col=output_col,
            prompt_type_col=prompt_type_col,
            max_retries=max_retries,
            valid_col=valid_col,
        )

        self.text_col = text_col
        self.chunk_size = chunk_size
        self.chunk_as_multiturn = chunk_as_multiturn
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            separators=separators,
            keep_separator=True,
            length_function=self.get_length,
        )

    def get_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_contextual_system_template(self, dct: dict, text: str) -> tuple[PromptTemplate, str]:
        template = self.system_template
        dct = {k: v for k, v in dct.items() if v is not None}
        dct[self.text_col] = text
        return template.render(**dct), template.name

    async def run_each(self, dct: dict) -> dict:
        if not self.chunk_as_multiturn:
            return await super().run_each(dct)

        text = dct[self.text_col]

        chunks = self.text_splitter.split_text(
            text,
        )

        if len(chunks) == 0:
            return None

        chunks = [chunks[i] + chunks[i + 1] for i in range(len(chunks) - 1)] + [chunks[-1]]

        if len(chunks) <= self.max_turns:
            idx = sorted(
                list(range(len(chunks)))
                + [random.randint(0, len(chunks) - 1)] * (self.max_turns - len(chunks))
            )
        else:
            idx = sorted(random.sample(list(range(len(chunks))), self.max_turns))

        chunks = [chunks[i] for i in idx]

        prompt_types_list = []
        original_text = dct[self.text_col]
        dct[self.text_col] = chunks[0]
        messages, prompt_types = self.get_messages(dct)
        prompt_types_list.extend(prompt_types)
        valid_scores_list = []
        for chunk in chunks[1:]:
            for i in range(self.max_retries):
                new_messages = await self.get_magpie_one_turn(messages)
                if not new_messages:
                    continue

                cpy_dct = deepcopy(dct)
                cpy_dct[self.output_col] = messages[1:] + new_messages
                valid_res, valid_scores = await self.get_validator_result(cpy_dct)
                if valid_res:
                    messages.extend(new_messages)
                    messages[0]["content"], new_prompt_type = self.get_contextual_system_template(
                        dct, chunk
                    )
                    prompt_types_list.append({"role": "system", "type": new_prompt_type})
                    valid_scores_list.append([valid_scores])
                    break
                elif i == self.max_retries - 1:
                    self.logger.warning(
                        f"Validator failed after {self.max_retries} retries. Skip this turn."
                    )
                    break
        dct[self.output_col] = messages[1:]
        dct[self.prompt_type_col] = prompt_types_list
        dct[self.valid_col] = valid_scores_list
        dct[self.text_col] = original_text
        return dct
