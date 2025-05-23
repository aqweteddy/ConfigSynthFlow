import copy
import json
import random
from typing import Any, Literal, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from transformers import AutoTokenizer

from config_synth_flow.base.prompt_template import PromptTemplate
from config_synth_flow.base.validator import Validator
from config_synth_flow.pipelines.chat import ChatGenerator


class Query(BaseModel):
    query: str


class QueryAnswer(BaseModel):
    query: str
    answer: str


class ReasonQueryAnswer(BaseModel):
    query: str
    answer: str
    reason: str


class DocSelfInstruct(ChatGenerator):
    """
    Generate a query/answer pair from a text.

    example config:
    ```yaml
    - import_path: DocSelfInstruct
        async_cfg:
            concurrency: 250
        init_kwargs:
            text_col: "text"
            chunk_as_multiturn: true
            chunk_sliding_window_size: 2
            multiturn_user_template: "You are a helpful assistant. Please answer the user's question based on the context provided."
            separators: ["\n\n", "。", "？", "！", "……", "…", "..."]
            chunk_size: 300
            max_turns: 1
    ```
    """

    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        user_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        system_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        output_col: str = "messages",
        prompt_type_col: str = "prompt_type",
        max_retries: int = 3,
        force_json_mode: bool = False,
        # doc self instruct
        text_col: str = "text",
        chunk_as_multiturn: bool = True,
        chunk_sliding_window_size: int = 2,
        multiturn_user_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        separators: list[str] = ["\n\n", "。", "？", "！", "……", "…", "..."],
        chunk_size: int = 300,
        max_turns: int = 1,
        tokenizer_path: str = "google/gemma-3-27b-it",
        response_format: Literal[
            "query", "query_answer", "reason_query_answer", "freeform"
        ] = "reason_query_answer",
        validator_list: list[dict] | None = None,
        valid_col: str = "validation_scores",
    ) -> None:
        """
        Initialize the DocSelfInstruct pipeline.

        Args:
            litellm_kwargs: Configuration for the LLM API
            user_template: User template for the chat
            system_template: System template for the chat
            output_col: Column name for the output
            prompt_type_col: Column name for the prompt type
            max_retries: Maximum number of retries for the chat
            force_json_mode: Whether to force JSON mode for the chat
            text_col: Column name for the text
            chunk_as_multiturn: Whether to chunk the text as multiturn
            chunk_sliding_window_size: The size of the sliding window for chunking
            multiturn_user_template: User template for the multiturn chat
            separators: Separators for the text
            chunk_size: The size of the chunk
            max_turns: The maximum number of turns
            tokenizer_path: The path to the tokenizer
            response_format: The format of the response.
                "query": Query
                "query_answer": QueryAnswer
                "reason_query_answer": ReasonQueryAnswer
            validator_list: List of validators
            valid_col: Column name for the validation scores
        """
        super().post_init(
            litellm_kwargs,
            user_template,
            system_template,
            output_col,
            prompt_type_col,
            max_retries,
            force_json_mode,
        )
        self.text_col = text_col
        self.chunk_as_multiturn = chunk_as_multiturn
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None
        self.chunk_sliding_window_size = chunk_sliding_window_size
        self.separators = separators
        self.chunk_size = chunk_size
        self.max_turns = max_turns
        self.multiturn_user_template = self.get_template(multiturn_user_template)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            separators=separators,
            keep_separator=True,
            length_function=self.get_length,
        )

        self.response_format = {
            "query": Query,
            "query_answer": QueryAnswer,
            "reason_query_answer": ReasonQueryAnswer,
            "freeform": "freeform",
        }[response_format]

        self.validator_list: list[Validator] = []
        if validator_list:
            for validator in validator_list:
                self.validator_list.append(Validator(**validator))
        self.valid_col = valid_col

    def get_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def gen_one_turn_query_answer(
        self, dct: dict
    ) -> tuple[list[dict], Optional[Union[QueryAnswer, ReasonQueryAnswer]]]:
        gen_messages, prompt_types = [], []
        system_template = self.system_template

        if system_template:
            gen_messages.append({"role": "system", "content": system_template.render(**dct)})
            prompt_types.append({"role": "system", "type": system_template.name})

        if self.multiturn_user_template is None or len(gen_messages) == 0:
            user_template = self.user_template
        else:
            user_template = self.get_template(self.multiturn_user_template)

        prompt_types.append({"role": "user", "type": user_template.name})
        gen_messages.append({"role": "user", "content": user_template.render(**dct)})

        try:
            if self.response_format != "freeform":
                resp = await self.chat(gen_messages, response_format=self.response_format)
                obj = self.response_format(**json.loads(resp.choices[0].message.content))
            else:
                resp = await self.chat(gen_messages)
                obj = Query(query=resp.choices[0].message.content)
        except Exception as e:
            self.logger.warning(f"Failed to parse response. {e}")
            return prompt_types, None

        return prompt_types, obj

    def _prepare_validation_data(
        self,
        base_dict: dict[str, Any],
        messages: list[dict[str, str]],
        obj: Union[QueryAnswer, ReasonQueryAnswer],
    ) -> dict[str, Any]:
        """
        Prepare a dictionary for validation by adding the generated query and answer.

        Args:
            base_dict: The base dictionary to copy
            messages: The existing messages
            obj: The generated query/answer object

        Returns:
            A dictionary ready for validation
        """
        validation_dict = copy.deepcopy(base_dict)

        # Add the new messages based on the response format
        new_messages = messages.copy()
        new_messages.append({"role": "user", "content": obj.query})

        # For query_answer and reason_query_answer formats, also add the answer
        if hasattr(obj, "answer"):
            new_messages.append({"role": "assistant", "content": obj.answer})

        validation_dict[self.output_col] = new_messages
        return validation_dict

    async def _validate_and_generate(
        self, base_dict: dict[str, Any], messages: list[dict[str, str]]
    ) -> tuple[Optional[list[dict[str, str]]], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """
        Generate a query/answer pair and validate it.

        Args:
            base_dict: The base dictionary with text and other parameters
            messages: The existing messages

        Returns:
            A tuple containing:
            - The prompt types if successful, None otherwise
            - The generated object if successful, None otherwise
            - The validation scores if successful, None otherwise
        """
        for _ in range(self.max_retries):
            # Generate a new query/answer pair
            prompt_types, obj = await self.gen_one_turn_query_answer(base_dict)
            if not obj:
                continue

            # Prepare data for validation
            validation_dict = self._prepare_validation_data(base_dict, messages, obj)

            # Validate the generated content
            valid, score_dict = await self.get_validator_result(validation_dict)

            if valid:
                return prompt_types, obj, score_dict

        # If we get here, we failed after all retries
        self.logger.warning(f"Failed to generate and validate QA after {self.max_retries} retries.")
        return None, None, None

    async def run_each(self, dct: dict) -> dict:
        # Split text into chunks if needed
        if self.chunk_as_multiturn:
            raw_chunks = self.text_splitter.split_text(dct[self.text_col])
            chunks = []
            for i in range(len(raw_chunks)):
                tmp = raw_chunks[i : i + self.chunk_sliding_window_size]
                chunks.append("".join(tmp))
        else:
            chunks = [dct[self.text_col]]

        if len(chunks) == 0:
            return None

        # Select chunks based on max_turns
        if len(chunks) <= self.max_turns:
            idx = sorted(
                list(range(len(chunks)))
                + [random.randint(0, len(chunks) - 1)] * (self.max_turns - len(chunks))
            )
        else:
            idx = sorted(random.sample(list(range(len(chunks))), self.max_turns))

        chunks = [chunks[i] for i in idx]

        # Initialize result containers
        messages = [] if self.output_col not in dct else dct[self.output_col]
        prompt_types = [] if self.prompt_type_col not in dct else dct[self.prompt_type_col]
        valid_scores_list = []

        # Process each chunk
        for chunk in chunks:
            # Prepare base dictionary for this chunk
            base_dict = copy.deepcopy(dct)
            base_dict[self.text_col] = chunk
            base_dict[self.output_col] = messages

            # Generate and validate content for this chunk
            new_prompt_types, obj, score_dict = await self._validate_and_generate(
                base_dict, messages
            )

            if not obj:
                continue

            # Add the generated content to our results
            messages.append({"role": "user", "content": obj.query})
            if self.response_format in [QueryAnswer, ReasonQueryAnswer]:
                messages.append({"role": "assistant", "content": obj.answer})
            prompt_types.extend(new_prompt_types)
            valid_scores_list.append(score_dict)

        # Update the original dictionary with results
        dct[self.output_col] = messages
        dct[self.prompt_type_col] = prompt_types
        dct[self.valid_col] = valid_scores_list

        return dct
