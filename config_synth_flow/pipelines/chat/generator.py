import json
import random
from typing import Any

import yaml

from config_synth_flow.base import AsyncChatBasePipeline, PromptTemplate, Validator


class ChatGenerator(AsyncChatBasePipeline):
    """Generate responses using a LLM-based chat pipeline with customizable templates.

    This class provides functionality to generate responses using a chat model with
    configurable system and user templates. Templates can be loaded from JSON, JSONL,
    or YAML files, or provided directly as PromptTemplate objects or lists.

    The generator supports validation of outputs through configurable validators,
    automatic retries for failed generations, and forcing JSON response format.

    Example YAML template format:
    ```yaml
    - name: system_template_1
      template: "You are a helpful assistant with expertise in {domain}."
      weight: 1.0
    - name: user_template_1
      template: "Please help me with {query}."
      weight: 0.7
    - name: user_template_2
      template: "I need assistance with {query} related to {domain}."
      weight: 0.3
    ```

    Attributes:
        output_col: Column name to store the generated output
        prompt_type_col: Column name to store template type information
        max_retries: Maximum number of retries for failed generations
        force_json_mode: Force model to return JSON responses
        validator_list: List of validators to evaluate generated responses
        valid_col: Column name to store validation scores
    """

    def post_init(
        self,
        litellm_kwargs: dict[str, Any],
        user_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        system_template: PromptTemplate | list[PromptTemplate] | str | None = None,
        output_col: str = "output",
        prompt_type_col: str = "_prompt_type",
        max_retries: int = 3,
        force_json_mode: bool = False,
        validator_list: list[Validator] | None = None,
        valid_col: str = "validation_scores",
    ) -> None:
        """Initialize the ChatGenerator with configuration parameters.

        Args:
            litellm_kwargs: Parameters for the LLM provider via litellm
            user_template: User prompt template (file path, PromptTemplate, list, or None)
            system_template: System prompt template (file path, PromptTemplate, list, or None)
            output_col: Column name to store generation outputs
            prompt_type_col: Column name to store prompt type information
            max_retries: Maximum number of generation retry attempts
            force_json_mode: Force JSON response format regardless of LLM settings
            validator_list: List of validator configurations
            valid_col: Column name to store validation scores
        """
        super().post_init(litellm_kwargs)
        self._system_template = system_template
        self._user_template = user_template
        self.output_col = output_col
        self.prompt_type_col = prompt_type_col
        self.max_retries = max_retries
        self.force_json_mode = force_json_mode
        self.validator_list = validator_list

        self.validator_list: list[Validator] = []
        if validator_list:
            for validator in validator_list:
                self.validator_list.append(Validator(**validator))

        self.valid_col = valid_col

    async def get_validator_result(self, dct: dict) -> tuple[bool, dict[str, float]]:
        """Run all validators on the input dictionary and return validation results.

        Args:
            dct: Dictionary containing data to validate

        Returns:
            tuple: (is_valid, scores_dict) where:
                - is_valid: Boolean indicating if all validators passed
                - scores_dict: Dictionary mapping validator names to their scores
        """
        scores = {}
        if self.validator_list:
            for validator in self.validator_list:
                valid, score = await validator.validate(dct, save_judge_result=True)
                scores[validator.name] = score
                if not valid:
                    return False, scores
        return True, scores

    def __read_template(self, path: str) -> list[PromptTemplate]:
        """Load templates from a file (JSONL, JSON, or YAML).

        Args:
            path: Path to the template file

        Returns:
            list: List of PromptTemplate objects loaded from the file

        Raises:
            ValueError: If the file format is not supported
        """
        templates = []
        if path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                if line:
                    dct = json.loads(line)
                    templates.append(PromptTemplate(**dct))
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
                templates = [PromptTemplate(**dct) for dct in data]
        elif path.endswith("yml") or path.endswith("yaml"):
            with open(path) as f:
                data = yaml.safe_load(f)
                templates = [PromptTemplate(**dct) for dct in data]
        else:
            raise ValueError(f"Invalid template file: {path}")
        return templates

    def sample_template(self, templates: list[PromptTemplate]) -> PromptTemplate:
        """Sample a template from a list based on template weights.

        Args:
            templates: List of PromptTemplate objects with associated weights

        Returns:
            PromptTemplate: A randomly selected template according to weights
        """
        return random.choices(
            templates,
            weights=[t.weight for t in templates],
            k=1,
        )[0]

    @property
    def system_template(self) -> PromptTemplate:
        """Get the processed system template.

        Returns:
            PromptTemplate or None: The selected system template
        """
        return self.get_template(self._system_template)

    @property
    def user_template(self) -> PromptTemplate:
        """Get the processed user template.

        Returns:
            PromptTemplate or None: The selected user template
        """
        return self.get_template(self._user_template)

    def get_template(
        self, template: str | PromptTemplate | list[PromptTemplate] | None
    ) -> PromptTemplate | None:
        """Process template input to return a usable PromptTemplate.

        Handles various input formats:
        - Path to template file
        - PromptTemplate object
        - List of PromptTemplate objects (will sample one)
        - None

        Args:
            template: Template in one of the supported formats

        Returns:
            PromptTemplate or None: The selected template or None if input was None
        """
        if isinstance(template, str):
            template = self.__read_template(template)
        if isinstance(template, PromptTemplate):
            return template
        elif isinstance(template, list):
            return self.sample_template(template)
        elif template is None:
            return None

    def get_messages(
        self,
        dct: dict,
        system_template: PromptTemplate | None = None,
        user_template: PromptTemplate | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Generate chat messages and their prompt types from templates.

        Args:
            dct: Dictionary containing values to populate template variables

        Returns:
            tuple: (messages, prompt_types) where:
                - messages: List of message dictionaries for the chat API
                - prompt_types: List of dictionaries with template type information
        """
        messages, prompt_type = [], []
        system_template = system_template or self.system_template
        user_template = user_template or self.user_template
        if system_template:
            messages.append({"role": "system", "content": system_template.render(**dct)})
            prompt_type.append({"role": "system", "type": system_template.name})
        if user_template:
            messages.append({"role": "user", "content": user_template.render(**dct)})
            prompt_type.append({"role": "user", "type": user_template.name})

        return messages, prompt_type

    async def run_each(self, dct: dict) -> dict:
        """Process a single input dictionary through the chat generation pipeline.

        This method:
        1. Builds chat messages from templates
        2. Calls the LLM with retry logic
        3. Handles JSON parsing if required
        4. Updates the input dictionary with results

        Args:
            dct: Input dictionary containing data for template rendering

        Returns:
            dict: Updated dictionary with generation results
        """
        json_mode = (
            self.litellm_kwargs.get("response_format", {}).get("type") == "json_schema"
            or self.force_json_mode
        )

        messages, prompt_types = self.get_messages(dct)

        for _ in range(self.max_retries):
            try:
                resp = await self.chat(messages=messages)
                resp = resp.choices[0].message.content
                if json_mode:
                    resp = self.read_json_resp(resp)
                break
            except Exception as e:
                self.logger.warning(f"Failed to generate response: {e}")
                resp = {} if json_mode else ""
                continue
        dct[self.output_col] = resp

        dct[self.prompt_type_col] = prompt_types
        return dct

    def read_json_resp(self, resp: str) -> dict:
        """Parse a JSON response from the model, handling various formats.

        Attempts to clean and parse JSON responses, including:
        - Removing markdown code block syntax
        - Parsing with json.loads
        - Fallback to eval for malformed JSON (use with caution)

        Args:
            resp: String response from the model

        Returns:
            dict: Parsed JSON dictionary

        Raises:
            Exception: If the response cannot be parsed
        """
        resp = resp.strip()
        # Remove markdown code block syntax if present
        if resp.startswith("```json"):
            resp = resp[7:]
        elif resp.startswith("```"):
            resp = resp[3:]

        if resp.endswith("```"):
            resp = resp[:-3]

        resp = resp.strip()

        try:
            return json.loads(resp)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response with json.loads: {e}")
            try:
                # Use eval as a fallback mechanism as requested
                # Note: eval can execute arbitrary code, so use with caution
                return eval(resp)
            except Exception as e2:
                self.logger.warning(f"Failed to parse with eval fallback: {e2}")
                raise e2
