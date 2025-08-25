from copy import deepcopy
from typing import Any, Optional

from transformers import AutoTokenizer

from config_synth_flow.base import PromptTemplate
from config_synth_flow.base.validator import Validator
from config_synth_flow.pipelines.chat import ChatGenerator


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

    def get_magpie_prompt(self, messages: list[dict[str, str]]) -> str:
        """Generate a prompt for the Magpie model using the tokenizer's chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string ready for completion
        """
        add_generation_prompt = messages[-1]['role'] == 'user'
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )

        if not add_generation_prompt:
            prompt = prompt + self.user_prefix
        
        return prompt

    async def _generate_completion_with_retry(
        self, 
        prompt: str, 
        max_tokens: int, 
        role: str
    ) -> Optional[str]:
        """Generate a completion with retry logic.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            role: Role for logging purposes ('user' or 'assistant')
            
        Returns:
            Generated content or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                resp = await self.completion(prompt=prompt, max_tokens=max_tokens)
                if resp.choices[0].finish_reason == "stop":
                    return resp.choices[0].text.strip("\n").strip()
            except Exception as e:
                self.logger.warning(f"{role.title()} message generation error (attempt {attempt + 1}): {e}")
                continue
        
        self.logger.warning(f"Failed to generate {role} message after {self.max_retries} retries")
        return None

    async def get_magpie_one_turn(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Generate one complete turn (user + assistant) of conversation.
        
        Args:
            messages: Existing conversation messages
            
        Returns:
            List containing new user and assistant messages, or empty list if generation failed
        """
        # Generate user message
        user_prompt = self.get_magpie_prompt(messages)
        user_content = await self._generate_completion_with_retry(
            user_prompt, self.user_max_tokens, "user"
        )
        
        if user_content is None:
            return []
        
        new_messages = [{"role": "user", "content": user_content}]
        
        # Generate assistant message
        assistant_prompt = self.get_magpie_prompt(messages + new_messages)
        assistant_content = await self._generate_completion_with_retry(
            assistant_prompt, self.assistant_max_tokens, "assistant"
        )
        
        if assistant_content is None:
            return []
        
        new_messages.append({"role": "assistant", "content": assistant_content})
        return new_messages

    async def _validate_messages(self, dct: dict[str, Any], messages: list[dict[str, str]]) -> tuple[bool, dict[str, float]]:
        """Validate generated messages using configured validators.
        
        Args:
            dct: Base dictionary for validation context
            messages: Messages to validate
            
        Returns:
            Tuple of (is_valid, validation_scores)
        """
        if not self.validator_list:
            return True, {}
        
        validation_dict = deepcopy(dct)
        validation_dict[self.output_col] = messages[:]
        
        return await self.get_validator_result(validation_dict)

    async def run_each(self, dct: dict[str, Any]) -> dict[str, Any]:
        """Process a single input dictionary to generate multi-turn conversations.
        
        Args:
            dct: Input dictionary containing conversation context
            
        Returns:
            Dictionary with generated messages, prompt types, and validation scores
        """
        system_template = self.system_template
        
        messages, prompt_types = self.get_messages(
            dct, system_template=system_template
        )

        retry_cnt = 0
        valid_scores_list = []
        max_conversation_turns = self.max_turns // 2
        
        while len(messages) <= max_conversation_turns:
            new_messages = await self.get_magpie_one_turn(messages)
            if not new_messages:
                self.logger.warning("Cannot generate messages. Skipping this turn.")
                continue
            
            # Extend messages and validate
            messages.extend(new_messages)
            valid_res, valid_scores = await self._validate_messages(dct, messages)
            
            if not valid_res:
                # Remove the failed turn and increment retry counter
                messages = messages[:-2]
                retry_cnt += 1
                
                if retry_cnt > self.max_retries:
                    self.logger.warning(
                        f"Max retries ({self.max_retries}) reached. "
                        f"Returning conversation with {len(messages)} messages."
                    )
                    break
            else:
                # Reset retry counter and save validation scores
                retry_cnt = 0
                valid_scores_list.append([valid_scores])

        # Update output dictionary
        dct[self.output_col] = messages
        dct[self.prompt_type_col] = prompt_types
        dct[self.valid_col] = valid_scores_list
        
        return dct
