import random
from copy import deepcopy
from typing import Any, Optional

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
        # contextual magpie specific parameters
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
        """Calculate text length using tokenizer encoding."""
        return len(self.tokenizer.encode(text))

    def get_contextual_system_template(self, dct: dict, text: str) -> tuple[str, str]:
        """Generate contextual system template with given text context.
        
        Args:
            dct: Dictionary containing template variables
            text: Context text to include in template
            
        Returns:
            Tuple of (rendered_template, template_name)
        """
        template = self.system_template
        # Filter out None values and add text context
        filtered_dct = {k: v for k, v in dct.items() if v is not None}
        filtered_dct[self.text_col] = text
        return template.render(**filtered_dct), template.name

    def _prepare_chunks(self, text: str) -> list[str]:
        """Split text into chunks and prepare overlapping chunks for better context.
        
        Args:
            text: Input text to split
            
        Returns:
            List of processed chunks, empty if no valid chunks
        """
        chunks = self.text_splitter.split_text(text)
        
        if not chunks:
            return []
        
        # Create overlapping chunks by combining adjacent chunks
        # This provides better context continuity
        if len(chunks) > 1:
            overlapping_chunks = [
                chunks[i] + chunks[i + 1] for i in range(len(chunks) - 1)
            ]
            overlapping_chunks.append(chunks[-1])  # Add the last chunk as-is
            return overlapping_chunks
        
        return chunks

    def _select_chunk_indices(self, num_chunks: int) -> list[int]:
        """Select chunk indices based on max_turns constraint.
        
        Args:
            num_chunks: Total number of available chunks
            
        Returns:
            Sorted list of selected chunk indices
        """
        if num_chunks <= self.max_turns:
            # If we have fewer chunks than max_turns, use all chunks
            # and randomly repeat some to reach max_turns
            base_indices = list(range(num_chunks))
            additional_indices = [
                random.randint(0, num_chunks - 1) 
                for _ in range(self.max_turns - num_chunks)
            ]
            return sorted(base_indices + additional_indices)
        else:
            # Randomly sample max_turns chunks
            return sorted(random.sample(list(range(num_chunks)), self.max_turns))

    async def _process_chunk_turn(
        self, 
        chunk: str, 
        messages: list[dict], 
        dct: dict, 
        prompt_types_list: list[dict],
        valid_scores_list: list[list[dict]]
    ) -> bool:
        """Process a single chunk turn with retry logic.
        
        Args:
            chunk: Text chunk to process
            messages: Current messages list (modified in-place)
            dct: Original data dictionary
            prompt_types_list: List to append prompt types (modified in-place)
            valid_scores_list: List to append validation scores (modified in-place)
            
        Returns:
            True if turn was successfully processed, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                # Generate new messages for this turn
                new_messages = await self.get_magpie_one_turn(messages)
                if not new_messages:
                    continue

                # Prepare validation data
                validation_dct = deepcopy(dct)
                validation_dct[self.output_col] = messages[1:] + new_messages
                
                # Validate the generated content
                valid_res, valid_scores = await self.get_validator_result(validation_dct)
                
                if valid_res:
                    # Success: update messages and metadata
                    messages.extend(new_messages)
                    
                    # Update system message with new chunk context
                    messages[0]["content"], new_prompt_type = self.get_contextual_system_template(
                        dct, chunk
                    )
                    
                    prompt_types_list.append({"role": "system", "type": new_prompt_type})
                    valid_scores_list.append([valid_scores])
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Error processing chunk turn (attempt {attempt + 1}): {e}")
                continue
        
        # All retries failed
        self.logger.warning(
            f"Validator failed after {self.max_retries} retries. Skipping this turn."
        )
        return False

    async def run_each(self, dct: dict) -> Optional[dict]:
        """Process a single data item through the contextual magpie pipeline.
        
        Args:
            dct: Input data dictionary
            
        Returns:
            Processed dictionary with generated messages, or None if processing fails
        """
        # Early return for non-multiturn mode
        if not self.chunk_as_multiturn:
            return await super().run_each(dct)

        # Validate required input
        if self.text_col not in dct:
            self.logger.error(f"Required column '{self.text_col}' not found in input data")
            return None
            
        text = dct[self.text_col]
        if not text or not isinstance(text, str):
            self.logger.warning(f"Invalid or empty text in column '{self.text_col}'")
            return None

        # Prepare chunks
        chunks = self._prepare_chunks(text)
        if not chunks:
            self.logger.warning("No valid chunks generated from input text")
            return None

        # Select chunk indices
        chunk_indices = self._select_chunk_indices(len(chunks))
        selected_chunks = [chunks[i] for i in chunk_indices]

        # Initialize processing state
        prompt_types_list = []
        valid_scores_list = []
        
        # Store original text and set up initial context
        original_text = dct[self.text_col]
        dct[self.text_col] = selected_chunks[0]
        
        try:
            # Get initial messages with first chunk
            messages, prompt_types = self.get_messages(dct)
            prompt_types_list.extend(prompt_types)

            # Process remaining chunks
            for chunk in selected_chunks[1:]:
                success = await self._process_chunk_turn(
                    chunk, messages, dct, prompt_types_list, valid_scores_list
                )
                if not success:
                    # Continue processing even if one turn fails
                    continue

            # Finalize results
            dct[self.output_col] = messages[1:]  # Exclude system message
            dct[self.prompt_type_col] = prompt_types_list
            dct[self.valid_col] = valid_scores_list
            
        finally:
            # Always restore original text
            dct[self.text_col] = original_text

        return dct
