import json
import random
import time

from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from ...base import AsyncBasePipeline, DictsGenerator


class AsyncOpenAIChat(AsyncBasePipeline):
    required_packages = ["openai", "jinja2"]

    def post_init(
        self,
        model: str = "gpt-4o-mini",
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        user_template: str = None,
        system_template: str = None,
        messages_col: str = "_prompt",
        output_col: str = "_response",
    ):
        """Pipeline to generate responses from OpenAI Chat API with asyncio. Notice that you can use openai_kwargs['base_url'] to use the custom openai endpoint.

        Args:
            model (str, optional): OpenAI model name. Defaults to "gpt-4o-mini".
            openai_kwargs (dict, optional): OpenAI client kwargs. Defaults to None.
            gen_kwargs (dict, optional): OpenAI generation kwargs. For example, `temperature`, `max_tokens`, `response_format`, etc. Defaults to None.
            messages_col (str, optional): column name for temp messages. If doesn't have `messages_col`, it will use `user_template` and `system_template` to generate messages. Defaults to "_prompt".
            output_col (str, optional): Output column name. Defaults to "_response".
        """

        self.openai_kwargs = openai_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}

        if "model" not in self.gen_kwargs:
            self.gen_kwargs["model"] = model

        self.messages_col = messages_col
        self.output_col = output_col
        self._user_template = user_template
        self._system_template = system_template
    @property
    def system_template(self) -> Template:
        if isinstance(self._system_template, str):
            self._system_template = Template(self._system_template)
        return self._system_template

    @property
    def user_template(self) -> Template:
        if isinstance(self._user_template, str):
            self._user_template = Template(self._user_template)
        return self._user_template

    @property
    def openai_client(self):
        if not hasattr(self, "_openai_client"):
            self._openai_client = AsyncOpenAI(**self.openai_kwargs)
        return self._openai_client

    def get_messages(
        self, dct: dict, messages_col: str = None, user_template: str = None
    ) -> list[dict]:
        """Get messages with the input dictionary and user_template/system_template.

        Args:
            dct (dict): Input dictionary.

        Returns:
            list[dict]: List of messages.
        """
        if messages_col is None and hasattr(self, "messages_col"):
            messages_col = self.messages_col

        user_template = user_template or self.user_template

        messages = []
        if self.system_template:
            messages.append(
                {"role": "system", "content": self.system_template.render(**dct)}
            )
        messages.append({"role": "user", "content": user_template.render(**dct)})
        return messages

    async def run_each(self, dct: dict) -> dict:
        """Generate a response for each dictionary using OpenAI Chat API.

        Args:
            dct (dict): Input dictionary.

        Returns:
            dict: Dictionary with the generated response.
        """
        dct[self.messages_col] = self.get_messages(dct, user_template=self.user_template)
        resp = await self.openai_client.chat.completions.create(
            messages=dct[self.messages_col], **self.gen_kwargs
        )
        resp: ChatCompletion = resp.choices[0].message.content
        is_output_json = "json" in self.gen_kwargs.get("response_format", {}).get(
            "type", ""
        )

        if is_output_json:
            resp = self.read_json_str(resp)
        dct[self.output_col] = resp
        return dct

    def read_json_str(self, json_str: str) -> dict:
        """Read a JSON string and convert it to a dictionary.

        Args:
            json_str (str): JSON string.

        Returns:
            dict: Parsed JSON dictionary.
        """
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[-1]
            json_str = json_str.replace("```", "").strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return eval(json_str)
            except Exception:
                return {}


class AsyncOpenAICompletion(AsyncOpenAIChat):
    async def run_each(self, dct: dict) -> dict:
        """Generate a completion for each dictionary using OpenAI Completion API.

        Args:
            dct (dict): Input dictionary.

        Returns:
            dict: Dictionary with the generated completion.
        """
        resp = await self.openai_client.completions.create(
            prompt=dct[self.messages_col][-1]["content"], **self.gen_kwargs
        )
        dct[self.output_col] = resp.choices[0].text
        return dct


class BatchOpenAIChat(AsyncOpenAIChat):
    def post_init(
        self,
        model: str = "gpt-4o-mini",
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        user_template: str = None,
        system_template: str = None,
        messages_col: str = "_prompt",
        output_col: str = "_response",
        batch_size: int = 100,
    ):
        """Pipeline to generate responses from OpenAI Chat API using BatchAPI. Notice that you can use openai_kwargs['base_url'] to use the custom openai endpoint.

        Args:
            model (str, optional): OpenAI model name. Defaults to "gpt-4o-mini".
            openai_kwargs (dict, optional): OpenAI client kwargs. Defaults to None.
            gen_kwargs (dict, optional): OpenAI generation kwargs. For example, `temperature`, `max_tokens`, `response_format`, etc. Defaults to None.
            messages_col (str, optional): Input messages column name. Defaults to "_prompt".
            output_col (str, optional): Output column name. Defaults to "_response".
            batch_size (int, optional): Batch size. Defaults to 100.
        """
        super().post_init(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
            user_template=user_template,
            system_template=system_template,
            messages_col=messages_col,
            output_col=output_col,
        )

        self.batch_size = batch_size

    @property
    def client(self):
        if not hasattr(self, "_client"):
            self._client = OpenAI(**self.openai_kwargs)
        return self._client

    def __write_batch_to_tmp(self, dcts: list[dict]):
        """Write batch data to a temporary file.

        Args:
            dcts (list[dict]): List of dictionaries to write.

        Returns:
            str: Path to the temporary file.
        """
        path = f"/tmp/batch_{random.randint(0, 100000)}.jsonl"
        f = open(path, "w")
        idx = 0

        for dct in dcts:
            lines = {
                "custom_id": f"{idx:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"messages": dct[self.messages_col], **self.gen_kwargs},
            }
            idx += 1
            f.write(json.dumps(lines) + "\n")
        f.close()

        return path

    def batch_generate(self, dcts: list[dict[str, str]]) -> list[str]:
        """Generate responses in batch using OpenAI Chat API.

        Args:
            dcts (list[dict[str, str]]): List of dictionaries to generate responses for.

        Returns:
            list[str]: List of generated responses.
        """
        dcts = list(dcts)
        file = self.__write_batch_to_tmp(dcts)
        batch_input_file = self.client.files.create(
            file=open(file, "rb"), purpose="batch"
        )
        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch_obj.id
        status = self.client.batches.retrieve(batch_id).status
        while status not in ["completed", "expired", "failed", "expired", "cancelled"]:
            batch_obj = self.client.batches.retrieve(batch_id)
            status = batch_obj.status
            self.logger.info(
                f"Batch status: {status}. Requests: {batch_obj.request_counts.completed} / {batch_obj.request_counts.total}"
            )
            time.sleep(random.randint(5, 15))

        if status in ["failed", "expired", "cancelled"]:
            raise ValueError(
                f"Batch failed with status: {status}, errors: {batch_obj.errors}"
            )

        resps = list(self.client.files.content(batch_obj.output_file_id).iter_lines())
        resps = [json.loads(resp) for resp in resps]
        resps = sorted(resps, key=lambda x: int(x["custom_id"]))

        result = []
        for resp in resps:
            try:
                result.append(
                    resp["response"]["body"]["choices"][0]["message"]["content"]
                )
            except KeyError:
                result.append("error")
        return result

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        for batch in self.get_chunk(dcts, self.batch_size):
            responses = self.batch_generate(batch)
            for dct, resp in zip(batch, responses):
                dct[self.output_col] = resp
                yield dct
