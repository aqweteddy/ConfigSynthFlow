from ...base import BasePipeline, DictsGenerator

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm
import asyncio
import random
import json
import time
from jinja2 import Template


class OpenaiTemplateMapper(BasePipeline):
    required_packages = ["jinja2", "openai"]
    
    def __post_init__(self, 
                      jinja_template: str, 
                      system_prompt: str = None,
                      output_col: str = "_prompt"):
        self.template = Template(jinja_template)
        self.output_col = output_col
        self.system_prompt = system_prompt
    
    def run_each(self, dct: dict) -> dict:
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        
        prompt = self.template.render(dct)
        messages.append({'role': 'user', 'content': prompt})
        dct[self.output_col] = messages
        return dct


class AsyncOpenAIChat(BasePipeline):
    def __post_init__(
        self,
        model: str = "gpt-4o-mini",
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        messages_col: str = "_prompt",
        output_col: str = "_response",
        concurrency: int = 100,
    ):
        """Pipeline to generate responses from OpenAI Chat API using AsyncAPI.
        
        Args:
            model (str, optional): OpenAI model name. Defaults to "gpt-4o-mini".
            openai_kwargs (dict, optional): OpenAI client kwargs. Defaults to None.
            gen_kwargs (dict, optional): OpenAI generation kwargs. For example, `temperature`, `max_tokens`, `response_format`, etc. Defaults to None.
            messages_col (str, optional): Input messages column name. Defaults to "messages".
            output_col (str, optional): Output column name. Defaults to "_response".
            concurrency (int, optional): Number of concurrent requests. Defaults to 100.
        """
        
        self.openai_kwargs = openai_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}

        if "model" not in self.gen_kwargs:
            self.gen_kwargs["model"] = model

        self.concurrency = concurrency
        self.messages_col = messages_col
        self.output_col = output_col

    @property
    def openai_client(self):
        if not hasattr(self, "_openai_client"):
            self._openai_client = AsyncOpenAI(**self.openai_kwargs)
        return self._openai_client

    def batch_generate(self, dcts: list[dict[str, str]]) -> list[str]:
        async def async_batch_generate() -> list[ChatCompletion]:
            tasks = []
            sem = asyncio.Semaphore(self.concurrency)
            async with AsyncOpenAI(**self.openai_kwargs) as client:
                async with sem:
                    for dct in dcts:
                        tasks.append(
                            client.chat.completions.create(
                                messages=dct[self.messages_col], 
                                **self.gen_kwargs
                            )
                        )
                return await tqdm.gather(*tasks, desc=self.class_name)

        resps = asyncio.run(async_batch_generate())
        return [resp.choices[0].message.content for resp in resps]

    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        dcts = list(dcts)
        resps = self.batch_generate(dcts)
        is_output_json = 'json' in self.gen_kwargs.get("response_format", {}).get("type", "")
        for dct, resp in zip(dcts, resps):
            if is_output_json:
                resp = self.read_json_str(resp)
            dct[self.output_col] = resp
            yield dct

    def read_json_str(self, json_str: str) -> dict:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return eval(json_str)
        except Exception:
            return "error"



class BatchOpenAIChat(AsyncOpenAIChat):
    def __post_init__(
        self,
        model="gpt-4o-mini",
        openai_kwargs=None,
        gen_kwargs=None,
        messages_col="messages",
        output_col="_response",
    ):
        """Pipeline to generate responses from OpenAI Chat API using BatchAPI.

        Args:
            model (str, optional): OpenAI model name. Defaults to "gpt-4o-mini".
            openai_kwargs (dict, optional): OpenAI client kwargs. Defaults to None.
            gen_kwargs (dict, optional): OpenAI generation kwargs. For example, `temperature`, `max_tokens`, `response_format`, etc. Defaults to None.
            messages_col (str, optional): Input messages column name. Defaults to "messages".
            output_col (str, optional): Output column name. Defaults to "_response".
        """
        self.openai_kwargs = openai_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        if "model" not in self.gen_kwargs:
            self.gen_kwargs["model"] = model

        self.messages_col = messages_col
        self.output_col = output_col
        self.client = OpenAI(**self.openai_kwargs)

    def __write_batch_to_tmp(self, dcts: list[dict]):
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
