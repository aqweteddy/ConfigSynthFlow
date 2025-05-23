import httpx

from config_synth_flow.base import AsyncBasePipeline, DictsGenerator, PromptTemplate


class FedGPTChat(AsyncBasePipeline):
    def post_init(
        self,
        user_template: PromptTemplate,
        api_key: str,
        system_template: PromptTemplate | None = None,
        api_base: str = "https://10.7.240.11",
        output_col: str = "response",
        response_format: dict = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.api = f"{api_base.strip('/')}/api/chat/v1/chat"
        self.output_col = output_col
        self.user_template = user_template
        self.system_template = system_template
        self.api_key = api_key
        self.response_format = response_format
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def client(self) -> httpx.AsyncClient:
        # if not hasattr(self, "_client"):
        _client = httpx.AsyncClient(
            headers={"X-Api-Key": self.api_key}, verify=False, timeout=self.timeout
        )
        return _client

    async def run_each(self, dct: dict) -> DictsGenerator:
        messages = []
        if self.system_template:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_template.render(**dct),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": self.user_template.render(**dct),
            }
        )
        req = {
            "model": "fedgpt-medium",
            "mode": "normal",
            "messages": messages,
            "responseFormat": self.response_format,
        }
        for _ in range(self.max_retries):
            try:
                resp = await self.client.post(
                    url=self.api,
                    json=req,
                )
                output = resp.json()
                dct[self.output_col] = output["messages"][-1]["content"]
                break
            except Exception as e:
                self.logger.warning(f"Failed to get response from FedGPT. {e}")

        return dct
