from jinja2 import Template

from .base import BaseAgent
from .constant import DEFAULT_SELF_INSTRUCT_FROM_DOC_TEMPLATE
from .QA_refinement import QueryResponseItem


class DocSelfInstructAgent(BaseAgent):
    """
    Agent for generating QA from a document.
    """

    def post_init(
        self,
        model="gpt-4o-mini",
        openai_kwargs=None,
        gen_kwargs=None,
        output_col="messages",
        system_prompt: str = "你是一個專注於從文章生成高品質問題與答案的 AI 助理，並嚴格遵守指定格式。",
        self_instruct_template: str = DEFAULT_SELF_INSTRUCT_FROM_DOC_TEMPLATE,
    ):
        super().post_init(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
            output_col=output_col,
        )
        self.self_instruct_template = Template(self_instruct_template)
        self.system_prompt = system_prompt
        self.gen_kwargs["response_format"] = QueryResponseItem

    async def run_agent(self, dct: dict) -> list:
        prompt = self.self_instruct_template.render(dct)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp: QueryResponseItem = await self.chat(messages)
        return [
            {"role": "user", "content": resp.query},
            {"role": "assistant", "content": resp.response},
        ]
