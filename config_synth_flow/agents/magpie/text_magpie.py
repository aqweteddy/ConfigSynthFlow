from openai.types import Completion
from transformers import AutoTokenizer

from ..base import BaseAgent

DEFAULT_SYSTEM_PROMPT = """你是一個專注於從文章中提取相關資訊並以台灣語境的繁體中文回答問題的 AI 助理。請遵循以下指導原則：

### User 問題特徵

1. 問題會與文章內容相關，但使用者總是完整的描述問題，不會提到「文章」或「根據文章」等字眼。
2. 問題包含明確的關鍵字或關鍵實體 (例如產品名稱、數字、編號)。
3. 問題類型會包括以下之一：
   - 產品推薦或服務比較
   - 數據/表格的提取與分析
   - 查詢特定產品/編號細節
   - 合規性檢查
4. 如果文章無法回答問題，你需要清楚說明未找到相關資訊。

### 回答原則

1. 回答應使用繁體中文且應該簡潔有力，避免冗長敘述。
2. 避免提及「根據文章」等字眼，但可以以自然方式表達資訊的來源。
3. 先分析問題需求再簡短有力的作答。使用標籤標註分析過程。
4. 如果問題提到的關鍵字、關鍵實體或需要的資訊在文章中未提及，回答應寫出「無法找到相關資訊」。""".strip()

DEFAULT_FIRST_TURN = [
    """# 文章
{{ text[:4000] }}
""".strip(),  # USER
    """
我將遵守以下指示回答問題：
1. 回答應簡潔有力，避免冗長敘述。
2. 避免提及「根據文章」等字眼，但可以以自然方式表達資訊的來源。
3. 如果問題提到的關鍵字、關鍵實體或需要的資訊在文章中未提及，回答應寫出「無法找到相關資訊」。

請告訴我你的問題！""".strip(),  # ASSISTANT
]


class MagpieBasedOnTextAgent(BaseAgent):
    def post_init(
        self,
        model="default",
        tokenizer_path: str = None,
        openai_kwargs=None,
        gen_kwargs=None,
        system_template: str = None,
        banned_words: list[str] = None,
        first_turn: list[str] = None,
        num_max_turns: int = 3,
        output_col: str = "messages",
        user_prefix: str = "<|im_start|>user\n",
        assistant_prefix: str = "<|im_start|>assistant\n",
    ):
        system_template = system_template or DEFAULT_SYSTEM_PROMPT
        super().post_init(
            model,
            openai_kwargs,
            gen_kwargs,
            output_col=output_col,
            system_template=system_template,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model)
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.first_turn = first_turn  # [USER, ASSISTANT]
        self.banned_words = banned_words or ["文章", "文中"]
        self.num_max_turns = num_max_turns

        if "max_tokens" not in self.gen_kwargs:
            self.gen_kwargs["max_tokens"] = 2048

    def check_completion(self, text: str) -> bool:
        for word in self.banned_words:
            if word in text:
                return False
        return True

    async def chat(self, messages: list[dict[str, str]], role: str) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt += self.user_prefix if role == "user" else self.assistant_prefix
        for i in range(3):
            resp: Completion = await self.openai_client.completions.create(
                prompt=prompt, **self.gen_kwargs
            )
            result = []
            for c in resp.choices:
                if c.finish_reason == "stop" and self.check_completion(c.text):
                    result.append(c.text)
            if len(result) != 0:
                return result[0]
        return ""

    async def run_agent(self, dct: dict) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self.system_template.render(**dct)},
        ]

        if self.first_turn:
            messages.append({"role": "user", "content": self.first_turn[0]})
            messages.append({"role": "assistant", "content": self.first_turn[1]})

        for _ in range(self.num_max_turns):
            user = await self.chat(messages, "user")
            if not user:
                break
            assistant = await self.chat(
                messages + [{"role": "user", "content": user}], "assistant"
            )
            if not assistant:
                break
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        if self.first_turn:
            return messages[3:]
        else:
            return messages[1:]
