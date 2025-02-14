import random

from pydantic import BaseModel

from ..base import BaseAgent


class UserTemplateType(BaseModel):
    name: str = None
    template: str = None
    weight: float | None = None

    def model_post_init(self, __context):
        if self.template is None:
            for t in USER_TEMPLATE_TYPES:
                if t.name == self.name:
                    self.template = t.template
                    break


USER_TEMPLATE_TYPES = [
    UserTemplateType(
        name="argument",
        template="將以下文章轉化為一段包含論證結構的段落，提取主要觀點，並補充支持或反駁的論據，確保論點清晰但可能帶有邏輯不一致之處。\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="debate",
        template="根據以下文章，模擬一場辯論記錄，包含正方和反方的觀點。每一方的論述應具邏輯性並相互回應：\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="conversation",
        template="將以下文章轉化為模擬對話情境，創造至少兩個個性鮮明的角色，讓他們以對話方式討論文章中的核心內容：\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="meeting",
        template="將以下文章整理成正式的會議記錄格式，包含會議時間、地點、與會者、議程和主要討論內容：\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="instruction",
        template="將以下文章轉化為清晰的一步步教學說明，結構化為可操作的步驟，每步需簡明扼要:\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="textbook",
        template="將以下文章轉化為教科書章節，包含標題、目錄、內容段落，確保章節結構清晰，內容完整：\n\n# 文章: \n{{ text }}",
    ),
    # UserTemplateType(
    #     name="long_text",
    #     template="在以下文章的基礎上，擴充內容，加入相關背景資訊、例子或補充細節，使其內容更豐富\n\n# 文章: \n{{ text }}",
    # ),
    UserTemplateType(name="raw", template=""),
    UserTemplateType(
        name="story",
        template="將以下文章改寫為一個生動的故事，加入角色、情節發展和情感描寫，使讀者能夠沉浸在敘事中。你的故事必須包含文章的核心知識。\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="wiki",
        template="根據以下文章內容，改寫為類似維基百科的風格，包括標題、副標題和中立的語氣，並補充相關背景資訊和結構化格式：\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="social",
        template="請將以下內容改寫為適合社交媒體發布的短文，使用網絡俚語、表情符號，並確保字數不超過 280 個字符。內容應該輕鬆、有吸引力，並能夠在社交平台上引起互動。\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="humorous",
        template="請將以下內容改寫為一個幽默風格的版本，使用有趣的比喻、戲謔語氣，讓讀者在輕鬆的氛圍中理解信息。請確保信息仍然完整且具有說服力。\n\n# 文章: \n{{ text }}",
    ),
    UserTemplateType(
        name="marketing",
        template="請將以下內容改寫為一個具有吸引力的行銷文案，使用煽動性語言，強調產品的價值，並引導讀者採取行動（如購買、註冊、試用等）。請確保內容有足夠的吸引力，並包含明確的 Call-to-Action (CTA)。\n\n# 文章: \n{{ text }}",
    ),
]


class DocTransformAgent(BaseAgent):
    def post_init(
        self,
        model: str = None,
        openai_kwargs: dict = None,
        gen_kwargs: dict = None,
        system_template: str = None,
        output_col: str = "rephrase",
        user_template_list: list[UserTemplateType] = None,
    ):
        self.user_template_list = self.set_template_weight(
            user_template_list or USER_TEMPLATE_TYPES
        )
        super().post_init(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
            output_col=output_col,
            system_template=system_template,
        )

        self.logger.info(f"User template list: {self.user_template_list}")

    def set_template_weight(
        self, template_types: list[UserTemplateType]
    ) -> list[UserTemplateType]:
        """
        Set the weight for each user template type.

        Args:
            template_types (list[UserTemplateType]): List of user template types.

        Returns:
            list[UserTemplateType]: List of user template types with weights.
        """
        for t in template_types:
            t.weight = t.weight or 1

        total_weight = sum([t.weight for t in template_types])
        for t in template_types:
            t.weight = t.weight / total_weight
        return template_types

    async def run_agent(self, dct: dict) -> dict:
        """
        Generate responses from OpenAI Chat API with asyncio.

        Args:
            dct (dict): Input dictionary.

        Returns:
            dict: Dictionary with response text.
        """
        user_template = random.choices(
            self.user_template_list,
            weights=[t.weight for t in self.user_template_list],
        )[0]

        if user_template.name == "raw":
            return {"text": dct["text"], "template": user_template.name}

        messages = self.get_messages(dct, user_template=user_template.template)
        resp = await self.chat(messages)
        return {"text": resp, "template": user_template.name}
