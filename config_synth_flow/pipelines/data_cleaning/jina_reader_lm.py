import re
from ..api import AsyncOpenAIChat


SCRIPT_PATTERN = re.compile(r"<[ ]*script.*?\/[ ]*script[ ]*>", re.IGNORECASE | re.MULTILINE | re.DOTALL)
STYLE_PATTERN = re.compile(r"<[ ]*style.*?\/[ ]*style[ ]*>", re.IGNORECASE | re.MULTILINE | re.DOTALL)
META_PATTERN = re.compile(r"<[ ]*meta.*?>", re.IGNORECASE | re.MULTILINE | re.DOTALL)
COMMENT_PATTERN = re.compile(r"<[ ]*!--.*?--[ ]*>", re.IGNORECASE | re.MULTILINE | re.DOTALL)
LINK_PATTERN = re.compile(r"<[ ]*link.*?>", re.IGNORECASE | re.MULTILINE | re.DOTALL)
BASE64_IMG_PATTERN = re.compile(r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>', re.IGNORECASE | re.MULTILINE | re.DOTALL)
SVG_PATTERN = re.compile(r"(<svg[^>]*>)(.*?)(<\/svg>)", re.IGNORECASE | re.MULTILINE | re.DOTALL)

def clean_html(html: str) -> str:
    for pattern in [SCRIPT_PATTERN, STYLE_PATTERN, META_PATTERN, COMMENT_PATTERN, LINK_PATTERN]:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(BASE64_IMG_PATTERN, '<img src="#"/>', html)
    html = re.sub(SVG_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return html


class JinaHtml2Markdown(AsyncOpenAIChat):
    def __post_init__(
        self,
        model="gpt-4o-mini",
        openai_kwargs=None,
        gen_kwargs=None,
        html_col: str = None,
        output_col="text",
    ):
        super().__post_init__(
            model=model,
            openai_kwargs=openai_kwargs,
            gen_kwargs=gen_kwargs,
            output_col=output_col,
        )
        self.html_col = html_col
        if 'frequency_penalty' not in self.gen_kwargs:
            self.gen_kwargs['frequency_penalty'] = 1.08

    async def run_each(self, dct: dict) -> dict:
        dct[self.html_col] = clean_html(dct[self.html_col])
        prompt = f'Extract the main content from the given HTML and convert it to Markdown format.\n```html\n{dct[self.html_col]}\n```'
        dct[self.messages_col] = [
            {"role": "user", "content": prompt},
        ]

        res = None

        for i in range(6):
            try:
                res = await super().run_each(dct)
                break
            except:
                self.logger.warning(f"Retry {i + 1} times")
                dct[self.html_col] = dct[self.html_col][
                    : len(dct[self.html_col]) // 3 * 2
                ]
                dct[self.messages_col] = [
                    {"role": "user", "content": dct[self.html_col]},
                ]
                res = dct

        res.pop(self.messages_col)
        return res
