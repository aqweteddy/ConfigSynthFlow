from typing import Any

from jinja2 import Template, TemplateError
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class PromptTemplate(BaseModel):
    """
    A template class for rendering prompts using Jinja2.

    This class wraps a Jinja2 template and provides methods for rendering
    the template with provided variables.

    Attributes:
        template_str (str): The Jinja2 template string.
        name (str): A descriptive name for the template.
        description (Optional[str]): An optional description of the template's purpose.
        _template (Template): The compiled Jinja2 template (created after initialization).
    """

    template_str: str = Field(..., description="The Jinja2 template string")
    name: str = "default"
    weight: float = 1.0
    template: Template | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Jinja2 Template object after model initialization."""
        self.template = Template(self.template_str)

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with the provided variables.

        Args:
            **kwargs: Variables to be passed to the template.

        Returns:
            str: The rendered template.

        Raises:
            TemplateError: If there's an error during template rendering.
        """
        try:
            return self.template.render(**kwargs)
        except TemplateError as e:
            raise TemplateError(f"Error rendering template '{self.name}': {str(e)}")

    @field_serializer("template")
    def serialize_template(self, template: Template) -> str:
        return self.template_str

    def __getstate__(self) -> dict[Any, Any]:
        dct = self.model_dump()
        dct.pop("template")
        return dct

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__init__(**state)
