import pickle
from unittest.mock import patch

import pytest
from jinja2 import Template, TemplateError

from config_synth_flow.base.prompt_template import PromptTemplate


class TestPromptTemplate:
    def test_initialization(self):
        """Test that PromptTemplate initializes correctly."""
        template_str = "Hello, {{ name }}!"
        name = "test_template"

        template = PromptTemplate(template_str=template_str, name=name)

        assert template.template_str == template_str
        assert template.name == name
        assert isinstance(template.template, Template)

    def test_render_success(self):
        """Test that template renders correctly with valid variables."""
        template_str = "Hello, {{ name }}! Welcome to {{ place }}."
        name = "test_template"

        template = PromptTemplate(template_str=template_str, name=name)
        result = template.render(name="John", place="Earth")

        assert result == "Hello, John! Welcome to Earth."

    def test_serialization(self):
        """Test that the template can be serialized and deserialized."""
        template_str = "Hello, {{ name }}!"
        name = "test_template"

        template = PromptTemplate(template_str=template_str, name=name)
        serialized = template.model_dump()

        assert serialized["template_str"] == template_str
        assert serialized["name"] == name

        # Check that the template field is serialized as the template_str
        assert "template" in serialized
        assert serialized["template"] == template_str

    def test_pickle_serialization(self):
        """Test that the template can be pickled and unpickled."""
        template_str = "Hello, {{ name }}!"
        name = "test_template"

        template = PromptTemplate(template_str=template_str, name=name)
        pickled = pickle.dumps(template)
        unpickled = pickle.loads(pickled)

        assert unpickled.template_str == template_str
        assert unpickled.name == name
        assert isinstance(unpickled.template, Template)

        # Test that the unpickled template still works
        result = unpickled.render(name="John")
        assert result == "Hello, John!"

    def test_complex_template(self):
        """Test rendering with a more complex template with conditionals and loops."""
        template_str = """
        {% if greeting %}{{ greeting }}{% else %}Hello{% endif %}, {{ name }}!
        
        {% if items %}
        Your items:
        {% for item in items %}
        - {{ item }}
        {% endfor %}
        {% else %}
        You have no items.
        {% endif %}
        """
        name = "complex_template"

        template = PromptTemplate(template_str=template_str, name=name)

        # Test with greeting and items
        result1 = template.render(
            greeting="Hi", name="John", items=["apple", "banana", "orange"]
        )
        assert "Hi, John!" in result1
        assert "Your items:" in result1
        assert "- apple" in result1
        assert "- banana" in result1
        assert "- orange" in result1

        # Test without greeting and with empty items
        result2 = template.render(name="John", items=[])
        assert "Hello, John!" in result2
        assert "You have no items." in result2

        # Test without items
        result3 = template.render(name="John")
        assert "Hello, John!" in result3
        assert "You have no items." in result3
