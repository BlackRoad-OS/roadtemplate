"""
RoadTemplate - Template Engine for BlackRoad
Render templates with variables, loops, conditionals, and inheritance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern
import html
import logging
import re

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    pass


@dataclass
class Token:
    type: str
    value: str
    line: int = 0


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1

    def tokenize(self) -> List[Token]:
        tokens = []
        text_buffer = ""
        
        while self.pos < len(self.source):
            if self.source[self.pos:self.pos+2] == "{{":
                if text_buffer:
                    tokens.append(Token("TEXT", text_buffer, self.line))
                    text_buffer = ""
                tokens.append(self._read_expression())
            elif self.source[self.pos:self.pos+2] == "{%":
                if text_buffer:
                    tokens.append(Token("TEXT", text_buffer, self.line))
                    text_buffer = ""
                tokens.append(self._read_statement())
            elif self.source[self.pos:self.pos+2] == "{#":
                if text_buffer:
                    tokens.append(Token("TEXT", text_buffer, self.line))
                    text_buffer = ""
                self._read_comment()
            else:
                if self.source[self.pos] == "\n":
                    self.line += 1
                text_buffer += self.source[self.pos]
                self.pos += 1
        
        if text_buffer:
            tokens.append(Token("TEXT", text_buffer, self.line))
        
        return tokens

    def _read_expression(self) -> Token:
        self.pos += 2
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos:self.pos+2] != "}}":
            self.pos += 1
        value = self.source[start:self.pos].strip()
        self.pos += 2
        return Token("EXPR", value, self.line)

    def _read_statement(self) -> Token:
        self.pos += 2
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos:self.pos+2] != "%}":
            self.pos += 1
        value = self.source[start:self.pos].strip()
        self.pos += 2
        return Token("STMT", value, self.line)

    def _read_comment(self) -> None:
        self.pos += 2
        while self.pos < len(self.source) and self.source[self.pos:self.pos+2] != "#}":
            self.pos += 1
        self.pos += 2


@dataclass
class Node:
    pass


@dataclass
class TextNode(Node):
    text: str


@dataclass
class ExprNode(Node):
    expr: str
    filters: List[str] = field(default_factory=list)


@dataclass
class ForNode(Node):
    var: str
    iterable: str
    body: List[Node] = field(default_factory=list)


@dataclass
class IfNode(Node):
    condition: str
    body: List[Node] = field(default_factory=list)
    else_body: List[Node] = field(default_factory=list)


@dataclass
class BlockNode(Node):
    name: str
    body: List[Node] = field(default_factory=list)


@dataclass
class ExtendsNode(Node):
    parent: str


@dataclass
class IncludeNode(Node):
    template: str


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> List[Node]:
        nodes = []
        while self.pos < len(self.tokens):
            node = self._parse_node()
            if node:
                nodes.append(node)
        return nodes

    def _parse_node(self) -> Optional[Node]:
        if self.pos >= len(self.tokens):
            return None
        
        token = self.tokens[self.pos]
        
        if token.type == "TEXT":
            self.pos += 1
            return TextNode(text=token.value)
        elif token.type == "EXPR":
            self.pos += 1
            parts = token.value.split("|")
            expr = parts[0].strip()
            filters = [f.strip() for f in parts[1:]]
            return ExprNode(expr=expr, filters=filters)
        elif token.type == "STMT":
            return self._parse_statement(token)
        
        self.pos += 1
        return None

    def _parse_statement(self, token: Token) -> Optional[Node]:
        parts = token.value.split(None, 1)
        keyword = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        
        if keyword == "for":
            return self._parse_for(rest)
        elif keyword == "if":
            return self._parse_if(rest)
        elif keyword == "block":
            return self._parse_block(rest)
        elif keyword == "extends":
            self.pos += 1
            return ExtendsNode(parent=rest.strip().strip("'\""))
        elif keyword == "include":
            self.pos += 1
            return IncludeNode(template=rest.strip().strip("'\""))
        elif keyword in ("endif", "endfor", "endblock", "else"):
            return None
        
        self.pos += 1
        return None

    def _parse_for(self, rest: str) -> ForNode:
        match = re.match(r"(\w+)\s+in\s+(.+)", rest)
        if not match:
            raise TemplateError(f"Invalid for syntax: {rest}")
        
        var, iterable = match.groups()
        self.pos += 1
        body = []
        
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type == "STMT" and token.value.strip() == "endfor":
                self.pos += 1
                break
            node = self._parse_node()
            if node:
                body.append(node)
        
        return ForNode(var=var, iterable=iterable.strip(), body=body)

    def _parse_if(self, condition: str) -> IfNode:
        self.pos += 1
        body = []
        else_body = []
        in_else = False
        
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type == "STMT":
                if token.value.strip() == "endif":
                    self.pos += 1
                    break
                elif token.value.strip() == "else":
                    in_else = True
                    self.pos += 1
                    continue
            
            node = self._parse_node()
            if node:
                if in_else:
                    else_body.append(node)
                else:
                    body.append(node)
        
        return IfNode(condition=condition.strip(), body=body, else_body=else_body)

    def _parse_block(self, name: str) -> BlockNode:
        self.pos += 1
        body = []
        
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if token.type == "STMT" and token.value.strip() == "endblock":
                self.pos += 1
                break
            node = self._parse_node()
            if node:
                body.append(node)
        
        return BlockNode(name=name.strip(), body=body)


class Environment:
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self.globals: Dict[str, Any] = {}
        self.filters: Dict[str, Callable] = {
            "upper": lambda x: str(x).upper(),
            "lower": lambda x: str(x).lower(),
            "title": lambda x: str(x).title(),
            "escape": lambda x: html.escape(str(x)),
            "trim": lambda x: str(x).strip(),
            "length": lambda x: len(x),
            "default": lambda x, d="": x if x else d,
            "join": lambda x, sep=",": sep.join(str(i) for i in x),
            "first": lambda x: x[0] if x else None,
            "last": lambda x: x[-1] if x else None,
        }

    def add_template(self, name: str, source: str) -> None:
        self.templates[name] = source

    def add_global(self, name: str, value: Any) -> None:
        self.globals[name] = value

    def add_filter(self, name: str, fn: Callable) -> None:
        self.filters[name] = fn

    def get_template(self, name: str) -> "Template":
        if name not in self.templates:
            raise TemplateError(f"Template not found: {name}")
        return Template(self.templates[name], self)


class Template:
    def __init__(self, source: str, env: Environment = None):
        self.source = source
        self.env = env or Environment()
        self.lexer = Lexer(source)
        self.tokens = self.lexer.tokenize()
        self.parser = Parser(self.tokens)
        self.nodes = self.parser.parse()

    def render(self, context: Dict[str, Any] = None) -> str:
        ctx = {**self.env.globals, **(context or {})}
        return self._render_nodes(self.nodes, ctx)

    def _render_nodes(self, nodes: List[Node], context: Dict[str, Any]) -> str:
        result = []
        blocks = {}
        parent = None
        
        for node in nodes:
            if isinstance(node, ExtendsNode):
                parent = node.parent
            elif isinstance(node, BlockNode):
                blocks[node.name] = node
            else:
                result.append(self._render_node(node, context, blocks))
        
        if parent:
            parent_template = self.env.get_template(parent)
            parent_ctx = {**context, "__blocks__": blocks}
            return parent_template._render_with_blocks(parent_ctx, blocks)
        
        return "".join(result)

    def _render_with_blocks(self, context: Dict[str, Any], child_blocks: Dict[str, BlockNode]) -> str:
        result = []
        for node in self.nodes:
            if isinstance(node, BlockNode):
                if node.name in child_blocks:
                    result.append(self._render_nodes(child_blocks[node.name].body, context))
                else:
                    result.append(self._render_nodes(node.body, context))
            else:
                result.append(self._render_node(node, context, child_blocks))
        return "".join(result)

    def _render_node(self, node: Node, context: Dict[str, Any], blocks: Dict[str, BlockNode] = None) -> str:
        if isinstance(node, TextNode):
            return node.text
        elif isinstance(node, ExprNode):
            return self._eval_expr(node, context)
        elif isinstance(node, ForNode):
            return self._render_for(node, context, blocks)
        elif isinstance(node, IfNode):
            return self._render_if(node, context, blocks)
        elif isinstance(node, IncludeNode):
            return self.env.get_template(node.template).render(context)
        elif isinstance(node, BlockNode):
            return self._render_nodes(node.body, context)
        return ""

    def _eval_expr(self, node: ExprNode, context: Dict[str, Any]) -> str:
        value = self._resolve(node.expr, context)
        for filter_name in node.filters:
            if filter_name in self.env.filters:
                value = self.env.filters[filter_name](value)
        return str(value) if value is not None else ""

    def _resolve(self, expr: str, context: Dict[str, Any]) -> Any:
        parts = expr.split(".")
        value = context.get(parts[0])
        for part in parts[1:]:
            if value is None:
                return None
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
        return value

    def _render_for(self, node: ForNode, context: Dict[str, Any], blocks: Dict[str, BlockNode]) -> str:
        iterable = self._resolve(node.iterable, context)
        if not iterable:
            return ""
        result = []
        items = list(iterable)
        for i, item in enumerate(items):
            loop_ctx = {
                **context,
                node.var: item,
                "loop": {"index": i + 1, "index0": i, "first": i == 0, "last": i == len(items) - 1, "length": len(items)}
            }
            result.append(self._render_nodes(node.body, loop_ctx))
        return "".join(result)

    def _render_if(self, node: IfNode, context: Dict[str, Any], blocks: Dict[str, BlockNode]) -> str:
        value = self._resolve(node.condition, context)
        if value:
            return self._render_nodes(node.body, context)
        return self._render_nodes(node.else_body, context)


class TemplateLoader:
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        self.env = Environment()

    def load(self, name: str) -> Template:
        import os
        path = os.path.join(self.base_path, name)
        with open(path, "r") as f:
            source = f.read()
        self.env.add_template(name, source)
        return self.env.get_template(name)


def example_usage():
    env = Environment()
    env.add_template("base.html", """
<!DOCTYPE html>
<html>
<head><title>{% block title %}Default{% endblock %}</title></head>
<body>{% block content %}{% endblock %}</body>
</html>
""")
    
    env.add_template("page.html", """
{% extends "base.html" %}
{% block title %}{{ title }}{% endblock %}
{% block content %}
<h1>{{ title }}</h1>
<ul>
{% for item in items %}
<li>{{ loop.index }}. {{ item.name|upper }}</li>
{% endfor %}
</ul>
{% if show_footer %}
<footer>Footer content</footer>
{% endif %}
{% endblock %}
""")
    
    template = env.get_template("page.html")
    output = template.render({
        "title": "My Page",
        "items": [{"name": "Apple"}, {"name": "Banana"}, {"name": "Cherry"}],
        "show_footer": True
    })
    print(output)

