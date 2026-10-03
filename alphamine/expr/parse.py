"""Turn formula text into a tree, and a tree back into text or RPN.

The canonical text form is function style - ``ts_mean(volume, 20) / ts_mean(volume, 60)``
- but infix ``+ - * /`` with the usual precedence is accepted too, which makes
the 101 Formulaic Alphas readable when they are transcribed by hand.

``to_rpn`` exists because the reinforcement-learning phase (P2) generates
formulas as reverse-Polish token sequences.
"""

from __future__ import annotations

import re

from .nodes import Const, Node, Op, Var
from .ops import OPERATORS


class ParseError(ValueError):
    """Raised for malformed formula text."""


_TOKEN = re.compile(
    r"""
    (?P<num>\d+\.\d*|\.\d+|\d+)
  | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>[+\-*/])
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<comma>,)
  | (?P<space>\s+)
  | (?P<bad>.)
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == "space":
            continue
        if kind == "bad":
            raise ParseError(f"unexpected character {value!r} in {text!r}")
        tokens.append((kind, value))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], text: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self.text = text

    # -- token helpers ------------------------------------------------------
    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ParseError(f"unexpected end of formula in {self.text!r}")
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> tuple[str, str]:
        tok = self.next()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise ParseError(f"expected {value or kind} but found {tok[1]!r} in {self.text!r}")
        return tok

    # -- grammar ------------------------------------------------------------
    def parse(self) -> Node:
        node = self.expr()
        if self.peek() is not None:
            raise ParseError(f"trailing input {self.peek()[1]!r} in {self.text!r}")
        return node

    def expr(self) -> Node:
        node = self.term()
        while True:
            tok = self.peek()
            if tok and tok[0] == "op" and tok[1] in "+-":
                self.next()
                rhs = self.term()
                node = Op("add" if tok[1] == "+" else "sub", (node, rhs))
            else:
                return node

    def term(self) -> Node:
        node = self.factor()
        while True:
            tok = self.peek()
            if tok and tok[0] == "op" and tok[1] in "*/":
                self.next()
                rhs = self.factor()
                node = Op("mul" if tok[1] == "*" else "div", (node, rhs))
            else:
                return node

    def factor(self) -> Node:
        tok = self.peek()
        if tok and tok[0] == "op" and tok[1] == "-":
            self.next()
            inner = self.factor()
            if isinstance(inner, Const):
                return Const(-inner.value)
            return Op("neg", (inner,))
        if tok and tok[0] == "op" and tok[1] == "+":
            self.next()
            return self.factor()
        return self.atom()

    def atom(self) -> Node:
        kind, value = self.next()
        if kind == "num":
            return Const(float(value))
        if kind == "lparen":
            node = self.expr()
            self.expect("rparen")
            return node
        if kind == "ident":
            nxt = self.peek()
            if nxt is not None and nxt[0] == "lparen":
                self.next()
                args: list[Node] = []
                if self.peek() is not None and self.peek()[0] != "rparen":
                    args.append(self.expr())
                    while self.peek() is not None and self.peek()[0] == "comma":
                        self.next()
                        args.append(self.expr())
                self.expect("rparen")
                return _make_op(value, tuple(args))
            return Var(value)
        raise ParseError(f"unexpected token {value!r} in {self.text!r}")


def _make_op(name: str, args: tuple[Node, ...]) -> Op:
    spec = OPERATORS.get(name)
    if spec is None:
        raise ParseError(f"unknown operator {name!r}")
    if len(args) != spec.arity:
        raise ParseError(f"{name} takes {spec.arity} arguments, got {len(args)}")
    for arg, kind in zip(args, spec.sig, strict=True):
        if kind == "W" and not isinstance(arg, Const):
            raise ParseError(f"{name}: window argument must be a literal, got {arg!r}")
        if kind == "K" and not isinstance(arg, Const):
            raise ParseError(f"{name}: constant argument must be a literal, got {arg!r}")
    return Op(name, args)


def parse(text: str) -> Node:
    """Parse ``text`` into a tree.  Raises :class:`ParseError` if malformed."""

    return _Parser(tokenize(text), text).parse()


def to_rpn(node: Node) -> list[str]:
    """Post-order token sequence, the representation the RL generator emits."""

    if isinstance(node, Var):
        return [node.name]
    if isinstance(node, Const):
        return [f"{node.value:g}"]
    if isinstance(node, Op):
        out: list[str] = []
        for arg in node.args:
            out.extend(to_rpn(arg))
        out.append(node.name)
        return out
    raise TypeError(f"unknown node {node!r}")
