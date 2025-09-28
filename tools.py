# tools.py
from duckduckgo_search import ddg
import ast
import operator as op
import math
from typing import Any

# --- Web search (DuckDuckGo)
def web_search(query: str, max_results: int = 5) -> str:
    results = ddg(query, max_results=max_results)
    if not results:
        return "No web results found."
    out = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("body") or ""
        href = r.get("href") or ""
        out.append(f"- {title}\n  {body}\n  ({href})")
    return "\n".join(out)

# --- Safe calculator (supports numbers and + - * / ** parentheses)
# Based on AST walking
SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: lambda x: x
}

def safe_eval(expr: str) -> Any:
    """
    Evaluate a math expression safely.
    Supports: + - * / ** parentheses and numbers.
    """
    def _eval(node):
        if isinstance(node, ast.Num):  # < Py3.8
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Operator {op_type} not allowed.")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unary operator {op_type} not allowed.")
        raise ValueError(f"Unsupported expression: {node}")

    node = ast.parse(expr, mode="eval").body
    return _eval(node)
