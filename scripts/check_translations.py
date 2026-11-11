"""Check that every translated notebook has exactly the same code as its English original.

English notebooks live in `notebooks/`, translations in `notebooks/<lang>/`.
Only text may differ: markdown, comments, docstrings and messages. Any difference
in code means the copies have drifted and the translation must be updated.

Usage:
    python3 scripts/check_translations.py
"""

import ast
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TRANSLATION_DIRS = ("ru",)


class _StripText(ast.NodeTransformer):
    """Replace every string literal with a placeholder; keep only the expressions inside f-strings."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.Constant(value="<text>")
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        expressions = [self.visit(part.value) for part in node.values if isinstance(part, ast.FormattedValue)]
        return ast.Tuple(elts=expressions, ctx=ast.Load())


def code_fingerprint(path: Path) -> str:
    return ast.dump(_StripText().visit(ast.parse(path.read_text(encoding="utf-8"))))


def find_problems() -> list[str]:
    problems = []
    for original in sorted(NOTEBOOKS_DIR.glob("*.py")):
        for lang in TRANSLATION_DIRS:
            translated = NOTEBOOKS_DIR / lang / original.name
            if not translated.exists():
                problems.append(f"missing translation: {translated.relative_to(NOTEBOOKS_DIR.parent)}")
            elif code_fingerprint(translated) != code_fingerprint(original):
                problems.append(f"code differs from the English original: {translated.relative_to(NOTEBOOKS_DIR.parent)}")
    return problems


def main() -> int:
    problems = find_problems()
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print("OK  all translations have the same code as the English notebooks")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
