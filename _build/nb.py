"""Tiny helper the notebook builders share: md() and code() cells, then save()."""
from pathlib import Path

import nbformat


def new():
    """Start an empty notebook with the repo's kernel."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def md(nb, text: str):
    nb.cells.append(nbformat.v4.new_markdown_cell(text.strip()))


def code(nb, text: str):
    nb.cells.append(nbformat.v4.new_code_cell(text.strip()))


def save(nb, name: str):
    out = Path(__file__).resolve().parents[1] / name
    nbformat.write(nb, out)
    print("wrote", out)
