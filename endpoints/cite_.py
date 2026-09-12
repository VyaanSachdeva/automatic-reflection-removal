"""Citation output for UnReflectAnything."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal


def _get_fallback_citation(format: str) -> str:
    """Return fallback citation when citations file is not available."""
    citations = {
        "bibtex": """@article{unreflectanything2024,
    title={UnReflectAnything: Removing Specular Reflections from RGB Images},
    author={Rota, Alberto and Kiray, Mert and Karaoglu, Mert Asim and Ruhkamp, Patrick and De Momi, Elena and Navab, Nassir and Busam, Benjamin},
    journal={arXiv preprint},
    year={2024}
}""",
        "apa": """Rota, A., Kiray, M., Karaoglu, M. A., Ruhkamp, P., De Momi, E., Navab, N., & Busam, B. (2024). UnReflectAnything: Removing Specular Reflections from RGB Images. arXiv preprint.""",
        "mla": """Rota, Alberto, et al. "UnReflectAnything: Removing Specular Reflections from RGB Images." arXiv preprint, 2024.""",
        "ieee": """A. Rota, M. Kiray, M. A. Karaoglu, P. Ruhkamp, E. De Momi, N. Navab, and B. Busam, "UnReflectAnything: Removing Specular Reflections from RGB Images," arXiv preprint, 2024.""",
        "plain": """Alberto Rota, Mert Kiray, Mert Asim Karaoglu, Patrick Ruhkamp, Elena De Momi, Nassir Navab, and Benjamin Busam. UnReflectAnything: Removing Specular Reflections from RGB Images. arXiv preprint, 2024.""",
    }
    return citations.get(format.lower(), citations["bibtex"])


def cite(format: Literal["bibtex", "apa", "mla", "ieee", "plain"] = "bibtex") -> str:
    """Get the citation for UnReflectAnything in the specified format.

    Args:
        format: Citation format. One of:
            - "bibtex": BibTeX format (default)
            - "apa": APA 7th edition format
            - "mla": MLA 9th edition format
            - "ieee": IEEE format
            - "plain": Plain text format

    Returns:
        Citation string in the requested format.
    """
    try:
        try:
            pkg = importlib.resources.files("unreflectanything")
            citations_path = pkg / "assets" / "citations.txt"
            if hasattr(citations_path, "read_text"):
                citations_text = citations_path.read_text(encoding="utf-8")
            else:
                citations_path = Path(__file__).parent / "assets" / "citations.txt"
                citations_text = citations_path.read_text(encoding="utf-8")
        except Exception:
            citations_path = Path(__file__).parent / "assets" / "citations.txt"
            if citations_path.exists():
                citations_text = citations_path.read_text(encoding="utf-8")
            else:
                citations_path = (
                    Path(__file__).parent.parent / "assets" / "citations.txt"
                )
                if citations_path.exists():
                    citations_text = citations_path.read_text(encoding="utf-8")
                else:
                    return _get_fallback_citation(format)
    except Exception:
        return _get_fallback_citation(format)

    citations = {}
    current_format = None
    current_lines = []

    for line in citations_text.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            if current_format and current_lines:
                citations[current_format] = "\n".join(current_lines).strip()
            current_format = line[1:-1].lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_format and current_lines:
        citations[current_format] = "\n".join(current_lines).strip()

    return citations.get(format.lower(), _get_fallback_citation(format))
