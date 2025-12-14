from __future__ import annotations

from typing import Dict, Tuple

# Maps friendly aliases to (module, attribute) import paths for built-in graders that can be
# instantiated via the RL dataserver config.
BUILTIN_PYTHON_GRADERS: Dict[str, Tuple[str, str]] = {
    "category_match": ("kolibrify.rl_dataserver.graders", "CategoryMatchGrader"),
    "completion_length_cap": ("kolibrify.rl_dataserver.graders", "CompletionLengthCapGrader"),
    "math-verify": ("kolibrify.rl_dataserver.graders", "MathVerifyGrader"),
    "latex_only": ("kolibrify.rl_dataserver.graders", "LatexOnlyAnswerGrader"),
}

__all__ = ["BUILTIN_PYTHON_GRADERS"]
