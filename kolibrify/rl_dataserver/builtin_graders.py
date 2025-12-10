from __future__ import annotations

from typing import Dict, Tuple

# Maps friendly aliases to (module, attribute) import paths for built-in graders that can be
# instantiated via the RL dataserver config.
BUILTIN_PYTHON_GRADERS: Dict[str, Tuple[str, str]] = {
    "category_match": ("kolibrify.rl_dataserver.graders", "CategoryMatchGrader"),
}

__all__ = ["BUILTIN_PYTHON_GRADERS"]
