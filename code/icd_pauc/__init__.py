"""DeepFuse code for the paper.

Importing the package reads `config.env` at the repository root (if present) into
the environment, without overriding variables that are already set, and puts the
PanDerm code (`$PANDERM_CODE/classification`) on `sys.path`. Scripts therefore run
the same way from `run.sh`, a terminal or an IDE such as PyCharm.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG = os.path.join(_REPO, "config.env")


def _load_config(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            os.environ.setdefault(key, value.strip("'\""))


def _add_panderm_path() -> None:
    code = os.environ.get("PANDERM_CODE")
    if not code:
        return
    path = os.path.join(code, "classification")
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def require_panderm() -> None:
    """Fail with a clear message when the PanDerm code is not available."""
    try:
        import datasets.derm_data  # noqa: F401
        import models.modeling_finetune  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The PanDerm code was not found. Clone https://github.com/SiyuanYan1/PanDerm "
            "and set PANDERM_CODE to that folder in config.env at the repository root "
            f"({_CONFIG}); see README, Quick start."
        ) from exc


_load_config(_CONFIG)
_add_panderm_path()
