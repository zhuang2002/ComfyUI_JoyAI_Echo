"""ComfyUI nodes for JoyAI-Echo: minute-level multi-shot audio-video generation."""

import os
import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
_ECHO_REPO = _NODE_ROOT / "JoyAI-Echo"

for _subpath in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
    _p = str(_ECHO_REPO / _subpath)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .nodes import JoyEcho_ModelLoader, JoyEcho_TextEncode, JoyEcho_Generate, JoyEcho_PromptFormat

NODE_CLASS_MAPPINGS = {
    "JoyEcho_ModelLoader": JoyEcho_ModelLoader,
    "JoyEcho_TextEncode": JoyEcho_TextEncode,
    "JoyEcho_Generate": JoyEcho_Generate,
    "JoyEcho_PromptFormat": JoyEcho_PromptFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyEcho_ModelLoader": "JoyEcho Model Loader",
    "JoyEcho_TextEncode": "JoyEcho Text Encode",
    "JoyEcho_Generate": "JoyEcho Generate (Multi-Shot)",
    "JoyEcho_PromptFormat": "JoyEcho Prompt Format (Helper)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
