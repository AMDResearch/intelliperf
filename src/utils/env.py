import os
from pathlib import Path

def get_guided_tuning_path():
    if os.environ.get("GT_TUNING"):
        return Path(os.environ["GT_TUNING"]).resolve()
    return (Path(__file__).resolve().parent / "../../external/guided-tuning").resolve()

