from pathlib import Path

def get_guided_tuning_path():
    return (Path(__file__).resolve().parent / "../../external/guided-tuning").resolve()

