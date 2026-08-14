import os
from pathlib import Path


def load_env_file(path, override=False):
    """Load the small KEY=VALUE subset used by this project."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(
                "Invalid environment entry at {}:{}".format(
                    env_path, line_number
                )
            )
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
