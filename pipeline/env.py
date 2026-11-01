"""Load repo-root `.env` without printing values."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    # override=True so a token already in the process env cannot hide a newer .env.
    load_dotenv(ROOT / ".env", override=True)
