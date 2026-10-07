"""Centralized demo configuration."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
load_dotenv(ROOT / ".env")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
RISK_BANDS = ((0.00, 0.30, "STABLE"), (0.30, 0.60, "WATCHLIST"), (0.60, 0.80, "AT RISK"), (0.80, 1.01, "CRITICAL"))

def risk_label(probability: float) -> str:
    return next(label for lower, upper, label in RISK_BANDS if lower <= probability < upper)
