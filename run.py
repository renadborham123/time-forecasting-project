"""Prepare reproducible model artifacts, then launch ChurnScope AI."""
from pathlib import Path
import subprocess, sys, os

ROOT=Path(__file__).resolve().parent
if not (ROOT/"models/pipeline.joblib").exists():
    subprocess.run([sys.executable,str(ROOT/"scripts/export_models.py")],cwd=ROOT,check=True)
import uvicorn
if __name__=="__main__":
    uvicorn.run("backend.main:app",host="127.0.0.1",port=int(os.getenv("CHURNSCOPE_PORT","8000")),reload=False,access_log=False)
