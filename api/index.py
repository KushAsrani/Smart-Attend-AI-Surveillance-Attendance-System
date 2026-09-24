import sys
from pathlib import Path

from flask import request

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app


@app.before_request
def normalize_vercel_path():
    if request.path == "/api/index":
        request.environ["PATH_INFO"] = "/"