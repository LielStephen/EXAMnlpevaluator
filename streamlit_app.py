from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parent / "nlpfinal" / "exam_evaluator"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app  # noqa: F401
