"""pytest 统一测试环境。"""

from pathlib import Path
import sys


APP_PATH = str(Path(__file__).resolve().parents[1] / "app")
if APP_PATH not in sys.path:
    sys.path.append(APP_PATH)
