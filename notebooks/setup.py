import os
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / 'src'))
os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)