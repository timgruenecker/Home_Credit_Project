import os
from pathlib import Path

# returns project root
def get_project_root(levels_up=1):
    path = Path.cwd()
    for _ in range(levels_up):
        path = path.parent
    return path.resolve()