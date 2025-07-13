import os
from pathlib import Path

def get_project_root(levels_up=1):
    """
    Return the absolute project root by going 'levels_up' from the current working directory.
    Useful in notebooks or scripts to construct absolute paths.

    Parameters:
        levels_up (int): How many directory levels to go up (default: 1)

    Returns:
        pathlib.Path: Absolute path to the project root
    """
    path = Path.cwd()
    for _ in range(levels_up):
        path = path.parent
    return path.resolve()