"""mjlab backend — myouser course UI package.

Registers only the myoUserUniversal task; all other myosuite tasks are not
included in this standalone distribution.

Entry point (pyproject.toml):
    [project.entry-points."mjlab.tasks"]
    myosuite = "myosuite.envs.myo.backends.mjlab"
"""

from __future__ import annotations

REGISTERED_TASKS: dict[str, type] = {}

try:
    from myosuite.envs.myo.backends.mjlab.register_mjlab_myouser_tasks import (
        register_mjlab_myouser_task,
    )
except Exception:
    pass
