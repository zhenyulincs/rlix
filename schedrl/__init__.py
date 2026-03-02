"""SchedRL: Ray-based multi-pipeline GPU time-sharing (ENG-123).

Phase 1 provides the core package skeleton + protocol contracts + Library Mode discovery.
"""

from __future__ import annotations

__all__ = [
    "init",
    "__version__",
    "SchedRLCoordinator",
    "SchedRLFullFinetunePipeline",
    "SchedRLMultiLoraPipeline",
]

__version__ = "0.0.0"

from schedrl.init import init  # noqa: E402
from schedrl.pipeline import SchedRLCoordinator, SchedRLFullFinetunePipeline, SchedRLMultiLoraPipeline  # noqa: E402
