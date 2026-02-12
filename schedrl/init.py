from __future__ import annotations

import os

from schedrl.client.client import connect


def init(*, create_if_missing: bool = True):
    rank_raw = os.environ.get("RANK", "0")
    try:
        rank = int(rank_raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid RANK={rank_raw!r}, expected int") from e

    if rank != 0:
        return None
    return connect(create_if_missing=create_if_missing)

