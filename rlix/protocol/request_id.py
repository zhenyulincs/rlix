from __future__ import annotations

REQUEST_ID_DELIMITER = ":"


def validate_pipeline_id(pipeline_id: str) -> None:
    if not isinstance(pipeline_id, str):
        raise ValueError(f"pipeline_id must be str, got {type(pipeline_id).__name__}")
    if pipeline_id == "":
        raise ValueError("pipeline_id must be non-empty")
    if REQUEST_ID_DELIMITER in pipeline_id:
        raise ValueError(f"pipeline_id must not contain {REQUEST_ID_DELIMITER!r}: {pipeline_id!r}")
