"""Structured logging helpers for knowledge ingestion."""

import json
import logging
from time import perf_counter
from typing import Callable, TypeVar


Result = TypeVar("Result")


class IngestionLogger:
    """Emit stable JSON events for every ingestion step."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def run_step(
        self,
        document_id: str,
        step: str,
        operation: Callable[[], Result],
    ) -> Result:
        started = perf_counter()
        try:
            result = operation()
        except Exception as error:
            self.emit(
                document_id=document_id,
                step=step,
                elapsed_seconds=perf_counter() - started,
                success=False,
                error_type=type(error).__name__,
            )
            raise

        self.emit(
            document_id=document_id,
            step=step,
            elapsed_seconds=perf_counter() - started,
            success=True,
        )
        return result

    def emit(
        self,
        *,
        document_id: str,
        step: str,
        elapsed_seconds: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "document_id": document_id,
            "elapsed_ms": round(elapsed_seconds * 1_000, 3),
            "event": "knowledge_ingestion_step",
            "step": step,
            "success": success,
        }
        if error_type is not None:
            event["error_type"] = error_type
        self._logger.info(json.dumps(event, sort_keys=True))
