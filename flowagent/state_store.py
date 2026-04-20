"""StateStore: the only data-passing mechanism between workflow states.

Each ``update`` creates an immutable :class:`StateStoreSnapshot` so the full
history of writes is auditable. States read typed context via
:meth:`StateStore.extract`, which projects the store onto the fields defined
by an ``input_schema`` pydantic model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class StateStoreSnapshot:
    """Immutable record of a single ``StateStore.update`` call."""

    timestamp: datetime
    source_state: str
    data_before: Dict[str, Any]
    updates: Dict[str, Any]


class StateStore:
    """Accumulates data across workflow states with snapshot history."""

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = copy.deepcopy(initial_data) if initial_data else {}
        self._history: List[StateStoreSnapshot] = []

    @property
    def data(self) -> Dict[str, Any]:
        """Deep copy of the current store contents."""
        return copy.deepcopy(self._data)

    @property
    def history(self) -> List[StateStoreSnapshot]:
        """Full snapshot history in insertion order."""
        return list(self._history)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a deep-copied value or ``default`` if missing."""
        if key in self._data:
            return copy.deepcopy(self._data[key])
        return default

    def update(self, updates: Dict[str, Any], source_state: str) -> None:
        """Merge ``updates`` into the store and record a snapshot.

        The snapshot captures ``data_before`` and the exact ``updates`` so
        callers can reconstruct the store at any point.
        """
        snapshot = StateStoreSnapshot(
            timestamp=datetime.now(timezone.utc),
            source_state=source_state,
            data_before=copy.deepcopy(self._data),
            updates=copy.deepcopy(updates),
        )
        self._data.update(copy.deepcopy(updates))
        self._history.append(snapshot)

    def extract(self, schema: Type[BaseModel]) -> BaseModel:
        """Build a pydantic model from the fields in ``schema`` that exist.

        Missing optional fields fall through to the schema's defaults; missing
        required fields raise pydantic ``ValidationError``.
        """
        payload = {
            k: copy.deepcopy(self._data[k])
            for k in schema.model_fields
            if k in self._data
        }
        return schema(**payload)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current store for checkpointing."""
        return copy.deepcopy(self._data)
