"""Unit tests for :mod:`flowagent.state_store`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from flowagent.state_store import StateStore


class Fields(BaseModel):
    vendor: str | None = None
    amount: float | None = None




def test_empty_init():
    store = StateStore()
    assert store.data == {}
    assert store.history == []


def test_init_with_data_is_deep_copied():
    initial = {"payload": {"a": 1}}
    store = StateStore(initial)
    initial["payload"]["a"] = 999
    assert store.data["payload"]["a"] == 1


def test_update_merges_and_records_snapshot():
    store = StateStore({"x": 1})
    store.update({"y": 2}, source_state="extract")
    assert store.data == {"x": 1, "y": 2}
    assert len(store.history) == 1
    snap = store.history[0]
    assert snap.source_state == "extract"
    assert snap.data_before == {"x": 1}
    assert snap.updates == {"y": 2}


def test_update_later_overwrites_earlier():
    store = StateStore()
    store.update({"k": 1}, source_state="a")
    store.update({"k": 2}, source_state="b")
    assert store.get("k") == 2
    assert len(store.history) == 2


def test_data_is_independent_copy():
    store = StateStore({"k": [1, 2]})
    snapshot = store.data
    snapshot["k"].append(3)
    assert store.get("k") == [1, 2]


def test_get_default():
    store = StateStore()
    assert store.get("missing", "fallback") == "fallback"


def test_extract_projects_onto_schema_fields_only():
    store = StateStore({"vendor": "Acme", "amount": 10.0, "extra": "ignored"})
    extracted = store.extract(Fields)
    assert isinstance(extracted, Fields)
    assert extracted.vendor == "Acme"
    assert extracted.amount == 10.0


class _RequiredMissing(BaseModel):
    needs: str


def test_extract_raises_on_missing_required():
    store = StateStore({"vendor": "ok"})
    with pytest.raises(ValidationError):
        store.extract(_RequiredMissing)


def test_to_dict_round_trip():
    data = {"a": 1, "nested": {"b": 2}}
    store = StateStore(data)
    dumped = store.to_dict()
    assert dumped == data
    dumped["nested"]["b"] = 99
    assert store.get("nested")["b"] == 2


def test_history_is_read_only_view():
    store = StateStore()
    store.update({"a": 1}, source_state="s1")
    hist = store.history
    hist.clear()
    assert len(store.history) == 1
