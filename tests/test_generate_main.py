import json
from unittest.mock import MagicMock, patch

import pytest

from generate_dataset import deduplicate, generate_batch


def _make_mock_response(samples: list[dict]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"samples": samples})
    return mock_resp


def test_generate_batch_returns_valid_samples():
    valid_sample = {
        "input": "出库2个轴承6208",
        "output": '[{"part_name":"轴承","quantity":2,"model":"6208","action_required":"出库","is_urgent":false,"description":null}]',
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response([valid_sample])

    with patch("generate_dataset._get_client", return_value=mock_client):
        results = generate_batch("显式标准指令", "明确包含出库/入库", "kb context", n=1)

    assert len(results) == 1
    assert results[0]["input"] == "出库2个轴承6208"


def test_generate_batch_filters_invalid():
    bad_sample = {"input": "x", "output": "not json"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response([bad_sample])

    with patch("generate_dataset._get_client", return_value=mock_client):
        results = generate_batch("显式标准指令", "desc", "kb", n=1)

    assert len(results) == 0


def test_generate_batch_handles_api_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("network error")

    with patch("generate_dataset._get_client", return_value=mock_client):
        with pytest.raises(Exception):
            generate_batch("显式标准指令", "desc", "kb", n=1)


def test_generate_batch_handles_malformed_json():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "not json at all"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("generate_dataset._get_client", return_value=mock_client):
        results = generate_batch("显式标准指令", "desc", "kb", n=1)

    assert results == []


def test_deduplicate_removes_same_input():
    samples = [
        ("类A", {"input": "出库2个轴承", "output": "[]"}),
        ("类A", {"input": "出库2个轴承", "output": "[]"}),  # 重复
        ("类A", {"input": "出库3个轴承", "output": "[]"}),
    ]
    result = deduplicate(samples)
    assert len(result) == 2


def test_deduplicate_keeps_order():
    samples = [
        ("类A", {"input": "A", "output": "[]"}),
        ("类B", {"input": "B", "output": "[]"}),
        ("类A", {"input": "A", "output": "[]"}),  # 重复，应被去掉
    ]
    result = deduplicate(samples)
    assert [s["input"] for _, s in result] == ["A", "B"]


def test_deduplicate_empty():
    assert deduplicate([]) == []
