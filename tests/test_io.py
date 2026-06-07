from wh_train.utils.io import read_jsonl, write_jsonl


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"a": 1}, {"b": "中文"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]
