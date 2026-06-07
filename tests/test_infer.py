from wh_train.infer.inference import build_messages, extract_user_text
from wh_train.schema import SYSTEM_PROMPT


def test_build_messages_uses_system_prompt():
    msgs = build_messages("出库2个轴承")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "出库2个轴承"}


def test_extract_user_text_from_messages():
    rec = {"messages": [{"role": "user", "content": "出库2个轴承"}]}
    assert extract_user_text(rec) == "出库2个轴承"


def test_extract_user_text_from_input_field():
    assert extract_user_text({"input": "入库5个滤芯"}) == "入库5个滤芯"
