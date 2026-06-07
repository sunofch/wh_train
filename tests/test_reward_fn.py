from wh_train.reward.reward_fn import compute_reward, reward_func

_GOLD = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
_GOLD_MULTI = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null},{"part_name":"电机","quantity":1,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'


def test_perfect_match_scores_near_one():
    assert compute_reward(_GOLD, _GOLD) > 0.99


def test_parse_failure_scores_zero():
    assert compute_reward("not json at all", _GOLD) == 0.0


def test_empty_array_scores_zero():
    assert compute_reward("[]", _GOLD) == 0.0


def test_partial_match_between_zero_and_perfect():
    # action 对、is_urgent 对、part_name 对，但 quantity 错
    pred = '[{"part_name":"轴承","quantity":99,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    score = compute_reward(pred, _GOLD)
    assert 0.0 < score < compute_reward(_GOLD, _GOLD)


def test_score_ordering_perfect_gt_partial_gt_fail():
    perfect = compute_reward(_GOLD, _GOLD)
    partial = compute_reward(
        '[{"part_name":"轴承","quantity":99,"model":null,"action_required":"入库","is_urgent":false,"description":null}]',
        _GOLD,
    )
    fail = compute_reward("garbage", _GOLD)
    assert perfect > partial > fail


def test_reward_hacking_extra_fields_penalized():
    clean = compute_reward(_GOLD, _GOLD)
    hacked = compute_reward(
        '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null,"priority":"high","location":"A1"}]',
        _GOLD,
    )
    assert hacked < clean


def test_length_mismatch_penalized():
    full = compute_reward(_GOLD_MULTI, _GOLD_MULTI)
    short = compute_reward(_GOLD, _GOLD_MULTI)  # 少一个工单
    assert short < full


def test_greedy_align_rewards_reordered_output():
    reordered = '[{"part_name":"电机","quantity":1,"model":null,"action_required":"出库","is_urgent":false,"description":null},{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    positional = compute_reward(reordered, _GOLD_MULTI, align="positional")
    greedy = compute_reward(reordered, _GOLD_MULTI, align="greedy")
    assert greedy > positional
    assert greedy > 0.99


def test_reward_func_batch_signature():
    # trl 调用约定：completions 为字符串列表，gold 经 kwargs 透传
    scores = reward_func(
        prompts=["x", "y"],
        completions=[_GOLD, "garbage"],
        gold=[_GOLD, _GOLD],
    )
    assert scores[0] > 0.99
    assert scores[1] == 0.0
