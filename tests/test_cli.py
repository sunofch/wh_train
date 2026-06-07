from wh_train.cli import build_parser


def test_parser_has_all_subcommands():
    parser = build_parser()
    sub = parser.parse_args(["eval", "--pred", "p.jsonl", "--gold", "g.jsonl"])
    assert sub.command == "eval"
    assert sub.pred == "p.jsonl"


def test_parser_gen_data_defaults():
    parser = build_parser()
    args = parser.parse_args(["gen-data"])
    assert args.command == "gen-data"
    assert args.output_dir == "data"


def test_parser_infer_text():
    parser = build_parser()
    args = parser.parse_args(["infer", "--text", "出库2个轴承"])
    assert args.command == "infer"
    assert args.text == "出库2个轴承"
