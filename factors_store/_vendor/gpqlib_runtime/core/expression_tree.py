from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re


_QLIB_FIELD_MAP = {
    "close": "$close",
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "volume": "$volume",
    "turn": "$turn",
    "vwap": "$vwap",
}
_QLIB_FIELD_MAP_INV = {v: k for k, v in _QLIB_FIELD_MAP.items()}
_QLIB_FIELD_ALIASES = {
    "$return": "returns",
    "$returns": "returns",
}
_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
    |(?P<NUMBER>(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)
    |(?P<FIELD>\$[A-Za-z_][A-Za-z0-9_]*)
    |(?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<CMP>>=|<=|==|!=|>|<)
    |(?P<OP>[+\-*/])
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<COMMA>,)
    """,
    re.VERBOSE,
)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported scalar {value!r} for qlib serialization")


def _qlib_arg(arg: Any) -> str:
    if isinstance(arg, ExprNode):
        return arg.to_qlib_expr()
    return _format_scalar(arg)


@dataclass
class ExprNode:
    op: str
    args: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # 生成稳定的嵌套结构，便于持久化或跨进程传输。
        def _ser(v: Any) -> Any:
            if isinstance(v, ExprNode):
                return v.to_dict()
            return v

        return {"op": self.op, "args": [_ser(a) for a in self.args]}

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExprNode":
        # 从序列化结果重建嵌套表达式树。
        def _de(v: Any) -> Any:
            if isinstance(v, dict) and "op" in v and "args" in v:
                return ExprNode.from_dict(v)
            return v

        return ExprNode(op=payload["op"], args=[_de(a) for a in payload.get("args", [])])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ExprNode":
        return ExprNode.from_dict(json.loads(s))

    def to_expr(self) -> str:
        # 生成人类可读表达式，便于日志和 CSV 展示。
        def _fmt(v: Any) -> str:
            if isinstance(v, ExprNode):
                return v.to_expr()
            if isinstance(v, str):
                return v
            return repr(v)

        return f"{self.op}({', '.join(_fmt(a) for a in self.args)})"

    def to_qlib_expr(self) -> str:
        def _child(idx: int) -> str:
            return _qlib_arg(self.args[idx])

        if self.op == "id":
            field = str(self.args[0])
            if field in _QLIB_FIELD_MAP:
                return _QLIB_FIELD_MAP[field]
            if field == "returns":
                return "($close / Ref($close, 1) - 1)"
            raise ValueError(f"Field {field!r} is not supported by qlib backend")

        if self.op == "ts_mean":
            if len(self.args) < 2 or self.args[2] is not None:
                raise ValueError("qlib backend only supports ts_mean without half_life")
            return f"Mean({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_std":
            if len(self.args) < 2 or self.args[2] is not None:
                raise ValueError("qlib backend only supports ts_std without half_life")
            return f"Std({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_var":
            if len(self.args) < 2 or self.args[2] is not None:
                raise ValueError("qlib backend only supports ts_var without half_life")
            return f"Var({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_ir":
            raise ValueError("qlib backend does not support ts_ir; use python backend for this operator")
        if self.op == "ts_skew":
            return f"Skew({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_kurt":
            return f"Kurt({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_max":
            return f"Max({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_min":
            return f"Min({_child(0)}, {int(self.args[1])})"
        if self.op in {"ts_max_diff", "ts_min_diff", "ts_min_max_diff", "ts_pct_change"}:
            raise ValueError(f"qlib backend does not support {self.op}; use python backend for this operator")
        if self.op == "ts_sum":
            return f"Sum({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_count":
            return f"Count({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_rank":
            return f"Rank({_child(0)}, {int(self.args[1])})"
        if self.op == "rank":
            return f"Rank({_child(0)})"
        if self.op == "ts_quantile":
            return f"Quantile({_child(0)}, {int(self.args[1])}, {_qlib_arg(self.args[2])})"
        if self.op == "ts_med":
            return f"Med({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_mad":
            return f"Mad({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_argmax":
            return f"IdxMax({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_argmin":
            return f"IdxMin({_child(0)}, {int(self.args[1])})"
        if self.op == "ts_corr":
            if len(self.args) < 3 or self.args[3] is not None:
                raise ValueError("qlib backend only supports ts_corr without half_life")
            return f"Corr({_child(0)}, {_child(1)}, {int(self.args[2])})"
        if self.op == "ts_cov":
            if len(self.args) < 3 or self.args[3] is not None:
                raise ValueError("qlib backend only supports ts_cov without half_life")
            return f"Cov({_child(0)}, {_child(1)}, {int(self.args[2])})"
        if self.op == "log":
            return f"Log({_child(0)})"
        if self.op == "abs":
            return f"Abs({_child(0)})"
        if self.op == "sign":
            return f"Sign({_child(0)})"
        if self.op == "not":
            return f"Not({_child(0)})"
        if self.op == "exp":
            return f"Exp({_child(0)})"
        if self.op == "power":
            return f"Pow({_child(0)}, {_qlib_arg(self.args[1])})"
        if self.op == "inv":
            return f"(1 / {_child(0)})"
        if self.op in {
            "zscore",
            "sma",
            "macd",
            "slog1p",
            "ts_linear_decay_mean",
            "ts_exp_weighted_mean_lagged",
            "ts_turnover_ref_price",
            "resiliency",
            "similar_path_reversal",
            "similar_path_low_volatility",
            "turnover_survival_loss",
            "prospect_pos_component",
            "prospect_neg_component",
            "cs_mean",
            "cs_sum",
            "cs_std",
            "cs_skew",
            "cs_zscore",
            "cs_multi_reg_resid",
            "csad_daily",
            "cs_corr_mean",
            "cssd_daily",
            "rolling_herding_beta",
            "rolling_peer_csad_ratio",
            "rolling_regression_beta_sum",
            "ts_sorted_mean_spread",
            "rolling_cs_spearman_mean",
            "ts_sorted_weighted_spread",
            "group_mean",
            "group_sum",
            "group_rank",
            "group_bucket",
            "group_combine",
            "neighbor_mean",
            "cs_reg_resid",
            "group_day_sum",
            "group_day_bar_sum",
            "group_day_mean",
            "group_day_last",
            "group_day_bucket_entropy",
            "group_day_kurt",
            "group_day_skew",
            "group_day_std",
            "group_day_count",
            "group_day_tripower_iv",
            "group_day_bipower_iv",
            "group_day_vwap",
            "rolling_score_top_volume_vwap_ratio",
            "group_day_top_prod",
            "group_day_top_mean",
            "group_day_cvar",
            "group_day_sorted_skew",
            "group_day_weighted_skew",
            "day_trend_ratio",
            "intraday_maxdrawdown",
            "upper_shadow",
            "same_bar_rolling_zscore",
            "same_bar_rolling_std",
            "same_bar_rolling_mean",
            "intraday_peak_signal",
            "intraday_ridge_signal",
            "intraday_broadcast",
            "month_broadcast",
            "intraday_downsample_last",
            "intraday_downsample_sum",
            "group_day_delay",
            "group_day_log_return",
            "intraday_jump_strength",
            "intraday_jump_count",
            "intraday_jump_return",
            "student_t_cdf",
            "group_day_corr",
            "group_month_sum",
            "group_month_mean",
            "group_month_regression_coef",
            "group_month_last",
            "group_ex_self_weighted_mean",
            "group_neutral",
        }:
            raise ValueError(f"qlib backend does not support {self.op}; use python backend for this operator")
        if self.op == "greater":
            return f"Greater({_child(0)}, {_child(1)})"
        if self.op == "less":
            return f"Less({_child(0)}, {_child(1)})"
        if self.op == "gt":
            return f"Gt({_child(0)}, {_child(1)})"
        if self.op == "ge":
            return f"Ge({_child(0)}, {_child(1)})"
        if self.op == "lt":
            return f"Lt({_child(0)}, {_child(1)})"
        if self.op == "le":
            return f"Le({_child(0)}, {_child(1)})"
        if self.op == "eq":
            return f"Eq({_child(0)}, {_child(1)})"
        if self.op == "ne":
            return f"Ne({_child(0)}, {_child(1)})"
        if self.op == "and":
            return f"And({_child(0)}, {_child(1)})"
        if self.op == "or":
            return f"Or({_child(0)}, {_child(1)})"
        if self.op == "ema":
            return f"EMA({_child(0)}, {int(self.args[1])})"
        if self.op == "wma":
            return f"WMA({_child(0)}, {int(self.args[1])})"
        if self.op == "delta":
            return f"Delta({_child(0)}, {int(self.args[1])})"
        if self.op in {"delay", "ref"}:
            return f"Ref({_child(0)}, {int(self.args[1])})"
        if self.op == "slope":
            return f"Slope({_child(0)}, {int(self.args[1])})"
        if self.op == "rsquare":
            return f"Rsquare({_child(0)}, {int(self.args[1])})"
        if self.op == "resi":
            return f"Resi({_child(0)}, {int(self.args[1])})"
        if self.op == "add":
            return f"({_child(0)} + {_child(1)})"
        if self.op == "sub":
            return f"({_child(0)} - {_child(1)})"
        if self.op == "mul":
            return f"({_child(0)} * {_child(1)})"
        if self.op == "div":
            return f"({_child(0)} / {_child(1)})"
        if self.op == "if_then_else":
            return f"If({_child(0)}, {_child(1)}, {_child(2)})"

        raise ValueError(f"Operator {self.op!r} is not supported by qlib backend")


FEATURE_LEAVES = ["close", "open", "high", "low", "volume", "vwap", "returns"]
WINDOW_CHOICES = [5, 10, 20, 30, 60]


def _coerce_window(value: Any) -> int:
    iv = int(float(value))
    return iv


def _coerce_number(value: str) -> int | float:
    fv = float(value)
    if fv.is_integer():
        return int(fv)
    return fv


class _QlibExprParser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            m = _TOKEN_RE.match(text, pos)
            if m is None:
                raise ValueError(f"Unexpected token near: {text[pos:pos+20]!r}")
            kind = m.lastgroup
            raw = m.group(0)
            pos = m.end()
            if kind == "WS":
                continue
            if kind is None:
                raise ValueError(f"Failed to tokenize near: {raw!r}")
            tokens.append((kind, raw))
        return tokens

    def _peek(self) -> tuple[str, str] | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _match(self, *kinds: str) -> tuple[str, str] | None:
        tok = self._peek()
        if tok is None or tok[0] not in kinds:
            return None
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        tok = self._match(kind)
        if tok is None:
            got = self._peek()
            raise ValueError(f"Expected {kind}, got {got!r}")
        return tok

    def parse(self) -> ExprNode:
        node = self._parse_expr()
        if self._peek() is not None:
            raise ValueError(f"Unexpected trailing token: {self._peek()!r}")
        if not isinstance(node, ExprNode):
            raise ValueError("Seed expression cannot be a scalar constant only")
        return node

    def _parse_expr(self) -> Any:
        node = self._parse_compare()
        return node

    def _parse_compare(self) -> Any:
        node = self._parse_term()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "CMP":
                break
            self.pos += 1
            rhs = self._parse_term()
            node = ExprNode(_map_infix_cmp(tok[1]), [node, rhs])
        return node

    def _parse_term(self) -> Any:
        node = self._parse_add_sub()
        return node

    def _parse_add_sub(self) -> Any:
        node = self._parse_mul_div()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in {"+", "-"}:
                break
            self.pos += 1
            rhs = self._parse_mul_div()
            node = ExprNode("add" if tok[1] == "+" else "sub", [node, rhs])
        return node

    def _parse_mul_div(self) -> Any:
        node = self._parse_factor()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in {"*", "/"}:
                break
            self.pos += 1
            rhs = self._parse_factor()
            node = ExprNode("mul" if tok[1] == "*" else "div", [node, rhs])
        return node

    def _parse_factor(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok[0] == "OP" and tok[1] in {"+", "-"}:
            self.pos += 1
            inner = self._parse_factor()
            if tok[1] == "+":
                return inner
            return ExprNode("mul", [-1.0, inner])

        if tok[0] == "NUMBER":
            self.pos += 1
            return _coerce_number(tok[1])

        if tok[0] == "FIELD":
            self.pos += 1
            field = _QLIB_FIELD_MAP_INV.get(tok[1])
            if field is None:
                field = _QLIB_FIELD_ALIASES.get(tok[1])
            if field is None:
                raise ValueError(f"Unsupported qlib field: {tok[1]!r}")
            return ExprNode("id", [field])

        if tok[0] == "LPAREN":
            self.pos += 1
            node = self._parse_expr()
            self._expect("RPAREN")
            return node

        if tok[0] == "NAME":
            return self._parse_call()

        raise ValueError(f"Unexpected token: {tok!r}")

    def _parse_call(self) -> ExprNode:
        name = self._expect("NAME")[1]
        self._expect("LPAREN")
        args: list[Any] = []
        if self._peek() is not None and self._peek()[0] != "RPAREN":
            while True:
                args.append(self._parse_expr())
                if self._match("COMMA") is None:
                    break
        self._expect("RPAREN")
        return _map_qlib_call(name, args)


def _map_qlib_call(name: str, args: list[Any]) -> ExprNode:
    key = name.strip()
    if key == "Ref":
        return ExprNode("delay", [args[0], _coerce_window(args[1])])
    if key in {"Mean", "TsMean"}:
        return ExprNode("ts_mean", [args[0], _coerce_window(args[1]), None])
    if key in {"Std", "TsStd"}:
        return ExprNode("ts_std", [args[0], _coerce_window(args[1]), None])
    if key in {"Var", "TsVar"}:
        return ExprNode("ts_var", [args[0], _coerce_window(args[1]), None])
    if key == "TsIr":
        return ExprNode("ts_ir", [args[0], _coerce_window(args[1]), None])
    if key in {"Skew", "TsSkew"}:
        return ExprNode("ts_skew", [args[0], _coerce_window(args[1])])
    if key in {"Kurt", "TsKurt"}:
        return ExprNode("ts_kurt", [args[0], _coerce_window(args[1])])
    if key in {"Max", "TsMax"}:
        return ExprNode("ts_max", [args[0], _coerce_window(args[1])])
    if key in {"Min", "TsMin"}:
        return ExprNode("ts_min", [args[0], _coerce_window(args[1])])
    if key == "TsMaxDiff":
        return ExprNode("ts_max_diff", [args[0], _coerce_window(args[1])])
    if key == "TsMinDiff":
        return ExprNode("ts_min_diff", [args[0], _coerce_window(args[1])])
    if key == "TsMinMaxDiff":
        return ExprNode("ts_min_max_diff", [args[0], _coerce_window(args[1])])
    if key in {"Sum", "TsSum"}:
        return ExprNode("ts_sum", [args[0], _coerce_window(args[1])])
    if key in {"Count", "TsCount"}:
        return ExprNode("ts_count", [args[0], _coerce_window(args[1])])
    if key in {"Rank", "TsRank"}:
        if len(args) == 1:
            return ExprNode("rank", [args[0]])
        return ExprNode("ts_rank", [args[0], _coerce_window(args[1])])
    if key in {"Quantile", "TsQuantile"}:
        return ExprNode("ts_quantile", [args[0], _coerce_window(args[1]), float(args[2])])
    if key in {"Med", "TsMed"}:
        return ExprNode("ts_med", [args[0], _coerce_window(args[1])])
    if key in {"Mad", "TsMad"}:
        return ExprNode("ts_mad", [args[0], _coerce_window(args[1])])
    if key == "IdxMax":
        return ExprNode("ts_argmax", [args[0], _coerce_window(args[1])])
    if key == "IdxMin":
        return ExprNode("ts_argmin", [args[0], _coerce_window(args[1])])
    if key == "Corr":
        return ExprNode("ts_corr", [args[0], args[1], _coerce_window(args[2]), None])
    if key == "Cov":
        return ExprNode("ts_cov", [args[0], args[1], _coerce_window(args[2]), None])
    if key == "Log":
        return ExprNode("log", [args[0]])
    if key == "Abs":
        return ExprNode("abs", [args[0]])
    if key == "Sign":
        return ExprNode("sign", [args[0]])
    if key == "Not":
        return ExprNode("not", [args[0]])
    if key == "Exp":
        return ExprNode("exp", [args[0]])
    if key == "Pow":
        return ExprNode("power", [args[0], args[1]])
    if key == "Inv":
        return ExprNode("inv", [args[0]])
    if key == "SLog1p":
        return ExprNode("slog1p", [args[0]])
    if key == "ZScore":
        return ExprNode("zscore", [args[0]])
    if key == "EMA":
        return ExprNode("ema", [args[0], _coerce_window(args[1])])
    if key == "SMA":
        return ExprNode("sma", [args[0], _coerce_window(args[1]), _coerce_window(args[2])])
    if key == "TsLinearDecayMean":
        return ExprNode("ts_linear_decay_mean", [args[0], _coerce_window(args[1])])
    if key == "TsExpWeightedMeanLagged":
        return ExprNode("ts_exp_weighted_mean_lagged", [args[0], _coerce_window(args[1])])
    if key == "TsTurnoverRefPrice":
        return ExprNode("ts_turnover_ref_price", [args[0], args[1], _coerce_window(args[2])])
    if key == "Resiliency":
        return ExprNode("resiliency", [args[0], _coerce_window(args[1]), _coerce_window(args[2]), float(args[3])])
    if key == "SimilarPathReversal":
        return ExprNode("similar_path_reversal", [args[0], args[1], _coerce_window(args[2]), _coerce_window(args[3]), float(args[4]), _coerce_window(args[5]), float(args[6])])
    if key == "SimilarPathLowVolatility":
        return ExprNode("similar_path_low_volatility", [args[0], _coerce_window(args[1]), _coerce_window(args[2]), float(args[3]), _coerce_window(args[4]), _coerce_window(args[5]), _coerce_window(args[6]), float(args[7])])
    if key == "TurnoverSurvivalLoss":
        return ExprNode("turnover_survival_loss", [args[0], args[1], args[2], _coerce_window(args[3]), float(args[4])])
    if key == "ProspectPosComponent":
        return ExprNode("prospect_pos_component", [args[0], _coerce_window(args[1]), float(args[2]), float(args[3])])
    if key == "ProspectNegComponent":
        return ExprNode("prospect_neg_component", [args[0], _coerce_window(args[1]), float(args[2]), float(args[3]), float(args[4])])
    if key == "WMA":
        return ExprNode("wma", [args[0], _coerce_window(args[1])])
    if key == "MACD":
        return ExprNode("macd", [args[0], _coerce_window(args[1]), _coerce_window(args[2])])
    if key == "Delta":
        return ExprNode("delta", [args[0], _coerce_window(args[1])])
    if key == "TsPctChange":
        return ExprNode("ts_pct_change", [args[0], _coerce_window(args[1])])
    if key == "Slope":
        return ExprNode("slope", [args[0], _coerce_window(args[1])])
    if key == "Rsquare":
        return ExprNode("rsquare", [args[0], _coerce_window(args[1])])
    if key == "Resi":
        return ExprNode("resi", [args[0], _coerce_window(args[1])])
    if key in {"CSMean", "CsMean"}:
        return ExprNode("cs_mean", [args[0]])
    if key in {"CSSum", "CsSum"}:
        return ExprNode("cs_sum", [args[0]])
    if key in {"CSStd", "CsStd"}:
        return ExprNode("cs_std", [args[0], *args[1:]])
    if key in {"CSSkew", "CsSkew"}:
        return ExprNode("cs_skew", [args[0], *args[1:]])
    if key in {"CSZScore", "CsZScore", "CSZscore", "CsZscore"}:
        return ExprNode("cs_zscore", [args[0]])
    if key == "CSMultiRegResid":
        return ExprNode("cs_multi_reg_resid", list(args))
    if key == "CSADDaily":
        return ExprNode("csad_daily", [args[0]])
    if key == "CSCorrMean":
        return ExprNode("cs_corr_mean", [args[0], args[1], *args[2:]])
    if key == "CSSDDaily":
        return ExprNode("cssd_daily", [args[0], args[1], *args[2:]])
    if key == "RollingHerdingBeta":
        return ExprNode("rolling_herding_beta", [args[0], args[1], *args[2:]])
    if key == "RollingPeerCSADRatio":
        return ExprNode("rolling_peer_csad_ratio", [args[0], _coerce_window(args[1]), _coerce_window(args[2]), _coerce_window(args[3]), *args[4:]])
    if key == "RollingRegressionBetaSum":
        return ExprNode("rolling_regression_beta_sum", [args[0], args[1], args[2], *args[3:]])
    if key == "RollingCSSpearmanMean":
        return ExprNode("rolling_cs_spearman_mean", [args[0], _coerce_window(args[1]), *args[2:]])
    if key == "TsSortedMeanSpread":
        return ExprNode("ts_sorted_mean_spread", [args[0], args[1], _coerce_window(args[2]), float(args[3])])
    if key == "TsSortedWeightedSpread":
        return ExprNode("ts_sorted_weighted_spread", [args[0], args[1], _coerce_window(args[2]), float(args[3])])
    if key == "GroupMean":
        return ExprNode("group_mean", [args[0], args[1], *args[2:]])
    if key == "GroupSum":
        return ExprNode("group_sum", [args[0], args[1], *args[2:]])
    if key == "GroupRank":
        return ExprNode("group_rank", [args[0], args[1], *args[2:]])
    if key == "GroupBucket":
        return ExprNode("group_bucket", [args[0], args[1], *args[2:]])
    if key == "GroupCombine":
        return ExprNode("group_combine", [args[0], args[1], *args[2:]])
    if key == "NeighborMean":
        return ExprNode("neighbor_mean", [args[0], *args[1:]])
    if key == "CSRegResid":
        return ExprNode("cs_reg_resid", [args[0], args[1], *args[2:]])
    if key == "GroupDaySum":
        return ExprNode("group_day_sum", [args[0], *args[1:]])
    if key == "GroupDayBarSum":
        return ExprNode("group_day_bar_sum", [args[0], *args[1:]])
    if key == "GroupDayMean":
        return ExprNode("group_day_mean", [args[0], *args[1:]])
    if key == "GroupDayLast":
        return ExprNode("group_day_last", [args[0], *args[1:]])
    if key == "GroupDayBucketEntropy":
        return ExprNode("group_day_bucket_entropy", [args[0], *args[1:]])
    if key == "GroupDayKurt":
        return ExprNode("group_day_kurt", [args[0], *args[1:]])
    if key == "GroupDaySkew":
        return ExprNode("group_day_skew", [args[0], *args[1:]])
    if key == "GroupDayStd":
        return ExprNode("group_day_std", [args[0], *args[1:]])
    if key == "GroupDayCount":
        return ExprNode("group_day_count", [args[0], *args[1:]])
    if key == "GroupDayTripowerIV":
        return ExprNode("group_day_tripower_iv", [args[0], *args[1:]])
    if key == "GroupDayBipowerIV":
        return ExprNode("group_day_bipower_iv", [args[0], *args[1:]])
    if key == "GroupDayVWAP":
        return ExprNode("group_day_vwap", [args[0], args[1], *args[2:]])
    if key == "RollingScoreTopVolumeVWAPRatio":
        return ExprNode("rolling_score_top_volume_vwap_ratio", [args[0], args[1], args[2], _coerce_window(args[3]), float(args[4]), *args[5:]])
    if key == "GroupDayTopProd":
        return ExprNode("group_day_top_prod", [args[0], float(args[1]), *args[2:]])
    if key == "GroupDayTopMean":
        return ExprNode("group_day_top_mean", [args[0], args[1], _coerce_window(args[2]), *args[3:]])
    if key == "GroupDayCVar":
        return ExprNode("group_day_cvar", [args[0], float(args[1]), *args[2:]])
    if key == "GroupDaySortedSkew":
        return ExprNode("group_day_sorted_skew", [args[0], args[1], float(args[2]), *args[3:]])
    if key == "GroupDayWeightedSkew":
        return ExprNode("group_day_weighted_skew", [args[0], args[1], *args[2:]])
    if key == "DayTrendRatio":
        return ExprNode("day_trend_ratio", [args[0], *args[1:]])
    if key in {"IntradayMaxdrawdown", "IntradayMaxDrawdown", "IntradayMaxDrawDown"}:
        return ExprNode("intraday_maxdrawdown", [args[0], *args[1:]])
    if key == "UpperShadow":
        return ExprNode("upper_shadow", [args[0], args[1], args[2], args[3]])
    if key == "SameBarRollingZScore":
        return ExprNode("same_bar_rolling_zscore", [args[0], *args[1:]])
    if key == "SameBarRollingStd":
        return ExprNode("same_bar_rolling_std", [args[0], *args[1:]])
    if key == "SameBarRollingMean":
        return ExprNode("same_bar_rolling_mean", [args[0], *args[1:]])
    if key == "IntradayPeakSignal":
        return ExprNode("intraday_peak_signal", [args[0]])
    if key == "IntradayRidgeSignal":
        return ExprNode("intraday_ridge_signal", [args[0]])
    if key == "IntradayBroadcast":
        return ExprNode("intraday_broadcast", [args[0], args[1]])
    if key == "MonthBroadcast":
        return ExprNode("month_broadcast", [args[0], args[1]])
    if key == "IntradayDownsampleLast":
        return ExprNode("intraday_downsample_last", [args[0], *args[1:]])
    if key == "IntradayDownsampleSum":
        return ExprNode("intraday_downsample_sum", [args[0], *args[1:]])
    if key == "GroupDayDelay":
        return ExprNode("group_day_delay", [args[0], *args[1:]])
    if key == "GroupDayLogReturn":
        return ExprNode("group_day_log_return", [args[0], *args[1:]])
    if key == "IntradayJumpStrength":
        return ExprNode("intraday_jump_strength", [args[0], *args[1:]])
    if key == "IntradayJumpCount":
        return ExprNode("intraday_jump_count", [args[0], *args[1:]])
    if key == "IntradayJumpReturn":
        return ExprNode("intraday_jump_return", [args[0], args[1], *args[2:]])
    if key == "StudentTCDF":
        return ExprNode("student_t_cdf", [args[0], *args[1:]])
    if key == "GroupDayCorr":
        return ExprNode("group_day_corr", [args[0], args[1], *args[2:]])
    if key == "GroupMonthSum":
        return ExprNode("group_month_sum", [args[0], *args[1:]])
    if key == "GroupMonthMean":
        return ExprNode("group_month_mean", [args[0], *args[1:]])
    if key == "GroupMonthRegressionCoef":
        return ExprNode("group_month_regression_coef", list(args))
    if key == "GroupMonthLast":
        return ExprNode("group_month_last", [args[0], *args[1:]])
    if key == "GroupExSelfWeightedMean":
        return ExprNode("group_ex_self_weighted_mean", [args[0], args[1], args[2], *args[3:]])
    if key == "GroupNeutral":
        return ExprNode("group_neutral", [args[0], args[1]])
    if key == "Greater":
        return ExprNode("greater", [args[0], args[1]])
    if key == "Less":
        return ExprNode("less", [args[0], args[1]])
    if key == "Gt":
        return ExprNode("gt", [args[0], args[1]])
    if key == "Ge":
        return ExprNode("ge", [args[0], args[1]])
    if key == "Lt":
        return ExprNode("lt", [args[0], args[1]])
    if key == "Le":
        return ExprNode("le", [args[0], args[1]])
    if key == "Eq":
        return ExprNode("eq", [args[0], args[1]])
    if key == "Ne":
        return ExprNode("ne", [args[0], args[1]])
    if key == "And":
        return ExprNode("and", [args[0], args[1]])
    if key == "Or":
        return ExprNode("or", [args[0], args[1]])
    if key == "If":
        return ExprNode("if_then_else", [args[0], args[1], args[2]])
    raise ValueError(f"Unsupported qlib function: {name!r}")


def _map_infix_cmp(op: str) -> str:
    mapping = {
        ">": "gt",
        ">=": "ge",
        "<": "lt",
        "<=": "le",
        "==": "eq",
        "!=": "ne",
    }
    try:
        return mapping[op]
    except KeyError as exc:
        raise ValueError(f"Unsupported infix comparison operator: {op!r}") from exc


def parse_qlib_expr(expr: str) -> ExprNode:
    return _QlibExprParser(expr).parse()
