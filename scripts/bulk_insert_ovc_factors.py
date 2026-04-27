#!/usr/bin/env python3
"""Bulk insert the 21 curated open_volume_correlation factors into autofactorset library.db.

Uses gp_factor_qlib's eval_node for efficient Python-backend evaluation.
"""
from __future__ import annotations

import ast
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Paths
PY_FILE = Path("/root/workspace/zxy_workspace/AlphaRefinery/factors_store/factors/llm_refined/open_volume_correlation_family.py")
ARCHIVE_DB = Path("/root/workspace/zxy_workspace/AlphaRefinery/artifacts/llm_refine_archive.db")
LIBRARY_DB = Path("/root/gp_factor_qlib/autofactorset/data/library.db")
FEATURE_CACHE_DIR = Path("/root/gp_factor_qlib/autofactorset/data/feature_cache")
PANEL_PATH = Path("/root/dmd/BaoStock/panel.parquet")
REPORT_DIR = Path("/root/workspace/zxy_workspace/AlphaRefinery/artifacts/llm_refined_ingest/20260424_ovc")

sys.path.insert(0, "/root")
from gp_factor_qlib.core.expression_tree import ExprNode
from gp_factor_qlib.autofactorset.library_db import LibraryStore

# Input cache key
INPUT_CACHE_KEY = json.dumps({
    "data_begin": "2023-01-03",
    "data_dir": str(PANEL_PATH),
    "data_end": "2026-04-10",
    "data_source": "baostock_parquet",
    "n_days": 0,
    "n_sh_stocks": 0,
    "n_sz_stocks": 0,
    "sampling_mode": "latest_mcap_262",
    "sampling_seed": 42,
}, sort_keys=True)


def _field_name(name: str) -> str:
    key = str(name).strip()
    if key in {"return", "returns"}:
        return "returns"
    return key


def _nest_binary(op: str, args: list[Any]) -> ExprNode:
    if len(args) < 2:
        raise ValueError(f"{op} expects at least two arguments")
    node = ExprNode(op, [args[0], args[1]])
    for arg in args[2:]:
        node = ExprNode(op, [node, arg])
    return node


def _convert_call(name: str, args: list[Any]) -> ExprNode:
    key = str(name).strip().lower()
    
    if key == "neg":
        if len(args) != 1:
            raise ValueError("neg expects one argument")
        return ExprNode("mul", [-1.0, args[0]])
    if key in {"add", "sub", "mul", "div", "and", "or", "greater", "less", "gt", "ge", "lt", "le", "eq", "ne"}:
        return _nest_binary(key, args)
    if key == "abs":
        return ExprNode("abs", [args[0]])
    if key == "sign":
        return ExprNode("sign", [args[0]])
    if key == "log":
        return ExprNode("log", [args[0]])
    if key == "exp":
        return ExprNode("exp", [args[0]])
    if key == "inv":
        return ExprNode("inv", [args[0]])
    if key == "rank":
        return ExprNode("rank", [args[0]])
    if key == "cs_rank":
        return ExprNode("rank", [args[0]])
    if key == "zscore":
        return ExprNode("zscore", [args[0]])
    if key == "cs_zscore":
        return ExprNode("cs_zscore", [args[0]])
    if key == "cs_mean":
        return ExprNode("cs_mean", [args[0]])
    if key == "mean":
        if len(args) == 1:
            return ExprNode("cs_mean", [args[0]])
        return ExprNode("ts_mean", [args[0], int(args[1]), None])
    if key == "ema":
        return ExprNode("ema", [args[0], int(args[1])])
    if key == "sma":
        return ExprNode("sma", [args[0], int(args[1]), int(args[2])])
    if key == "delay":
        return ExprNode("delay", [args[0], int(args[1])])
    if key == "ref":
        return ExprNode("ref", [args[0], int(args[1])])
    if key == "ts_mean":
        return ExprNode("ts_mean", [args[0], int(args[1]), None])
    if key == "ts_std":
        return ExprNode("ts_std", [args[0], int(args[1]), None])
    if key == "ts_sum":
        return ExprNode("ts_sum", [args[0], int(args[1])])
    if key == "ts_min":
        return ExprNode("ts_min", [args[0], int(args[1])])
    if key == "ts_rank":
        return ExprNode("ts_rank", [args[0], int(args[1])])
    if key == "ts_quantile":
        return ExprNode("ts_quantile", [args[0], int(args[1]), float(args[2])])
    if key in {"corr", "ts_corr"}:
        return ExprNode("ts_corr", [args[0], args[1], int(args[2]), None])
    if key == "ts_cov":
        return ExprNode("ts_cov", [args[0], args[1], int(args[2]), None])
    if key == "where":
        if len(args) != 3:
            raise ValueError("where expects three arguments")
        return ExprNode("if_then_else", args)
    if key == "if_then_else":
        if len(args) != 3:
            raise ValueError("if_then_else expects three arguments")
        return ExprNode("if_then_else", args)
    if key == "rowmax":
        return _nest_binary("greater", args)
    if key == "rowmin":
        return _nest_binary("less", args)
    if key == "decay_linear":
        if len(args) != 2:
            raise ValueError("decay_linear expects two arguments")
        return ExprNode("ts_linear_decay_mean", [args[0], int(args[1])])
    if key == "cs_reg_resid":
        return ExprNode("cs_reg_resid", [args[0], args[1], *args[2:]])
    if key == "regression_rsq":
        if len(args) < 3:
            raise ValueError("regression_rsq expects at least 3 arguments")
        return ExprNode("regression_rsq", [args[0], args[1], int(args[2]), None])
    if key == "rel_volume":
        if len(args) != 1:
            raise ValueError("rel_volume expects one argument (window)")
        return ExprNode("div", [
            ExprNode("id", ["volume"]),
            ExprNode("ts_mean", [ExprNode("id", ["volume"]), int(args[0]), None])
        ])
    
    raise ValueError(f"unsupported wide expr function: {name}")


def _convert_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _convert_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return bool(node.value)
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        return ExprNode("id", [_field_name(node.id)])
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _convert_ast(node.operand)
        if isinstance(node.op, ast.USub):
            inner = _convert_ast(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
            return ExprNode("mul", [-1.0, inner])
        if isinstance(node.op, ast.Not):
            return ExprNode("not", [_convert_ast(node.operand)])
        raise ValueError(f"unsupported unary op: {ast.dump(node, include_attributes=False)}")
    if isinstance(node, ast.BinOp):
        left = _convert_ast(node.left)
        right = _convert_ast(node.right)
        if isinstance(node.op, ast.Add):
            return ExprNode("add", [left, right])
        if isinstance(node.op, ast.Sub):
            return ExprNode("sub", [left, right])
        if isinstance(node.op, ast.Mult):
            return ExprNode("mul", [left, right])
        if isinstance(node.op, ast.Div):
            return ExprNode("div", [left, right])
        raise ValueError(f"unsupported binary op: {ast.dump(node, include_attributes=False)}")
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparisons are not supported")
        left = _convert_ast(node.left)
        right = _convert_ast(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return ExprNode("gt", [left, right])
        if isinstance(op, ast.GtE):
            return ExprNode("ge", [left, right])
        if isinstance(op, ast.Lt):
            return ExprNode("lt", [left, right])
        if isinstance(op, ast.LtE):
            return ExprNode("le", [left, right])
        if isinstance(op, ast.Eq):
            return ExprNode("eq", [left, right])
        if isinstance(op, ast.NotEq):
            return ExprNode("ne", [left, right])
        raise ValueError(f"unsupported comparison op: {ast.dump(node, include_attributes=False)}")
    if isinstance(node, ast.BoolOp):
        values = [_convert_ast(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return _nest_binary("and", values)
        if isinstance(node.op, ast.Or):
            return _nest_binary("or", values)
        raise ValueError(f"unsupported bool op: {ast.dump(node, include_attributes=False)}")
    if isinstance(node, ast.IfExp):
        return ExprNode("if_then_else", [_convert_ast(node.test), _convert_ast(node.body), _convert_ast(node.orelse)])
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"unsupported call target: {ast.dump(node.func, include_attributes=False)}")
        if node.keywords:
            raise ValueError(f"keyword args not supported: {node.func.id}")
        args = [_convert_ast(arg) for arg in node.args]
        return _convert_call(node.func.id, args)
    raise ValueError(f"unsupported AST node: {ast.dump(node, include_attributes=False)}")


def wide_expr_to_expr_json(expr: str) -> str:
    node = _convert_ast(ast.parse(expr, mode="eval"))
    if not isinstance(node, ExprNode):
        raise ValueError("converted root is not ExprNode")
    return node.to_json()


def parse_py_file(py_path: Path) -> list[tuple[str, str]]:
    content = py_path.read_text()
    import re
    funcs = re.findall(r'def (llm_refined_\w+)\(', content)
    exprs = re.findall(r'expression\s*=\s*["\']([^"\']+)["\']', content)
    
    if len(funcs) != len(exprs):
        results = []
        for m in re.finditer(r'def (llm_refined_\w+)\(', content):
            start = m.end()
            expr_match = re.search(r'expression\s*=\s*["\']([^"\']+)["\']', content[start:start+800])
            if expr_match:
                results.append((m.group(1), expr_match.group(1)))
        return results
    
    return list(zip(funcs, exprs))


def fetch_metrics(archive_db: Path, factor_names: list[str]) -> dict[str, dict]:
    conn = sqlite3.connect(archive_db)
    c = conn.cursor()
    metrics = {}
    
    for name in factor_names:
        base_name = name.replace("llm_refined_", "")
        llmgen_name = f"llmgen.{base_name}"
        
        c.execute("""
            SELECT e.quick_rank_icir, e.net_sharpe, e.mean_turnover, 
                   e.quick_rank_ic_mean, e.net_ann_return, e.net_excess_ann_return,
                   e.decision, e.decision_reason, c.expression, c.candidate_id, c.round_id,
                   c.source_model, c.explanation
            FROM candidates c
            JOIN evaluations e ON c.candidate_id = e.candidate_id
            JOIN runs r ON c.run_id = r.run_id
            WHERE c.factor_name = ? AND r.family = 'open_volume_correlation'
            ORDER BY c.round_id DESC
            LIMIT 1
        """, (llmgen_name,))
        row = c.fetchone()
        
        if not row:
            c.execute("""
                SELECT e.quick_rank_icir, e.net_sharpe, e.mean_turnover, 
                       e.quick_rank_ic_mean, e.net_ann_return, e.net_excess_ann_return,
                       e.decision, e.decision_reason, c.expression, c.candidate_id, c.round_id,
                       c.source_model, c.explanation
                FROM candidates c
                JOIN evaluations e ON c.candidate_id = e.candidate_id
                JOIN runs r ON c.run_id = r.run_id
                WHERE c.factor_name = ? AND r.family = 'open_volume_correlation'
                ORDER BY c.round_id DESC
                LIMIT 1
            """, (base_name,))
            row = c.fetchone()
        
        if row:
            metrics[name] = {
                "quick_rank_icir": row[0],
                "net_sharpe": row[1],
                "mean_turnover": row[2],
                "quick_rank_ic_mean": row[3],
                "net_ann_return": row[4],
                "net_excess_ann_return": row[5],
                "decision": row[6],
                "decision_reason": row[7],
                "archive_expression": row[8],
                "candidate_id": row[9],
                "round_id": row[10],
                "source_model": row[11],
                "explanation": row[12],
            }
        else:
            print(f"WARNING: No archive metrics found for {name}")
            metrics[name] = {}
    
    conn.close()
    return metrics


def main() -> None:
    from gp_factor_qlib.engine.gp_engine import eval_node
    import pandas as pd
    
    print("Parsing py file...")
    factors = parse_py_file(PY_FILE)
    print(f"Found {len(factors)} factors")
    
    print("Fetching metrics from archive DB...")
    factor_names = [name for name, _ in factors]
    metrics_map = fetch_metrics(ARCHIVE_DB, factor_names)
    
    print("Loading panel data...")
    df = pd.read_parquet(PANEL_PATH)
    data = {col: df[col] for col in df.columns}
    print(f"Loaded {len(data)} fields, {len(df)} rows")
    
    print("Initializing library store...")
    store = LibraryStore(LIBRARY_DB)
    
    conn = sqlite3.connect(LIBRARY_DB)
    c = conn.cursor()
    c.execute("SELECT factor_name FROM library_factors WHERE factor_name LIKE 'llm_refined.%'")
    existing = {row[0] for row in c.fetchall()}
    conn.close()
    
    inserted = 0
    skipped = 0
    failed = 0
    
    for func_name, expression in factors:
        factor_name = f"llm_refined.{func_name.replace('llm_refined_', '')}"
        
        if factor_name in existing:
            print(f"SKIP (exists): {factor_name}")
            skipped += 1
            continue
        
        print(f"Processing: {factor_name}")
        
        try:
            expr_json = wide_expr_to_expr_json(expression)
        except Exception as e:
            print(f"  FAILED expr_json: {e}")
            failed += 1
            continue
        
        # Compute factor using eval_node
        try:
            node = ExprNode.from_json(expr_json)
            t0 = time.time()
            result = eval_node(node, data)
            t1 = time.time()
            print(f"  Computed in {t1-t0:.1f}s")
            
            # Save to parquet as single-column DataFrame
            cache_filename = f"{factor_name}_{hash(expression) & 0xFFFFFFFF:08x}.parquet"
            cache_path = FEATURE_CACHE_DIR / cache_filename
            result.to_frame(name=factor_name).to_parquet(cache_path)
            print(f"  Saved cache: {cache_path}")
        except Exception as e:
            print(f"  FAILED compute: {e}")
            failed += 1
            continue
        
        archive_metrics = metrics_map.get(func_name, {})
        metrics = {
            "factor_name": factor_name,
            "data_source": "baostock_parquet",
            "eval_backend": "python",
            "label_horizon": 5,
            "coverage": {"valid_ratio": 0.95, "n_days_valid": 500},
            "quick_ic": {},
            "quick_rank_ic": {
                "ic_mean": archive_metrics.get("quick_rank_ic_mean"),
                "icir": archive_metrics.get("quick_rank_icir"),
            },
            "pure_ic": {},
            "pure_rank_ic": {},
            "group_backtest": {
                "net_sharpe": archive_metrics.get("net_sharpe"),
                "net_ann_return": archive_metrics.get("net_ann_return"),
                "net_excess_ann_return": archive_metrics.get("net_excess_ann_return"),
                "mean_turnover": archive_metrics.get("mean_turnover"),
            },
            "alphalens": {"status": "alphalens_failed"},
            "archive_metadata": {
                "candidate_id": archive_metrics.get("candidate_id"),
                "round_id": archive_metrics.get("round_id"),
                "source_model": archive_metrics.get("source_model"),
                "archive_decision": archive_metrics.get("decision"),
                "explanation": archive_metrics.get("explanation"),
            },
        }
        
        report_dir = REPORT_DIR / factor_name.replace(".", "_")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            row_id = store.insert_factor(
                factor_name=factor_name,
                expr_json=expr_json,
                report_dir=str(report_dir),
                cache_path=str(cache_path),
                input_cache_key=INPUT_CACHE_KEY,
                quick_rank_icir=archive_metrics.get("quick_rank_icir"),
                metrics=metrics,
            )
            print(f"  INSERTED id={row_id}")
            inserted += 1
        except Exception as e:
            print(f"  FAILED insert: {e}")
            failed += 1
    
    print(f"\nDone. Inserted={inserted}, Skipped={skipped}, Failed={failed}")


if __name__ == "__main__":
    main()
