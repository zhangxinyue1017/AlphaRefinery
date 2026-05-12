# qp_weighted_price_centroid 实验完整分析报告

> 实验目录: `artifacts/runs/llm_refine_family_loop/20260511_081152_qp_weighted_price_centroid`
> 生成时间: 2026-05-11
> 分析范围: Broad (5 scheduler rounds) + Focused (5 scheduler rounds), 共 368 candidates

---

## 一、实验配置概览

| 配置项 | 值 |
|--------|-----|
| Family | `qp_weighted_price_centroid` |
| Target Profile | `raw_alpha` |
| 模型 | gpt-5.4, deepseek-v3.1, qwen3.5-plus, claude-sonnet-4-6, kimi-k2 |
| Broad 策略 | exploratory, 8 candidates/round, max 3 rounds |
| Focused 策略 | balanced, 6 candidates/round, max 3 rounds |
| Anchor Gate | min_icir=0.45, min_sharpe=3.0, max_turnover=0.45 |
| stop_if_no_new_winner | 2 |
| 实际运行 | Broad 5 scheduler rounds (r6-30) + Focused 5 scheduler rounds (r31-55) |

---

## 二、Broad 阶段逐轮分析 (r6–r30)

Broad 阶段共 **5 个 scheduler rounds**，每轮 5 个模型并行，探索性生成 8 个候选。

| Scheduler Round | Archive Rounds | Candidates | Winners | Keeps | Drops | Avg Winner ICIR | Max ICIR | Avg Sharpe | Avg TO |
|-----------------|----------------|------------|---------|-------|-------|-----------------|----------|------------|--------|
| **R1** | r6–r10 | 40 | 5 | 26 | 9 | 0.618 | **0.732** | 6.35 | 0.252 |
| **R2** | r11–r15 | 38 | 4 | 18 | 16 | 0.707 | **0.753** | 5.36 | 0.373 |
| **R3** | r16–r20 | 38 | 5 | 18 | 15 | 0.591 | **0.688** | 4.65 | 0.187 |
| **R4** | r21–r25 | 37 | 4 | 9 | 24 | 0.656 | **0.742** | 5.25 | 0.295 |
| **R5** | r26–r30 | 40 | **1** | 5 | 34 | **0.419** | 0.419 | **2.67** | 0.058 |

### Broad 阶段关键发现

1. **前 4 轮高产**: 每轮产出 4–5 个 winner，ICIR 在 0.59–0.71 区间，Sharpe 4.6–6.4
2. **R5 急剧衰退**: winner 从 4 个骤降至 1 个，avg ICIR 从 0.66 跌至 0.42，avg Sharpe 从 5.25 跌至 2.67
3. **R5 唯一 winner** (`vwap_mean_decay_smooth_30`, r28): ICIR=0.419, TO=0.058 — 超低换手率但预测能力大幅下降
4. **Drop 率攀升**: 从 R1 的 22% (9/40) 升至 R5 的 85% (34/40)，说明 broad 探索空间趋于枯竭
5. **系统决策**: R5 后 `search_improved=True` 但 `consecutive_no_improve=0`，stage_policy 触发 `stage_policy_confirmation`，决定转入 focused

---

## 三、Focused 阶段逐轮分析 (r31–r55)

Focused 阶段基于 **Anchor** (`vwap_mean_volconfirm_20`, r7) 进行精细化迭代，每轮 6 个候选。

| Scheduler Round | Archive Rounds | Candidates | Winners | Keeps | Drops | Avg Winner ICIR | Max ICIR | Avg Sharpe | Avg TO |
|-----------------|----------------|------------|---------|-------|-------|-----------------|----------|------------|--------|
| **R1** | r31–r35 | 30 | 2 | 7 | 21 | 0.771 | **0.806** | 6.13 | 0.233 |
| **R2** | r36–r40 | 28 | 3 | 11 | 14 | 0.739 | 0.744 | 6.19 | 0.217 |
| **R3** | r41–r45 | 29 | 3 | 7 | 19 | 0.772 | **0.781** | 6.33 | 0.280 |
| **R4** | r46–r50 | 28 | 2 | 3 | 23 | 0.773 | **0.791** | 6.33 | 0.354 |
| **R5** | r51–r55 | 30 | 2 | 7 | 21 | 0.756 | 0.761 | 6.28 | 0.295 |

### Focused 阶段关键发现

1. **质量显著提升**: 每轮 avg winner ICIR 稳定在 **0.74–0.77**，显著高于 broad 的 0.42–0.71
2. **Sharpe 稳定高位**: 6.1–6.3，波动极小，说明 focused 阶段产出质量高度一致
3. **Winner 数量减少**: 每轮 2–3 个（vs broad 的 4–5 个），但质量门槛更高
4. **Drop 率更高**: 50–82%（vs broad 的 22–85%），说明筛选更严格
5. **无衰退迹象**: 5 轮 focused 的 avg ICIR 几乎持平，未出现 broad R5 的断崖式下跌

---

## 四、Broad → Focused 过渡分析

### 4.1 过渡触发原因

| 信号 | 值 | 含义 |
|------|-----|------|
| anchor_strength | **strong** | 存在强 anchor |
| winner_quality | **strong** | winner 质量强劲 |
| material_gain | True | 有实质性提升 |
| corr_pressure | low | 相关性压力低 |
| turnover_pressure | low | 换手率压力低 |
| frontier_health | high | 前沿健康度高 |
| saturation_grade | **low** | 饱和度极低 |

### 4.2 过渡效果量化

| 指标 | Anchor (r7) | Focused Best (r52) | Delta | 变化率 |
|------|-------------|-------------------|-------|--------|
| **IC Mean** | 0.0887 | 0.0896 | **+0.0009** | +1.0% |
| **ICIR** | 0.7324 | 0.7608 | **+0.0284** | +3.9% |
| **Ann Return** | 4.618 | 4.990 | **+0.372** | +8.1% |
| **Excess Return** | 0.488 | 0.660 | **+0.172** | +35.1% |
| **Sharpe** | 6.059 | 6.295 | **+0.237** | +3.9% |
| **Turnover** | 0.216 | 0.291 | **+0.075** | +34.8% |

### 4.3 过渡结论

- ✅ **预测能力微增**: ICIR +3.9%, Sharpe +3.9%
- ✅ **收益能力显著提升**: Ann Return +8.1%, Excess +35.1%
- ⚠️ **换手成本上升**: TO +34.8%（从 0.216 到 0.291）
- ⚠️ **未超越 anchor 的质变**: `focused_improved_vs_anchor=False`，提升是渐进式的，未产生"突破性"改进
- ✅ **饱和控制有效**: saturation=low, score=0.0，系统在健康状态下主动终止

---

## 五、顶级 Factor 详细分析

### 5.1 全实验 Top 15 Factors (Winners + Keeps)

| 排名 | Factor 名称 | Round | Model | Status | Role | ICIR | Sharpe | TO |
|------|------------|-------|-------|--------|------|------|--------|-----|
| 1 | `vwap_mean_spread_vol_regime` | r9 | claude | keep_exploratory | confirmation | **0.838** | 4.18 | 0.785 |
| 2 | `vwap_centroid_zscore_trend` | r19 | claude | keep | stretch | **0.811** | 5.13 | 0.531 |
| 3 | `vwap_mean_gap_zscaled_confirm` | r31 | gpt | **winner** | conservative | **0.806** | 6.16 | 0.251 |
| 4 | `vwap_mean_zscore_turnconfirm` | r39 | claude | keep | conservative | **0.800** | 5.93 | 0.251 |
| 5 | `vwap_mean_gap_zturnconfirm` | r36 | gpt | keep | enhancing | **0.800** | 5.93 | 0.251 |
| 6 | `vwap_mean_voladj_20` | r38 | qwen | keep | enhancing | **0.800** | 5.93 | 0.251 |
| 7 | `vwap_mean_volregime_z` | r7 | deepseek | keep | stretch | **0.799** | 5.94 | 0.251 |
| 8 | `vwap_typical_amtrank_volconfirm20` | r49 | claude | **winner** | enhancing | **0.791** | 6.40 | 0.414 |
| 9 | `vwap_typical_turntrend_20` | r41 | gpt | **winner** | enhancing | **0.781** | 6.43 | 0.282 |
| 10 | `vwap_typical_volturn_confirm20` | r44 | claude | **winner** | enhancing | **0.780** | 6.40 | 0.257 |
| 11 | `vwap_typical_vol_med_confirm20` | r53 | qwen | keep | conservative | **0.768** | 6.08 | 0.349 |
| 12 | `llmgen_vwap_typical_medvol20` | r52 | deepseek | **winner** | conservative | **0.761** | 6.30 | 0.291 |
| 13 | `vwap_typical_turn_confirm20` | r53 | qwen | keep | decorrelating | **0.757** | 6.26 | 0.294 |
| 14 | `vwap_typical_med_turnconfirm_20` | r43 | qwen | **winner** | decorrelating | **0.755** | 6.17 | 0.300 |
| 15 | `vwap_mean_turnrank_20` | r38 | qwen | keep | decorrelating | **0.755** | 5.74 | 0.366 |

### 5.2 核心主题识别

所有高质量 factor 围绕一个核心结构：

```
neg(mul(sub(volume_weighted_mean(close, volume, window), mean_price_proxy), volume_signal))
```

即：**VWAP 与价格均值的偏离 × 成交量信号**

**价格均值代理 (mean_price_proxy) 的演进：**
| 代数 | 表达式 | 代表 Factor |
|------|--------|------------|
| v1 | `ts_mean(close, 20)` | `vwap_mean_volconfirm_20` (anchor) |
| v2 | `ts_mean(typical_price, 20)` | `vwap_typical_vol60_confirm20` (r48) |
| v3 | `ts_mean(typical_price, 20)` + `ts_med(volume, 60)` | `llmgen_vwap_typical_medvol20` (r52, best) |

**成交量信号的演进：**
| 代 | 表达式 | 特征 |
|----|--------|------|
| v1 | `decay_linear(div(volume, ts_mean(volume, 40)), 5)` | 短期衰减加权 |
| v2 | `div(volume, ts_mean(volume, 60))` | 60日均值标准化 |
| v3 | `div(volume, ts_med(volume, 60))` | **中位数标准化** (r52 的关键改进) |

---

## 六、Parent-Child 对比分析

### 6.1 核心改进链：Seed → Anchor → Focused Best

```
seed::qp_pressure.vwap_minus_mean_30
  └── r7: vwap_mean_volconfirm_20 (ANCHOR)
        ICIR=0.732, Sharpe=6.06, TO=0.216
        Expr: neg(mul(vwap_mean(close,vol,20) - ts_mean(close,20), 
                       decay_linear(div(vol, ts_mean(vol,40)), 5)))
        
        └── ... (多轮 broad 迭代) ...
        
        └── r48: vwap_typical_vol60_confirm20
              ICIR=0.755, Sharpe=6.26, TO=0.295
              Expr: neg(mul(vwap_mean(close,vol,20) - ts_mean(typical_price,20),
                             div(vol, ts_mean(vol,60))))
              
              └── r52: llmgen_vwap_typical_medvol20 (FOCUSED BEST)
                    ICIR=0.761, Sharpe=6.30, TO=0.291
                    Expr: neg(mul(vwap_mean(close,vol,20) - ts_mean(typical_price,20),
                                   div(vol, ts_med(vol,60))))
```

### 6.2 关键改进点

| 改进 | 从 | 到 | 效果 |
|------|-----|-----|------|
| **价格基准** | `ts_mean(close, 20)` | `ts_mean(typical_price, 20)` | 引入 high/low/close 均价，更全面 |
| **成交量标准化** | `decay_linear` 短期加权 | `div(vol, ts_mean(vol, 60))` | 长期均值稳定 |
| **关键突破** | `ts_mean(vol, 60)` | `ts_med(vol, 60)` | **中位数抗异常值** |

从 r48 到 r52 仅改动一个函数：`ts_mean` → `ts_med`，ICIR 从 0.755 提升到 0.761，Sharpe 从 6.26 提升到 6.30。说明 **中位数对异常成交量的鲁棒性** 是这一 family 的关键优化方向。

### 6.3 所有 Winner 的 Parent-Child Delta

| Round | Winner | Parent | Parent ICIR | Winner ICIR | Delta |
|-------|--------|--------|-------------|-------------|-------|
| r28 | `vwap_mean_decay_smooth_30` | `vwap_minus_mean_30_norm` | 0.256 | 0.419 | **+0.163** |
| r31 | `vwap_mean_gap_zscaled_confirm` | `seed::vwap_mean_volconfirm` | N/A | 0.806 | N/A |
| r36 | `vwap_typical_turnconfirm_20` | `vwap_mean_turnconfirm_20` | 0.735 | 0.744 | **+0.009** |
| r38 | `vwap_typical_turnconfirm_20` | `vwap_mean_turnconfirm_20` | 0.735 | 0.741 | **+0.006** |
| r39 | `vwap_typical_turnconfirm_20` | `vwap_mean_turnconfirm_20` | 0.735 | 0.732 | **-0.003** |
| r41 | `vwap_typical_turntrend_20` | `vwap_typical_turnconfirm_20` | 0.741 | 0.781 | **+0.040** |
| r43 | `vwap_typical_med_turnconfirm_20` | `vwap_typical_turnconfirm_20` | 0.741 | 0.755 | **+0.014** |
| r44 | `vwap_typical_volturn_confirm20` | `vwap_typical_turnconfirm_20` | 0.741 | 0.780 | **+0.039** |
| r48 | `vwap_typical_vol60_confirm20` | `vwap_typical_volturn_confirm20` | 0.780 | 0.755 | **-0.025** |
| r49 | `vwap_typical_amtrank_volconfirm20` | `vwap_typical_volturn_confirm20` | 0.780 | 0.791 | **+0.011** |
| r52 | `llmgen_vwap_typical_medvol20` | `vwap_typical_vol60_confirm20` | 0.755 | 0.761 | **+0.006** |
| r53 | `vwap_typical_vol_decay_confirm20` | `vwap_typical_vol60_confirm20` | 0.755 | 0.751 | **-0.004** |

### 6.4 Parent-Child 规律

- **Focused 阶段增益递减**: 早期 focused round (r41, r44) 能从 parent 获得 +0.04 ICIR 的提升，但后期 (r52, r53) 提升仅 +0.006，甚至为负
- **收益天花板显现**: 从 r41 的 0.781 到 r49 的 0.791，再到 r52 的 0.761，ICIR 在 0.75–0.80 区间振荡，难以突破
- **Best Keep 常优于 Winner**: 多个 round 的 keep 比 winner 有更高 ICIR（如 r53 keep `vol_med_confirm20` ICIR=0.768 > winner `vol_decay_confirm20` ICIR=0.751）

---

## 七、模型贡献分析

| 模型 | Total | Winners | Keeps | Drops | Avg Winner ICIR | Max ICIR | Avg Sharpe |
|------|-------|---------|-------|-------|-----------------|----------|------------|
| **gpt-5.4** | 70 | 7 | 21 | 42 | 0.705 | **0.806** | 5.51 |
| **claude-sonnet-4-6** | 70 | 7 | 31 | 32 | 0.739 | **0.791** | 6.28 |
| **deepseek-v3.1** | 66 | 4 | 21 | 41 | 0.680 | 0.761 | 5.38 |
| **qwen3.5-plus** | 70 | 10 | 24 | 36 | 0.673 | 0.755 | 5.41 |
| **kimi-k2** | 62 | 3 | 14 | 45 | 0.500 | 0.559 | 5.68 |

### 模型特征

- **gpt-5.4**: 峰值最高 (0.806)，但平均产出稳定性一般，drops 率 60%
- **claude-sonnet-4-6**: 平均质量最高 (avg ICIR=0.739, avg Sharpe=6.28)，keep 率最高 (44%)，最"稳健"
- **qwen3.5-plus**: **winner 数量最多** (10 个)，但峰值 ICIR 最低 (0.755)，擅长"量产"
- **deepseek-v3.1**:  winner 数量较少但贡献了 **Focused Best** (r52) 和 **Anchor** (r7)，关键节点贡献者
- **kimi-k2**: 明显弱于其他模型，max ICIR 仅 0.559，drops 率 73%，可能是该 family 上不匹配

---

## 八、综合评估

### 8.1 成功之处

1. **Anchor 质量极高**: `vwap_mean_volconfirm_20` (ICIR=0.732, Sharpe=6.06) 是一个极强的 baseline
2. **Focused 阶段质量稳定**: 5 轮 focused 的 avg winner ICIR 稳定在 0.74–0.77，未出现退化
3. **饱和控制优秀**: saturation=low (0.0)，所有组件均为 0，系统在健康状态下主动终止
4. **主题收敛清晰**: 所有高质量 factor 都围绕 "VWAP-价格均值偏离 × 成交量信号" 这一核心结构
5. **发现了关键优化**: `ts_mean(volume)` → `ts_med(volume)` 的中位数替换是实质性改进

### 8.2 局限与风险

1. **Focused 提升幅度有限**: 相比 anchor，focused best 仅 ICIR +3.9%, Sharpe +3.9%，未产生质变
2. **换手率上升**: Focused best 的 TO (0.291) 比 anchor (0.216) 高 35%，交易成本增加
3. **Top keep 未被充分利用**: `vwap_mean_spread_vol_regime` (ICIR=0.838) 和 `vwap_centroid_zscore_trend` (ICIR=0.811) 被标记为 keep 而非 winner，可能因 TO 过高 (0.785, 0.531)
4. **Decorrelation 数据缺失**: 137/142 个有效 factor 的 decorrelation_grade 为 unknown，未配置 decorrelation targets
5. **kimi-k2 表现不佳**: 该模型在这个 family 上明显弱于其他模型

### 8.3 从 Seed 到 Anchor 的关键跃迁

**Seed factor (`qp_pressure.vwap_minus_mean_30`) 已按 r7 同一 selection backtest 口径补评**。seed pool 中的原始公式为：
```
volume_weighted_mean(close, volume, 30) - mean(close, 30)
```

由于该 family 的配置是 `direction: use_negative_sign`，family-loop 实际作为 parent 评估的是：

```
neg(volume_weighted_mean(close, volume, 30) - mean(close, 30))
```

补充口径：取 r7 anchor 所在 child run 的 `family_backtest_selection_summary.csv` 中 parent baseline 行，
与 Anchor (r7) / Focused Best (r52) 使用同一 selection evaluation 口径。

| | Seed (原始) | Anchor (r7) | Focused Best (r52) |
|--|-------------|-------------|-------------------|
| 结构 | `- (vwap - mean)` | `(vwap-mean) × vol_decay` | `(vwap-typical) × vol/med` |
| ICIR | **0.500** | **0.732** | **0.761** |
| Sharpe | **3.91** | **6.06** | **6.30** |
| TO | **0.136** | 0.216 | 0.291 |

**关键跃迁是 "seed → anchor"**：系统将一个已有正向 baseline 的原始想法 `vwap_minus_mean`
演化为复合结构 `(vwap-mean) × volume_decay`，ICIR 从 0.500 提升到 0.732，Sharpe 从 3.91
提升到 6.06。这说明主要增益来自成交量确认项，而不是简单的方向翻转或纯窗口替换。

后续 focused 阶段（0.732 → 0.761）则是在已验证结构上的渐进优化。

### 8.4 系统建议

根据 `family_loop_summary.md`：
- **recommended_next_step**: `donor_mode`
- **recommended_next_stage_preset**: `donor_validation`
- **reason**: focused 阶段没有继续抬高 anchor，但当前 anchor 足够强，适合转 donor/confirmation

**建议行动：**
1. **将 anchor (`vwap_mean_volconfirm_20`) 和 focused best (`llmgen_vwap_typical_medvol20`) 注册为 donor**
2. **对 Top keep (`vwap_mean_spread_vol_regime`, ICIR=0.838) 进行二次评估** — 尽管 TO=0.785 偏高，但 ICIR 极高，可能在特定策略中有价值
3. **配置 decorrelation targets** 后重新运行，确保新 factor 与 library 中现有 factor 的独立性
4. **考虑降低 kimi-k2 在此 family 的权重**，或为其提供更强的 prompt 引导

---

*报告结束。*
