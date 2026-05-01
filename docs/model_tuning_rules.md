# 模型调参铁律（Anti-Leakage Protocol）

> **违反任一条 = 数据泄露（Leakage）。模型再炫都等于零。**

---

## 五条规则

1. **Feature Engineering — No Look-ahead**
   全部特征只能使用 T 日及之前可见的信息。绝对不能使用 T+1 或更远未来的任何数据构造特征。

2. **Time-based Split**
   训练集（Train）= 过去的数据；测试集（Test）= 未来的数据。
   不允许随机划分（random split），必须按时间严格切分。

3. **Train + Tune (CV) — 只在训练集内**
   交叉验证（Cross Validation）和超参选择（Hyperparameter Tuning）只能在训练集内部进行。
   测试集的任何信息不得用于选择超参、特征或模型结构。

4. **Final Test Once — 一次审计**
   测试集仅作一次最终评估（One-time Audit）。
   看完测试集结果后，不可回头修改模型、调参数、改特征再重跑测试集。

5. **违反即作废**
   以上任何一条被违反，该实验结果全部作废，必须从头来过。

6. **Multi-split Validation — 不接受单窗口结论**
   任何"改进"声明必须通过 >=3 个不同 val 窗口的 paired test。
   单 val 窗口的 equity 方差达 ±3%（2026-04-16 实测），足以伪造显著改进。
   判定标准：wins > losses 且 sign test p < 0.05 且 mean equity 提升 >= 1%。
   脚本模板：`/tmp/phase3_final.py`、`/tmp/phase4_verify_std5d.py`。

7. **LR 无法表达 Interaction Signal — 用 Decision Rule**
   如果一个特征的 alpha 来自条件组合（如 `main>0 AND retail<0`），LR 的线性叠加无法捕获。
   此类信号应作为 post-model decision rule（rerank/filter），而非塞进特征向量。
   实证：flow_signal_flag univariate p=0.0074，但加入 LR 后 Δmean <=0 全部失败（Phase 5, 2026-04-17）。

---

## 检查清单

每次调参/训练/回测前，逐项确认：

```
[ ] 特征是否只用了 T 日及之前的数据？
[ ] 数据集是否按时间切分（train在前，test在后）？
[ ] CV 是否只在训练集内部做的？
[ ] 测试集是否是第一次（也是唯一一次）跑？
[ ] 有没有根据测试集结果回头改过模型？
```

[ ] 改进结论是否经过 >=3 val 窗口 paired test 验证？
[ ] 新特征是否是 interaction/conditional signal？若是，是否改用 decision rule？
```

全部打勾才能认定结果有效。

---

## 常见泄露陷阱

- 用全量数据 fit StandardScaler 再划分 → **泄露**（scaler 应只在训练集上 fit）
- 用 T+1 的 label 筛选样本 → **泄露**
- 看了测试集 metrics 后调 threshold 再重跑 → **泄露**
- 随机 train_test_split 而非时间切分 → **泄露**（时序数据自相关）
- Walk-forward 回测中用了未来日的参数 → **泄露**
