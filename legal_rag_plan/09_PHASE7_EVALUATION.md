# Phase 7 · Evaluation Harness

## 依赖

- Phase 5：HarnessRuntime 端到端可用。
- Phase 6 可选：memory 自进化（不影响评估指标计算）。

## 本阶段交付物

1. `src/legal_rag/eval/metrics.py`
2. `src/legal_rag/eval/retrieval_eval.py`
3. `src/legal_rag/eval/answer_eval.py`
4. `src/legal_rag/eval/multiturn_eval.py`     ← **多轮专项**
5. `src/legal_rag/eval/regression_gates.py`
6. `src/legal_rag/eval/run_eval.py`
7. `scripts/run_eval.py`
8. `data/eval/queries.jsonl`（≥5 条示例，正式 ≥100 条）
9. `data/eval/multiturn.jsonl`（多轮场景，≥5 条）
10. `tests/test_eval_gates.py`

---

## 1. 测试集格式

```json
{"id":"q001","query":"公司单方面解除劳动合同是否合法？","domain":"劳动法","task_type":"legal_opinion","must_include_articles":["劳动合同法第39条","劳动合同法第40条","劳动合同法第87条"]}
{"id":"q002","query":"合同里的竞业限制条款没有补偿有效吗？","domain":"劳动法","task_type":"contract_review","must_include_articles":["劳动合同法第23条","劳动合同法第24条"]}
```

`must_include_articles` 用于 `must_include_hit_rate`：检索 top-K 至少含一个目标条文记 1。

---

## 2. 检索指标

```python
# metrics.py
def recall_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float: ...
def mrr_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float: ...
def ndcg_at_k(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float: ...

def must_include_hit_rate(
    retrieved_meta: list[dict], must_include_articles: list[str]
) -> float:
    """retrieved_meta 取每个 chunk 的 (law_name, article_number)，与 must_include_articles 对齐。"""
```

`must_include_articles` 解析规则：`r"(.+?)第(.+?)条"` → `(law_name, article_number_normalized)`。

---

## 3. 回答指标

```python
def citation_coverage(answer_text: str, citations: list[dict]) -> float:
    """答案中的关键结论段落是否都带 [evidence_id] 标记；MVP 用粗略规则：每段以"."结尾的非空中文段必须含至少一个 [ev_xxx]。"""

def groundedness_score(answer_text: str, citations: list[dict], evidences: list[dict]) -> float:
    """每条 citation 的 quote 是否真的支持答案（MVP 简化：quote 是 evidence.text 子串 + answer 含该 quote 的关键词）。"""

def legal_uncertainty_score(answer_text: str) -> float:
    """检查"建议咨询律师 / 不构成正式法律意见 / 视具体情况"等关键词是否出现。"""

def ungrounded_claim_rate(answer_text: str, citations: list[dict]) -> float:
    """答案中以"根据""依据""依照"开头但后面无 [evidence_id] 的句子比例。"""

def answer_completeness(answer_text: str) -> float:
    """检查 7 段固定结构（结论/法律依据/证据分析/类案参考/风险提示/补充信息/引用来源）是否齐全。"""
```

> Phase 7 用规则版打分；后续可换成 LLM-as-judge，接口签名保持一致。

---

## 4. retrieval_eval.py / answer_eval.py

```python
def run_retrieval_eval(eval_file: Path, runtime: HarnessRuntime) -> dict[str, float]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for sample in iter_jsonl(eval_file):
        # 只跑 retrieval (调 runtime.retriever)，不调 LLM —— 控制评估成本
        ...
    return {
        "recall@5": mean(metrics["r5"]),
        "recall@10": mean(metrics["r10"]),
        "mrr@10": mean(metrics["mrr10"]),
        "must_include_hit_rate": mean(metrics["mihr"]),
    }

def run_answer_eval(eval_file: Path, runtime: HarnessRuntime) -> dict[str, float]:
    metrics: dict[str, list[float]] = defaultdict(list)
    latencies = []
    for sample in iter_jsonl(eval_file):
        t0 = time.perf_counter()
        state = runtime.run(sample["query"], jurisdiction="CN")
        latencies.append((time.perf_counter() - t0) * 1000)
        ...
    return {
        "citation_coverage": mean(...),
        "groundedness_score": mean(...),
        "legal_uncertainty_score": mean(...),
        "ungrounded_claim_rate": mean(...),
        "answer_completeness": mean(...),
        "p50_latency_ms": pct(latencies, 50),
        "p95_latency_ms": pct(latencies, 95),
    }
```

---

## 4.5 多轮指标（multiturn_eval.py）

`data/eval/multiturn.jsonl` 每行一个多轮场景：

```json
{"id":"m001","turns":[
  {"user":"公司单方解除劳动合同合法吗？","expect_kind":"clarification"},
  {"user":"我严重违反规章制度被开除，没补偿","expect_kind":"answer","must_include_articles":["劳动合同法第39条"]},
  {"user":"那条法条具体写了什么？","expect_kind":"answer","expect_reuses_evidence":true}
]}
```

指标：

```python
def clarification_precision(scenarios) -> float:
    """expect_kind=clarification 的轮里 kind 真为 clarification 的比例。"""

def evidence_reuse_rate(scenarios) -> float:
    """expect_reuses_evidence=True 的轮里，本轮 citation evidence_id 与上一轮重叠比例。"""

def sticky_intake_consistency(scenarios) -> float:
    """同一 session 内 task_type 不变的轮占比（除非用户显式切换话题）。"""

def compaction_token_saving_ratio(scenarios) -> float:
    """触发 ContextCompactor 的 session 中 mean(1 - digest.token_after/digest.token_before)。"""

def compaction_pinned_validity_rate(scenarios) -> float:
    """digest.pinned_evidence_ids 中后续真的被 Answer Agent 引用的比例（验证 pinned 选择质量）。"""

def multiturn_groundedness(scenarios) -> float:
    """跨轮回答的 groundedness（含历史 evidence pool）。"""
```

MVP 目标：

```text
clarification_precision        >= 0.80
evidence_reuse_rate            >= 0.70
sticky_intake_consistency      >= 0.95
compaction_token_saving_ratio  >= 0.40   (触发压缩的 session 至少省 40% token)
compaction_pinned_validity_rate>= 0.60   (pinned_evidence_ids 中 ≥60% 在后续被引用)
multiturn_groundedness         >= 0.70
```

## 5. regression_gates.py

```python
GATES = {
    "recall@10":                     ("ge", 0.75),
    "must_include_hit_rate":         ("ge", 0.80),
    "citation_coverage":             ("ge", 0.90),
    "groundedness_score":            ("ge", 0.75),
    "ungrounded_claim_rate":         ("le", 0.10),
    "p95_latency_ms":                ("le", 25_000),
    # 多轮门禁
    "clarification_precision":       ("ge", 0.80),
    "evidence_reuse_rate":           ("ge", 0.70),
    "sticky_intake_consistency":     ("ge", 0.95),
    "compaction_token_saving_ratio": ("ge", 0.40),
    "compaction_pinned_validity_rate":("ge", 0.60),
    "multiturn_groundedness":        ("ge", 0.70),
}

def check_gates(metrics: dict[str, float]) -> list[str]:
    failures = []
    for name, (op, threshold) in GATES.items():
        v = metrics.get(name)
        if v is None: failures.append(f"{name} missing"); continue
        ok = (v >= threshold) if op == "ge" else (v <= threshold)
        if not ok: failures.append(f"{name}={v} fails {op} {threshold}")
    return failures
```

---

## 6. run_eval.py

```python
# scripts/run_eval.py
import typer, json, time
from pathlib import Path
from legal_rag.eval.run_eval import run_all
from legal_rag.eval.regression_gates import check_gates

app = typer.Typer()

@app.command()
def main(
    eval_file: Path = typer.Option(Path("data/eval/queries.jsonl")),
    out: Path = typer.Option(Path("logs/eval/latest.json")),
    enforce_gates: bool = typer.Option(True),
):
    metrics = run_all(eval_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), "utf-8")
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))
    if enforce_gates:
        fails = check_gates(metrics)
        if fails:
            for f in fails: typer.echo(f"GATE FAIL: {f}", err=True)
            raise typer.Exit(code=2)

if __name__ == "__main__":
    app()
```

---

## 端到端验收

### 验收命令

```bash
# 评估集示例
cat > data/eval/queries.jsonl <<'EOF'
{"id":"q001","query":"公司单方面解除劳动合同是否合法？","domain":"劳动法","task_type":"legal_opinion","must_include_articles":["劳动合同法第39条","劳动合同法第40条"]}
{"id":"q002","query":"竞业限制没有补偿有效吗？","domain":"劳动法","task_type":"contract_review","must_include_articles":["劳动合同法第23条","劳动合同法第24条"]}
EOF

EMBEDDING_PROVIDER=siliconflow LLM_PROVIDER=siliconflow USE_RERANKER=true RERANKER_PROVIDER=siliconflow \
  python scripts/run_eval.py --eval-file data/eval/queries.jsonl --out logs/eval/latest.json

cat logs/eval/latest.json

pytest -q tests/test_eval_gates.py
```

### 验收通过条件

- `logs/eval/latest.json` 含全部 8 个指标。
- `check_gates` 在合成数据上能正确发现违例（单测覆盖）。
- `regression_gates.py` CLI 不达标时退出码 2。

---

## Codex Prompt

```text
基于 Phase 5–6，实现 Phase 7：Evaluation Harness（含多轮专项）。

按 PLAN/09_PHASE7_EVALUATION.md 实现：

1. src/legal_rag/eval/{metrics,retrieval_eval,answer_eval,multiturn_eval,regression_gates,run_eval}.py
2. scripts/run_eval.py
3. data/eval/queries.jsonl（先放 5 条单轮示例）
4. data/eval/multiturn.jsonl（先放 3 条多轮示例，覆盖：clarification、代词引用、长会话压缩）
5. tests/test_eval_gates.py

要求：
- retrieval_eval 不调用 LLM，直接 runtime.deps.retriever 检索后算指标。
- answer_eval 调 runtime.run_oneshot() 完整单轮流程；记录 wall clock 算 p50/p95。
- multiturn_eval 调 runtime.start_session/run_turn/close_session 跑完整 scenarios。
- 所有指标函数签名固定，便于将来替换为 LLM-as-judge。
- regression_gates 阈值含多轮门禁。
- 测试覆盖：recall@k / mrr@k 小样本手算可验证；check_gates 在伪造 metrics 上能正确报错；多轮指标在 mock LLM 下可计算（即使数值低）。

本 Phase 不修改 graph / agents。

验收：
  EMBEDDING_PROVIDER=mock LLM_PROVIDER=mock RERANKER_PROVIDER=noop \
    python scripts/run_eval.py --eval-file data/eval/queries.jsonl --multiturn-file data/eval/multiturn.jsonl --enforce-gates=False
  pytest -q
```
