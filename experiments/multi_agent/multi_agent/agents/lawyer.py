"""LawyerAgent — real consultation agent with five-section prompt.

One class, runtime-selected specialty. The system prompt is built from
the shared skeleton + specialty markdown file.
"""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.lawyer import LawyerOutput


_VALID_SPECIALTIES: tuple[str, ...] = ("通用", "民事", "劳动", "交通", "婚姻", "房产")


class LawyerAgent(BaseAgent):
    """Consultation agent. ReAct over statute_search / read_article tools."""

    specialty: str = "通用"

    def model_post_init(self, __context) -> None:
        if self.specialty not in _VALID_SPECIALTIES:
            raise ValueError(
                f"unknown specialty: {self.specialty!r}. "
                f"Choices: {list(_VALID_SPECIALTIES)}"
            )

    def system_prompt(self) -> str:
        """Concatenate _five_section_skeleton.md + specialty_<name>.md."""
        prompts_pkg = files("multi_agent.prompts.lawyer")
        skeleton = prompts_pkg.joinpath("_five_section_skeleton.md").read_text(encoding="utf-8")
        specialty_md = prompts_pkg.joinpath(f"specialty_{self.specialty}.md").read_text(encoding="utf-8")
        return f"{skeleton}\n\n---\n\n{specialty_md}"

    def output_schema(self) -> type[LawyerOutput]:
        return LawyerOutput

    def _render_input(self, input) -> str:
        """Build Lawyer 输入. payload 支持以下可选 key:
          - query: 当前用户问题 (必需)
          - sub_cases: multi-issue 子议题列表 (Phase 3)
          - prefetched_evidences: Phase 6f 快路径预检索结果
          - recent_turns: Phase 6h 多轮上下文 (list of dict with question/final_answer)
          - cited_articles: Phase 6h sticky 里累积的引用 (list of dict)
          - history_summary: Phase 6h sticky 压缩后的 prose summary
        """
        import json as _json
        payload = input.payload
        query = str(payload.get("query", ""))
        prefetched = payload.get("prefetched_evidences")
        sub_cases = payload.get("sub_cases", [])
        recent_turns = payload.get("recent_turns") or []
        cited_articles = payload.get("cited_articles") or []
        history_summary = (payload.get("history_summary") or "").strip()

        parts: list[str] = []

        # Phase 6h: 多轮上下文 prefix — 让 Lawyer 知道前文聊过什么.
        # 顺序: history_summary (压缩老 turn) → cited_articles (累积法条) → recent_turns (最近 3 轮原文)
        # 这样不论问"这个法律"还是"接着上面"都有上下文可循.
        has_context = bool(recent_turns or cited_articles or history_summary)
        if has_context:
            parts.append('# 本会话历史上下文 (供你理解"前文"/"这个法律"等指代)')
            parts.append("")
            if history_summary:
                parts.append("## 早期对话摘要")
                parts.append(history_summary)
                parts.append("")
            if cited_articles:
                parts.append("## 本会话累积引用过的法条")
                for c in cited_articles[:10]:  # 防爆
                    law = c.get("law", "") if isinstance(c, dict) else getattr(c, "law", "")
                    art = c.get("article", "") if isinstance(c, dict) else getattr(c, "article", "")
                    if law and art:
                        parts.append(f"  - 《{law}》第{art}条")
                parts.append("")
            if recent_turns:
                parts.append("## 最近 N 轮对话原文")
                for i, t in enumerate(recent_turns, 1):
                    q = (t.get("question", "") if isinstance(t, dict) else getattr(t, "question", ""))[:200]
                    a = (t.get("final_answer", "") if isinstance(t, dict) else getattr(t, "final_answer", ""))[:300]
                    parts.append(f"### 第 {i} 轮")
                    parts.append(f"Q: {q}")
                    parts.append(f"A: {a}")
                    parts.append("")
            parts.append("---")
            parts.append("")

        # 快路径: prefetched_evidences 注入 (Phase 6f)
        if prefetched:
            parts.append(f"# 当前用户咨询")
            parts.append(query)
            parts.append("")
            parts.append("以下是已经检索好的相关法条 (你必须从这里选择 citation, 不要再调任何工具):")
            parts.append("```json")
            parts.append(_json.dumps(prefetched, ensure_ascii=False, indent=2))
            parts.append("```")
            parts.append("")
            parts.append("请基于上述法条 + 上下文撰写五段式 JSON 答复. mode 设为 'consultation' (或信息真不足时设 'clarification', 但优先用上下文消解指代).")
            return "\n".join(parts)

        # 多议题: 跟上下文 prefix 串在一起
        if sub_cases:
            parts.append(f"# 当前用户咨询: {query}")
            parts.append("")
            parts.append("本案包含以下独立子议题(请逐一回答):")
            for i, sc in enumerate(sub_cases, 1):
                issue = sc.get("issue", "") if isinstance(sc, dict) else sc.issue
                specialty = sc.get("specialty", "") if isinstance(sc, dict) else sc.specialty
                parts.append(f"{i}. [{specialty}] {issue}")
            return "\n".join(parts)

        # 默认: 只放 query (前面可能已经有 context prefix)
        if has_context:
            parts.append(f"# 当前用户问题")
            parts.append(query)
            parts.append("")
            parts.append('**重要**: 如果当前问题含"这个/那个/上面/前面"等指代, 必须先从上面的历史上下文里消解指代, 不要轻易反问澄清.')
            return "\n".join(parts)
        return query
