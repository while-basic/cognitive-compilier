
Drop this into your project (you can paste into `cognitive_compiler.py` and replace/extend pieces).

````python
# === ADD THESE IMPORTS ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable
import re

# ============================================================
# 1) SCORING: make path ranking deterministic + explainable
# ============================================================

def score_token(token: Token) -> float:
    """
    Deterministic heuristic score.
    Tune weights as you like.
    """
    base = float(token.exponential_score or 0)

    # leverage bonuses
    base += 1.5 if token.automatable else 0.0
    base += 1.5 if token.scalable else 0.0
    base += 1.0 if token.network_effects else 0.0

    # small penalty for tokens that require more inputs (harder to activate)
    base -= 0.2 * len(token.inputs)

    return base

def score_path(path: List[str], library: "TokenLibrary") -> float:
    """
    Score a path as sum of token scores with mild diminishing returns
    (prevents giant paths from always winning).
    """
    total = 0.0
    for i, name in enumerate(path):
        tok = library.get(name)
        if not tok:
            continue
        tok_score = score_token(tok)
        # diminishing returns
        total += tok_score * (0.92 ** i)
    return total


# ============================================================
# 2) GRAPH LOGIC: activations + beam search
# ============================================================

@dataclass(frozen=True)
class PlanState:
    available_outputs: frozenset[str]
    active_tokens: frozenset[str]
    path: Tuple[str, ...]  # tokens activated in order

def _outputs_of_tokens(library: "TokenLibrary", token_names: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for name in token_names:
        t = library.get(name)
        if t:
            out.update(t.outputs)
    return out

def goal_satisfied(goal_requires: List[str], available_outputs: Set[str]) -> bool:
    return set(goal_requires).issubset(available_outputs)

def missing_for_goal(goal_requires: List[str], available_outputs: Set[str]) -> List[str]:
    missing = sorted(set(goal_requires) - set(available_outputs))
    return missing

def beam_search_paths(
    library: "TokenLibrary",
    start_tokens: List[str],
    goal_requires: List[str],
    beam_width: int = 25,
    max_steps: int = 6,
    num_paths: int = 3,
) -> List[Dict[str, Any]]:
    """
    Deterministic planner:
    - tokens are "activatable" if inputs ⊆ available_outputs
    - activating a token adds its outputs
    - returns top-k paths by heuristic score
    """

    start_outputs = _outputs_of_tokens(library, start_tokens)
    start_state = PlanState(
        available_outputs=frozenset(start_outputs),
        active_tokens=frozenset(start_tokens),
        path=tuple(),
    )

    beam: List[PlanState] = [start_state]
    solutions: List[PlanState] = []
    seen: Set[Tuple[frozenset[str], frozenset[str]]] = set()

    for _step in range(max_steps):
        candidates: List[PlanState] = []

        for state in beam:
            key = (state.available_outputs, state.active_tokens)
            if key in seen:
                continue
            seen.add(key)

            avail = set(state.available_outputs)

            # If goal satisfied, store as solution
            if goal_satisfied(goal_requires, avail):
                solutions.append(state)
                continue

            # Try activating any token not already active
            for tok in library.tokens.values():
                if tok.name in state.active_tokens:
                    continue

                # activatable?
                if set(tok.inputs).issubset(avail):
                    new_avail = frozenset(avail | set(tok.outputs))
                    new_active = frozenset(set(state.active_tokens) | {tok.name})
                    new_path = tuple(list(state.path) + [tok.name])
                    candidates.append(PlanState(new_avail, new_active, new_path))

        # rank candidates by path score
        candidates.sort(
            key=lambda s: score_path(list(s.path), library),
            reverse=True
        )
        beam = candidates[:beam_width]

        # early exit if we already have enough solutions
        if len(solutions) >= num_paths:
            break

    # finalize: score + bottlenecks
    solutions.sort(
        key=lambda s: score_path(list(s.path), library),
        reverse=True
    )
    solutions = solutions[:num_paths]

    out: List[Dict[str, Any]] = []
    for sol in solutions:
        avail = set(sol.available_outputs)
        out.append({
            "tokens_used": list(sol.path),
            "path_score": round(score_path(list(sol.path), library), 3),
            "goal_missing_after_path": missing_for_goal(goal_requires, avail),
            "available_outputs_count": len(avail),
        })

    # If no solution found, still return best partials (use beam)
    if not out and beam:
        best = beam[:num_paths]
        for st in best:
            avail = set(st.available_outputs)
            out.append({
                "tokens_used": list(st.path),
                "path_score": round(score_path(list(st.path), library), 3),
                "goal_missing_after_path": missing_for_goal(goal_requires, avail),
                "available_outputs_count": len(avail),
                "note": "No complete path found within max_steps; returning best partial."
            })

    return out


# ============================================================
# 3) JSON PARSE + REPAIR (robustness)
# ============================================================

def extract_json_block(text: str) -> str:
    # strip fenced blocks if present
    if "```" in text:
        # prefer ```json
        m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return text.strip()

def parse_json_or_repair(client: anthropic.Anthropic, model: str, text: str) -> Any:
    raw = extract_json_block(text)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Ask the model ONLY to repair JSON (no new content)
        repair_prompt = f"""Repair the following into VALID JSON.
Rules:
- Do not add new keys or new items beyond what is already implied.
- Do not include markdown fences.
- Output JSON only.

BROKEN JSON:
{raw}
"""
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": repair_prompt}]
        )
        repaired = extract_json_block(resp.content[0].text)
        return json.loads(repaired)


# ============================================================
# 4) LLM EXPLAINER PASS: analyze computed paths (not invent)
# ============================================================

def build_explainer_prompt(
    library: "TokenLibrary",
    from_tokens: List[str],
    goal: str,
    goal_requires: List[str],
    computed_paths: List[Dict[str, Any]],
    num_paths: int
) -> str:
    context = library.to_prompt_context()

    return f"""You are a strategic optimizer.
IMPORTANT: The paths below were computed deterministically. Do NOT invent new paths.
Only analyze, explain, and improve next-steps for the given paths.

AVAILABLE TOKENS:
{context}

CURRENT TOKENS (active at start):
{from_tokens}

GOAL:
{goal}

GOAL REQUIREMENTS (must be satisfied by available outputs):
{goal_requires}

COMPUTED PATHS (do not change tokens_used):
{json.dumps(computed_paths, indent=2)}

For each of the top {num_paths} paths:
- Explain why it compounds (exponential mechanism)
- Estimate impact (1-10)
- Give concrete next steps (5-10 bullets)
- Predict bottlenecks (including anything still missing)
- Give timeline estimate

Output as JSON array ONLY:
[
  {{
    "name": "short_path_name",
    "tokens_used": [...],
    "exponential_reasoning": "...",
    "impact_score": 1-10,
    "next_steps": ["..."],
    "bottlenecks": ["..."],
    "timeline": "..."
  }}
]
"""


# ============================================================
# 5) DROP-IN REPLACEMENT: find_exponential_paths()
# ============================================================

# Add this inside your CognitiveCompiler class:

def find_exponential_paths_v2(
    self,
    from_tokens: List[str],
    to_goal: str,
    goal_requires: List[str],
    num_paths: int = 3,
    beam_width: int = 25,
    max_steps: int = 6,
    explainer_model: str = "claude-sonnet-4-20250514",
) -> List[Dict[str, Any]]:
    """
    2-stage:
    1) deterministic beam search to produce candidate paths
    2) LLM explains paths (doesn't invent them)
    """

    computed = beam_search_paths(
        library=self.library,
        start_tokens=from_tokens,
        goal_requires=goal_requires,
        beam_width=beam_width,
        max_steps=max_steps,
        num_paths=num_paths
    )

    prompt = build_explainer_prompt(
        library=self.library,
        from_tokens=from_tokens,
        goal=to_goal,
        goal_requires=goal_requires,
        computed_paths=computed,
        num_paths=num_paths
    )

    resp = self.client.messages.create(
        model=explainer_model,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )

    return parse_json_or_repair(self.client, explainer_model, resp.content[0].text)
````

### How to call it in `__main__`

You need to define what the **goal requires** are (otherwise the planner can’t know what “done” means). Example:

```python
# Example goal requirements (you should tune these)
goal_requires = ["monetizable_product", "market_access", "payment_stack"]

paths = compiler.find_exponential_paths_v2(
    from_tokens=[
        "beatsaber_dataset",
        "pattern_visualizations",
        "search_index",
        "working_pipeline",
        "llm_integration",
        "nvidia_ecosystem_access",
    ],
    to_goal="Generate $1000+ MRR in 30 days",
    goal_requires=goal_requires,
    num_paths=3,
    beam_width=30,
    max_steps=7
)

print(json.dumps(paths, indent=2))
```

---
