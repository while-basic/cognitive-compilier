#----------------------------------------------------------------------------
#File:       cognitive_compiler.py
#Project:    strategy-cli
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Turn any concept into a composable token that LLMs can reason about
#Version:    1.0.0
#License:    MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

"""
Turn any concept into a composable token that LLMs can reason about
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable, Set
from enum import Enum
import json
import os
import re
import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.markdown import Markdown
from rich import box


class TokenType(Enum):
    """What kind of thing is this?"""
    ASSET = "asset"              # Something you have (data, code, model)
    CAPABILITY = "capability"    # Something you can do (skill, tool access)
    CONSTRAINT = "constraint"    # Limitation (time, money, knowledge)
    GOAL = "goal"               # What you want to achieve
    KNOWLEDGE = "knowledge"      # Domain expertise
    RESOURCE = "resource"        # External thing you can use (API, service)
    PATTERN = "pattern"          # Reusable solution template
    LEVERAGE = "leverage"        # Multiplier (network effects, automation)


@dataclass
class Token:
    """Universal representation of ANY concept"""
    name: str
    type: TokenType

    # What it provides
    outputs: List[str]

    # What it requires
    inputs: List[str]

    # Metadata
    metadata: Dict[str, Any]

    # Exponential potential (1-10)
    exponential_score: Optional[int] = None

    # Can this be automated?
    automatable: bool = False

    # Can this scale without linear cost increase?
    scalable: bool = False

    # Does this create network effects?
    network_effects: bool = False

    def to_prompt_string(self) -> str:
        """Format for LLM consumption"""

        props = [
            f"Name: {self.name}",
            f"Type: {self.type.value}",
            f"Provides: {', '.join(self.outputs)}",
            f"Requires: {', '.join(self.inputs)}",
        ]

        if self.exponential_score:
            props.append(f"Exponential Score: {self.exponential_score}/10")

        if self.automatable:
            props.append("✓ Automatable")
        if self.scalable:
            props.append("✓ Scalable")
        if self.network_effects:
            props.append("✓ Network Effects")

        for key, value in self.metadata.items():
            props.append(f"{key}: {value}")

        return " | ".join(props)


class TokenLibrary:
    """Collection of all available tokens"""

    def __init__(self):
        self.tokens: Dict[str, Token] = {}

    def add(self, token: Token):
        """Add token to library"""
        self.tokens[token.name] = token

    def get(self, name: str) -> Optional[Token]:
        """Get token by name"""
        return self.tokens.get(name)

    def search(self,
               type: Optional[TokenType] = None,
               provides: Optional[str] = None,
               requires: Optional[str] = None,
               min_exponential: Optional[int] = None) -> List[Token]:
        """Find tokens matching criteria"""

        results = []
        for token in self.tokens.values():
            if type and token.type != type:
                continue
            if provides and provides not in token.outputs:
                continue
            if requires and requires not in token.inputs:
                continue
            if min_exponential and (not token.exponential_score or token.exponential_score < min_exponential):
                continue
            results.append(token)

        return results

    def to_prompt_context(self) -> str:
        """Format entire library for LLM"""

        sections = {}
        for token in self.tokens.values():
            type_name = token.type.value
            if type_name not in sections:
                sections[type_name] = []
            sections[type_name].append(token.to_prompt_string())

        output = []
        for type_name, tokens in sections.items():
            output.append(f"\n## {type_name.upper()}S")
            for token_str in tokens:
                output.append(f"- {token_str}")

        return "\n".join(output)


# ============================================================
# SCORING: make path ranking deterministic + explainable
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


def score_path(path: List[str], library: TokenLibrary) -> float:
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
# GRAPH LOGIC: activations + beam search
# ============================================================

@dataclass(frozen=True)
class PlanState:
    available_outputs: frozenset[str]
    active_tokens: frozenset[str]
    path: Tuple[str, ...]  # tokens activated in order


def _outputs_of_tokens(library: TokenLibrary, token_names: Iterable[str]) -> Set[str]:
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
    library: TokenLibrary,
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
# JSON PARSE + REPAIR (robustness)
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


def parse_json_or_repair(model: str, text: str) -> Any:
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
        try:
            resp = ollama.generate(
                model=model,
                prompt=repair_prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 2000
                }
            )
            repaired = extract_json_block(resp['response'])
            return json.loads(repaired)
        except Exception as e:
            # Fallback: try to extract JSON array or object
            json_match = re.search(r'(\[.*\]|\{.*\})', raw, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
            raise ValueError(f"Could not parse or repair JSON: {e}")


# ============================================================
# LLM EXPLAINER PASS: analyze computed paths (not invent)
# ============================================================

def build_explainer_prompt(
    library: TokenLibrary,
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


class CognitiveCompiler:
    """Uses LLM to find optimal token compositions"""

    def __init__(self, model: str = "gemma3:1b", base_url: Optional[str] = None):
        self.model = model
        self.base_url = base_url
        self.library = TokenLibrary()

    def tokenize_current_state(self, state_description: Dict[str, Any]):
        """Convert your current situation into tokens"""

        # Assets you have
        if 'assets' in state_description:
            for asset_name, details in state_description['assets'].items():
                self.library.add(Token(
                    name=asset_name,
                    type=TokenType.ASSET,
                    outputs=details.get('provides', []),
                    inputs=details.get('requires', []),
                    metadata=details.get('metadata', {}),
                    exponential_score=details.get('exponential_score'),
                    scalable=details.get('scalable', False),
                    automatable=details.get('automatable', False)
                ))

        # Capabilities you have
        if 'capabilities' in state_description:
            for cap_name, details in state_description['capabilities'].items():
                self.library.add(Token(
                    name=cap_name,
                    type=TokenType.CAPABILITY,
                    outputs=details.get('enables', []),
                    inputs=details.get('requires', []),
                    metadata=details.get('metadata', {}),
                    exponential_score=details.get('exponential_score')
                ))

        # Constraints you face
        if 'constraints' in state_description:
            for const_name, details in state_description['constraints'].items():
                self.library.add(Token(
                    name=const_name,
                    type=TokenType.CONSTRAINT,
                    outputs=[],
                    inputs=[],
                    metadata=details
                ))

        # Goals you want
        if 'goals' in state_description:
            for goal_name, details in state_description['goals'].items():
                self.library.add(Token(
                    name=goal_name,
                    type=TokenType.GOAL,
                    outputs=[],
                    inputs=details.get('requires', []),
                    metadata=details.get('metadata', {})
                ))

    def find_exponential_paths(self,
                               from_tokens: List[str],
                               to_goal: str,
                               num_paths: int = 3) -> List[Dict]:
        """
        Ask LLM to find the most exponential paths from current state to goal
        """

        context = self.library.to_prompt_context()

        prompt = f"""You are a strategic optimizer finding exponential paths to goals.

AVAILABLE TOKENS:
{context}

CURRENT STATE:
You have access to: {', '.join(from_tokens)}

GOAL:
{to_goal}

Find {num_paths} different paths from current state to goal. For each path:

1. IDENTIFY which tokens to combine
2. EXPLAIN why this combination is exponential (not linear)
3. ESTIMATE impact (1-10 scale)
4. LIST concrete next steps
5. PREDICT bottlenecks

Prioritize paths that:
- Have network effects
- Are automatable
- Scale without linear cost
- Create compounding value

Output as JSON array of paths:
[
  {{
    "name": "path_name",
    "tokens_used": ["token1", "token2"],
    "exponential_reasoning": "why this compounds",
    "impact_score": 8,
    "next_steps": ["step1", "step2"],
    "bottlenecks": ["bottleneck1"],
    "timeline": "X days/weeks"
  }}
]

Output JSON only."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 8000
                }
            )
            response_text = response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response if parsing fails
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    def find_exponential_paths_v2(
        self,
        from_tokens: List[str],
        to_goal: str,
        goal_requires: List[str],
        num_paths: int = 3,
        beam_width: int = 25,
        max_steps: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        2-stage deterministic path finding:
        1) deterministic beam search to produce candidate paths
        2) LLM explains paths (doesn't invent them)
        """
        # Stage 1: Deterministic beam search
        computed = beam_search_paths(
            library=self.library,
            start_tokens=from_tokens,
            goal_requires=goal_requires,
            beam_width=beam_width,
            max_steps=max_steps,
            num_paths=num_paths
        )

        if not computed:
            return []

        # Stage 2: LLM explains the computed paths
        prompt = build_explainer_prompt(
            library=self.library,
            from_tokens=from_tokens,
            goal=to_goal,
            goal_requires=goal_requires,
            computed_paths=computed,
            num_paths=num_paths
        )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 3000
                }
            )
            response_text = response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise

        return parse_json_or_repair(self.model, response_text)

    def optimize_token_set(self,
                          objective: str,
                          optimization_criteria: str) -> Dict:
        """
        Ask LLM: "What tokens should I acquire/build to maximize objective?"
        """

        context = self.library.to_prompt_context()

        prompt = f"""You are a strategic advisor optimizing for exponential growth.

CURRENT TOKENS:
{context}

OBJECTIVE:
{objective}

OPTIMIZATION CRITERIA:
{optimization_criteria}

Recommend:
1. Which EXISTING tokens to prioritize/leverage
2. Which NEW tokens to create/acquire
3. Which tokens to ignore/deprioritize
4. How to combine tokens for maximum exponential effect

Output as JSON:
{{
  "prioritize": [
    {{
      "token": "existing_token_name",
      "reason": "why this is high leverage",
      "action": "what to do with it"
    }}
  ],
  "acquire": [
    {{
      "token_to_create": "new_token_name",
      "type": "asset/capability/resource",
      "reason": "why this unlocks exponential growth",
      "effort": "low/medium/high"
    }}
  ],
  "ignore": ["token_name"],
  "combinations": [
    {{
      "tokens": ["token1", "token2"],
      "yields": "outcome",
      "exponential_mechanism": "why this compounds"
    }}
  ],
  "recommended_sequence": ["step1", "step2", "step3"]
}}

Output JSON only."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 8000
                }
            )
            response_text = response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response if parsing fails
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    def discover_hidden_combinations(self) -> List[Dict]:
        """
        Ask LLM to find non-obvious token combinations
        """

        context = self.library.to_prompt_context()

        prompt = f"""You are a pattern recognition system finding non-obvious synergies.

AVAILABLE TOKENS:
{context}

Find unexpected combinations where:
- Outputs of token A enable exponential use of token B
- Multiple tokens combine to create emergent capability
- Token combination eliminates a constraint
- Combination creates network effects

Look for:
- Cross-domain synergies (e.g., robotics data + music AI)
- Constraint elimination (what becomes possible when X + Y?)
- Flywheel effects (what creates self-reinforcing loops?)
- Arbitrage opportunities (undervalued tokens?)

Output as JSON array:
[
  {{
    "combination": ["token1", "token2", "token3"],
    "emergent_capability": "what this unlocks",
    "exponential_mechanism": "why this compounds",
    "surprise_factor": "why this is non-obvious",
    "example_application": "concrete use case"
  }}
]

Find the 5 most surprising combinations. Output JSON only."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 8000
                }
            )
            response_text = response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response if parsing fails
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")


# ==========================================
# YOUR CURRENT STATE AS TOKENS
# ==========================================

def load_beatsaber_state() -> Dict[str, Any]:
    """Your current situation as tokens"""

    return {
        'assets': {
            'beatsaber_dataset': {
                'provides': ['7274_patterns', 'robot_motion_data', 'choreography_labels'],
                'requires': [],
                'metadata': {
                    'size': '19MB',
                    'quality': 'human_curated',
                    'cost': 'free'
                },
                'exponential_score': 9,
                'scalable': True,
                'automatable': True
            },
            'pattern_visualizations': {
                'provides': ['200_images', '3d_animations', 'visual_search'],
                'requires': ['beatsaber_dataset'],
                'metadata': {
                    'format': 'png_gif',
                    'searchable': True
                },
                'exponential_score': 6,
                'scalable': True,
                'automatable': True
            },
            'search_index': {
                'provides': ['semantic_search', 'pattern_discovery'],
                'requires': ['pattern_visualizations'],
                'metadata': {
                    'index_size': '200_patterns',
                    'search_type': 'CLIP_embeddings'
                },
                'exponential_score': 7,
                'scalable': True,
                'automatable': True
            },
            'working_pipeline': {
                'provides': ['scraping', 'extraction', 'conversion', 'visualization'],
                'requires': [],
                'metadata': {
                    'status': 'production_ready',
                    'test_coverage': '100%'
                },
                'exponential_score': 8,
                'scalable': True,
                'automatable': True
            }
        },

        'capabilities': {
            'llm_integration': {
                'enables': ['pattern_generation', 'workflow_optimization'],
                'requires': ['anthropic_api'],
                'metadata': {
                    'models': ['claude-sonnet-4'],
                    'budget': 'available'
                },
                'exponential_score': 9
            },
            'nvidia_ecosystem_access': {
                'enables': ['nemo_training', 'isaac_simulation'],
                'requires': ['gpu_compute'],
                'metadata': {
                    'tools': ['NeMo', 'Isaac_Sim', 'Isaac_Lab'],
                    'status': 'available'
                },
                'exponential_score': 10
            },
            'data_pipeline_engineering': {
                'enables': ['new_domain_application', 'rapid_prototyping'],
                'requires': [],
                'metadata': {
                    'skill_level': 'expert',
                    'proven': True
                },
                'exponential_score': 8
            },
            'systems_thinking': {
                'enables': ['cross_domain_synthesis', 'pattern_recognition'],
                'requires': [],
                'metadata': {
                    'unique_positioning': True
                },
                'exponential_score': 9
            }
        },

        'constraints': {
            'timeline': {
                'value': '30_days',
                'flexible': False
            },
            'team_size': {
                'value': 1,
                'flexible': False
            },
            'budget': {
                'value': 'minimal',
                'flexible': True
            }
        },

        'goals': {
            'revenue_in_30_days': {
                'requires': ['monetizable_product', 'market_access'],
                'metadata': {
                    'priority': 'high',
                    'target': '$1000+_MRR'
                }
            },
            'nvidia_partnership': {
                'requires': ['technical_demonstration', 'unique_value_prop'],
                'metadata': {
                    'priority': 'high',
                    'type': 'strategic'
                }
            },
            'public_launch': {
                'requires': ['demo_video', 'documentation', 'github_repo'],
                'metadata': {
                    'priority': 'medium',
                    'timeline': 'february_2026'
                }
            },
            'defensible_moat': {
                'requires': ['unique_dataset', 'technical_complexity', 'network_effects'],
                'metadata': {
                    'priority': 'high',
                    'type': 'strategic'
                }
            }
        }
    }


# ==========================================
# OUTPUT FORMATTING HELPERS
# ==========================================

def display_token_library(console: Console, library: TokenLibrary):
    """Display token library in a formatted table"""
    table = Table(title="📚 Token Library", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Provides", style="yellow", overflow="fold")
    table.add_column("Score", justify="right", style="bright_blue")
    table.add_column("Flags", style="dim")
    
    for token in library.tokens.values():
        flags = []
        if token.automatable:
            flags.append("🤖")
        if token.scalable:
            flags.append("📈")
        if token.network_effects:
            flags.append("🌐")
        
        score_str = f"{token.exponential_score}/10" if token.exponential_score else "N/A"
        provides_str = ", ".join(token.outputs[:3])
        if len(token.outputs) > 3:
            provides_str += f" (+{len(token.outputs) - 3} more)"
        
        table.add_row(
            token.name,
            token.type.value,
            provides_str,
            score_str,
            " ".join(flags) if flags else "—"
        )
    
    console.print(table)


def display_exponential_paths(console: Console, paths: List[Dict]):
    """Display exponential paths in formatted panels"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🚀 EXPONENTIAL PATHS TO GOAL[/bold cyan]",
        border_style="cyan"
    ))
    
    for i, path in enumerate(paths, 1):
        tokens_used = ", ".join(path.get('tokens_used', []))
        impact = path.get('impact_score', 'N/A')
        timeline = path.get('timeline', 'Unknown')
        
        # Create impact badge
        if isinstance(impact, int):
            if impact >= 8:
                impact_color = "bright_green"
            elif impact >= 6:
                impact_color = "yellow"
            else:
                impact_color = "red"
            impact_text = f"[{impact_color}]{impact}/10[/{impact_color}]"
        else:
            impact_text = str(impact)
        
        # Show path score if available (from deterministic scoring)
        path_score = path.get('path_score')
        score_info = f"  [dim]Path Score:[/dim] [cyan]{path_score}[/cyan]" if path_score else ""
        
        path_content = f"""
[bold]{path.get('name', f'Path {i}')}[/bold]

[dim]Tokens Used:[/dim] {tokens_used}
[dim]Impact Score:[/dim] {impact_text}{score_info}  [dim]Timeline:[/dim] {timeline}

[bold]Exponential Reasoning:[/bold]
{path.get('exponential_reasoning', 'N/A')}

[bold]Next Steps:[/bold]
"""
        for step in path.get('next_steps', []):
            path_content += f"  • {step}\n"
        
        if path.get('bottlenecks'):
            path_content += "\n[bold red]⚠️  Bottlenecks:[/bold red]\n"
            for bottleneck in path.get('bottlenecks', []):
                path_content += f"  • {bottleneck}\n"
        
        console.print(Panel(
            path_content.strip(),
            title=f"Path {i}",
            border_style="blue" if i == 1 else "dim",
            title_align="left"
        ))
        console.print()


def display_optimization(console: Console, optimization: Dict):
    """Display token optimization recommendations"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]⚡ TOKEN OPTIMIZATION[/bold green]",
        border_style="green"
    ))
    
    # Prioritize section
    if optimization.get('prioritize'):
        console.print("\n[bold cyan]📌 PRIORITIZE[/bold cyan]")
        for item in optimization.get('prioritize', []):
            console.print(Panel(
                f"[bold]{item.get('token', 'N/A')}[/bold]\n\n"
                f"[dim]Reason:[/dim] {item.get('reason', 'N/A')}\n"
                f"[dim]Action:[/dim] {item.get('action', 'N/A')}",
                border_style="cyan",
                box=box.ROUNDED
            ))
    
    # Acquire section
    if optimization.get('acquire'):
        console.print("\n[bold yellow]🆕 ACQUIRE[/bold yellow]")
        for item in optimization.get('acquire', []):
            effort = item.get('effort', 'unknown')
            effort_color = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'red'
            }.get(effort.lower(), 'white')
            
            console.print(Panel(
                f"[bold]{item.get('token_to_create', 'N/A')}[/bold] "
                f"([{effort_color}]{effort.upper()}[/{effort_color}] effort)\n\n"
                f"[dim]Type:[/dim] {item.get('type', 'N/A')}\n"
                f"[dim]Reason:[/dim] {item.get('reason', 'N/A')}",
                border_style="yellow",
                box=box.ROUNDED
            ))
    
    # Ignore section
    if optimization.get('ignore'):
        console.print("\n[bold red]❌ IGNORE[/bold red]")
        for token in optimization.get('ignore', []):
            console.print(f"  • [dim]{token}[/dim]")
    
    # Combinations section
    if optimization.get('combinations'):
        console.print("\n[bold magenta]🔗 POWERFUL COMBINATIONS[/bold magenta]")
        for combo in optimization.get('combinations', []):
            tokens = " + ".join(combo.get('tokens', []))
            console.print(Panel(
                f"[bold]{tokens}[/bold]\n\n"
                f"[dim]Yields:[/dim] {combo.get('yields', 'N/A')}\n"
                f"[dim]Mechanism:[/dim] {combo.get('exponential_mechanism', 'N/A')}",
                border_style="magenta",
                box=box.ROUNDED
            ))
    
    # Recommended sequence
    if optimization.get('recommended_sequence'):
        console.print("\n[bold blue]📋 RECOMMENDED SEQUENCE[/bold blue]")
        for i, step in enumerate(optimization.get('recommended_sequence', []), 1):
            console.print(f"  [bold]{i}.[/bold] {step}")


def display_hidden_combinations(console: Console, combinations: List[Dict]):
    """Display hidden token combinations"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]💎 HIDDEN COMBINATIONS[/bold yellow]",
        border_style="yellow"
    ))
    
    for i, combo in enumerate(combinations, 1):
        combo_tokens = " + ".join(combo.get('combination', []))
        
        combo_content = f"""
[bold]{combo_tokens}[/bold]

[bold]✨ Emergent Capability:[/bold]
{combo.get('emergent_capability', 'N/A')}

[bold]🚀 Exponential Mechanism:[/bold]
{combo.get('exponential_mechanism', 'N/A')}

[bold]💡 Why This Is Surprising:[/bold]
{combo.get('surprise_factor', 'N/A')}

[bold]📝 Example Application:[/bold]
{combo.get('example_application', 'N/A')}
"""
        console.print(Panel(
            combo_content.strip(),
            title=f"Combination {i}",
            border_style="yellow",
            title_align="left"
        ))
        console.print()


# ==========================================
# RUN THE COGNITIVE COMPILER
# ==========================================

if __name__ == "__main__":
    console = Console()
    
    # Initialize with Ollama
    model = os.getenv('OLLAMA_MODEL', 'gemma3:1b')
    
    console.print(Panel.fit(
        f"[bold cyan]🤖 Cognitive Compiler[/bold cyan]\n\n"
        f"[dim]Model:[/dim] {model}\n"
        f"[dim]Status:[/dim] Make sure Ollama is running: [cyan]ollama serve[/cyan]\n"
        f"[dim]Pull model:[/dim] [cyan]ollama pull {model}[/cyan]",
        border_style="cyan",
        title="Initialization"
    ))
    
    compiler = CognitiveCompiler(model=model)

    # Load your current state
    console.print("\n[bold yellow]⏳ Loading current state...[/bold yellow]")
    current_state = load_beatsaber_state()
    compiler.tokenize_current_state(current_state)

    console.print("\n")
    display_token_library(console, compiler.library)

    # Find exponential paths to revenue
    console.print("\n[bold yellow]⏳ Finding exponential paths (using deterministic beam search)...[/bold yellow]")
    
    # Define goal requirements (what outputs are needed to achieve the goal)
    goal_requires = ["monetizable_product", "market_access", "payment_stack"]
    
    # Use the new v2 method with deterministic beam search
    paths = compiler.find_exponential_paths_v2(
        from_tokens=[
            'beatsaber_dataset',
            'pattern_visualizations',
            'search_index',
            'working_pipeline',
            'llm_integration',
            'nvidia_ecosystem_access'
        ],
        to_goal="Generate $1000+ MRR in 30 days",
        goal_requires=goal_requires,
        num_paths=3,
        beam_width=30,
        max_steps=7
    )
    
    display_exponential_paths(console, paths)

    # Optimize token set
    console.print("\n[bold yellow]⏳ Optimizing token set...[/bold yellow]")
    optimization = compiler.optimize_token_set(
        objective="Maximum exponential growth in 30 days",
        optimization_criteria="Network effects, automation, defensibility"
    )
    
    display_optimization(console, optimization)

    # Discover hidden combinations
    console.print("\n[bold yellow]⏳ Discovering hidden combinations...[/bold yellow]")
    combinations = compiler.discover_hidden_combinations()
    
    display_hidden_combinations(console, combinations)
    
    console.print("\n[bold green]✅ Analysis complete![/bold green]\n")
