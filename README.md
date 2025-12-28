# Cognitive Compiler

> Turn **everything** into composable tokens that an LLM can recombine into optimal strategies.

A strategic planning system that represents your assets, capabilities, constraints, and goals as composable tokens, then uses deterministic graph-based planning and LLM reasoning to find exponential paths to your objectives.

## 🎯 The Core Insight

Every element of work can be represented as:

```text
CONCEPT → SYMBOL → OPERATION → COMPOSITION
```

By tokenizing your current state and feeding it to a strategic optimizer, you can discover non-obvious paths to exponential growth.

## ✨ Features

- **📚 Token-Based Representation**: Model any concept (assets, capabilities, constraints, goals) as composable tokens
- **🔍 Deterministic Path Finding**: Graph-based beam search ensures reproducible, explainable results
- **🤖 LLM-Powered Analysis**: Strategic explanations and recommendations from your local LLM
- **🎨 Beautiful Terminal Output**: Rich formatting with colors, tables, and panels
- **⚡ Two-Stage Planning**: Deterministic search + LLM explanation for best of both worlds
- **🎯 Goal-Oriented**: Validates paths against specific goal requirements

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- `gemma3:1b` model (or any compatible model)

### Installation

```bash
# 1. Install Ollama (if not already installed)
# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# 2. Pull the model
ollama pull gemma3:1b

# 3. Install Python dependencies
python3 -m pip install -r requirements.txt

# 4. Start Ollama server (in a separate terminal)
ollama serve
```

### Run

```bash
python3 cognitive_compiler.py
```

That's it! The compiler will:
1. Load your current state (assets, capabilities, constraints, goals)
2. Find exponential paths to your goals using deterministic beam search
3. Optimize your token set with strategic recommendations
4. Discover hidden token combinations you haven't considered

## 📊 Output

The compiler produces beautifully formatted terminal output with:

### 📚 Token Library

A color-coded table showing all your tokens with:
- Exponential scores (1-10)
- Flags: 🤖 automatable, 📈 scalable, 🌐 network effects
- Input/output dependencies

### 🚀 Exponential Paths

Color-coded panels showing the top paths to your goal with:
- Impact scores (green/yellow/red based on value)
- Deterministic path scores
- Exponential reasoning
- Concrete next steps
- Predicted bottlenecks
- Timeline estimates

### ⚡ Token Optimization

Strategic recommendations organized by:
- **📌 Prioritize**: Existing tokens to leverage
- **🆕 Acquire**: New tokens to create/build
- **❌ Ignore**: Tokens to deprioritize
- **🔗 Combinations**: Powerful token synergies
- **📋 Sequence**: Recommended action sequence

### 💎 Hidden Combinations

Surprising token synergies with:
- Emergent capabilities
- Exponential mechanisms
- Why they're non-obvious
- Example applications

## 🏗️ Architecture

### Two-Stage Path Finding

The compiler uses a hybrid approach combining deterministic planning with LLM reasoning:

#### Stage 1: Deterministic Beam Search


1. **Token Activation**: Tokens are "activated" when their required inputs are satisfied
2. **Graph Traversal**: Beam search explores the token dependency graph
3. **Goal Validation**: Paths are validated against goal requirements
4. **Scoring**: Each path receives a deterministic score based on:
   - Token exponential scores
   - Automatable/scalable/network_effects bonuses
   - Input complexity penalties
   - Diminishing returns for longer paths

#### Stage 2: LLM Explanation


1. **Path Analysis**: LLM explains why computed paths are exponential
2. **Enrichment**: Adds impact scores, next steps, bottlenecks, timelines
3. **No Invention**: LLM only explains, never invents new paths

### Benefits

- **Reproducible**: Same inputs = same paths
- **Explainable**: Clear scoring heuristics
- **Goal-Oriented**: Validates against requirements
- **Efficient**: Beam search limits exploration space

## 📖 Usage

### Basic Usage

The default script loads a sample state and runs all analyses:

```bash
python3 cognitive_compiler.py
```

### Customizing Your State

Edit `load_beatsaber_state()` in `cognitive_compiler.py`:

```python
def load_your_state() -> Dict[str, Any]:
    return {
        'assets': {
            'your_asset': {
                'provides': ['capability1', 'capability2'],
                'requires': [],
                'metadata': {'key': 'value'},
                'exponential_score': 8,
                'scalable': True,
                'automatable': True
            }
        },
        'capabilities': {
            'your_skill': {
                'enables': ['outcome1', 'outcome2'],
                'requires': ['input1'],
                'metadata': {},
                'exponential_score': 7
            }
        },
        'constraints': {
            'timeline': {'value': '30_days', 'flexible': False}
        },
        'goals': {
            'your_goal': {
                'requires': ['required_output1', 'required_output2'],
                'metadata': {'priority': 'high'}
            }
        }
    }
```

### Programmatic Usage

```python
from cognitive_compiler import CognitiveCompiler, load_beatsaber_state

# Initialize
compiler = CognitiveCompiler(model='gemma3:1b')

# Load state
state = load_beatsaber_state()
compiler.tokenize_current_state(state)

# Find paths with goal requirements
paths = compiler.find_exponential_paths_v2(
    from_tokens=['token1', 'token2'],
    to_goal="Your goal description",
    goal_requires=['required_output1', 'required_output2'],
    num_paths=3,
    beam_width=30,
    max_steps=7
)

# Optimize token set
optimization = compiler.optimize_token_set(
    objective="Your objective",
    optimization_criteria="Your criteria"
)

# Discover hidden combinations
combinations = compiler.discover_hidden_combinations()
```

### Daily Strategic Review

Use the `daily_strategic_review()` function from `utils.py`:

```python
from utils import daily_strategic_review
from cognitive_compiler import CognitiveCompiler

compiler = CognitiveCompiler(model='gemma3:1b')
# ... load your state ...

daily_strategic_review(compiler)
```

## 🧩 Token Types

| Type       | Description                    | Example                    |
| ---------- | ------------------------------ | -------------------------- |
| **ASSET** | Something you have | Dataset, codebase, model |
| **CAPABILITY** | Something you can do | Skill, tool access, API access |
| **CONSTRAINT** | Limitation | Time, budget, team size |
| **GOAL** | What you want to achieve | Revenue target, partnership |
| **KNOWLEDGE** | Domain expertise | Industry knowledge, patterns |
| **RESOURCE** | External thing you can use | API, service, platform |
| **PATTERN** | Reusable solution template | Architecture pattern, workflow |
| **LEVERAGE** | Multiplier | Network effects, automation |

## ⚙️ Configuration

### Model Selection

```bash
# Use a different Ollama model
export OLLAMA_MODEL='llama3:8b'
python3 cognitive_compiler.py
```

### Path Finding Parameters

```python
paths = compiler.find_exponential_paths_v2(
    from_tokens=[...],
    to_goal="...",
    goal_requires=[...],
    num_paths=3,        # Number of paths to return
    beam_width=30,      # Beam search width (higher = more exploration)
    max_steps=7         # Maximum path length
)
```

## 🔧 Advanced Usage

### Adding Tokens Programmatically

```python
from cognitive_compiler import Token, TokenType

compiler.library.add(Token(
    name="youtube_channel",
    type=TokenType.ASSET,
    outputs=['audience', 'credibility', 'distribution'],
    inputs=['content', 'consistency'],
    metadata={'subscribers': 0, 'started': '2025-01-15'},
    exponential_score=8,
    scalable=True,
    network_effects=True
))
```

### Token Scoring

Tokens are scored deterministically using:

```python
score = exponential_score
    + 1.5 if automatable
    + 1.5 if scalable
    + 1.0 if network_effects
    - 0.2 * len(inputs)  # Penalty for complexity
```

Paths use diminishing returns: `score * (0.92 ** position)` to prevent giant paths from always winning.

### Goal Requirements

Define what outputs are needed to achieve your goal:

```python
goal_requires = [
    'monetizable_product',
    'market_access',
    'payment_stack'
]
```

The beam search will only return paths that satisfy these requirements (or best partials if no complete path exists).

## 📁 Project Structure

```text
strategy-cli/
├── cognitive_compiler.py  # Main compiler implementation
├── utils.py                # Utility functions (daily review, etc.)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── init.md                # Original concept documentation
```

## 🛠️ Dependencies

- `ollama>=0.1.0` - Local LLM integration
- `rich>=13.0.0` - Beautiful terminal formatting

## 🤝 Contributing

This is a strategic planning tool. To extend it:

1. Add new token types in `TokenType` enum
2. Customize scoring weights in `score_token()`
3. Adjust beam search parameters for your use case
4. Extend the state loading functions for your domain

## 📝 License

MIT

---

Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)
