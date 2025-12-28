#----------------------------------------------------------------------------
#File:       utils.py
#Project:    strategy-cli
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Utility functions for cognitive compiler
#Version:    1.0.0
#License:    MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

"""
Utility functions for cognitive compiler
"""

import json
import ollama
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich import box
from cognitive_compiler import CognitiveCompiler, Token, TokenType


def daily_strategic_review(compiler: CognitiveCompiler):
    """Run this every morning"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]🌅 Daily Strategic Review[/bold cyan]",
        border_style="cyan"
    ))
    
    # What's the single most exponential thing I can do today?
    console.print("\n[bold yellow]⏳ Finding today's most exponential action...[/bold yellow]")
    today_action = compiler.find_exponential_paths(
        from_tokens=list(compiler.library.tokens.keys()),
        to_goal="Maximum progress toward revenue + NVIDIA partnership",
        num_paths=1
    )
    
    if today_action:
        action = today_action[0]
        tokens_used = ", ".join(action.get('tokens_used', []))
        impact = action.get('impact_score', 'N/A')
        
        if isinstance(impact, int):
            impact_color = "bright_green" if impact >= 8 else "yellow" if impact >= 6 else "red"
            impact_text = f"[{impact_color}]{impact}/10[/{impact_color}]"
        else:
            impact_text = str(impact)
        
        action_content = f"""
[bold]{action.get('name', 'Today\'s Action')}[/bold]

[dim]Tokens:[/dim] {tokens_used}
[dim]Impact:[/dim] {impact_text}

[bold]Why This Compounds:[/bold]
{action.get('exponential_reasoning', 'N/A')}

[bold]Next Steps:[/bold]
"""
        for step in action.get('next_steps', []):
            action_content += f"  • {step}\n"
        
        if action.get('bottlenecks'):
            action_content += "\n[bold red]⚠️  Watch Out For:[/bold red]\n"
            for bottleneck in action.get('bottlenecks', []):
                action_content += f"  • {bottleneck}\n"
        
        console.print(Panel(
            action_content.strip(),
            title="🎯 TODAY'S MOST EXPONENTIAL ACTION",
            border_style="bright_green",
            box=box.ROUNDED
        ))
    
    # What should I stop doing?
    console.print("\n[bold yellow]⏳ Analyzing what to stop...[/bold yellow]")
    optimization = compiler.optimize_token_set(
        objective="Focus on highest leverage only",
        optimization_criteria="Exponential score > 7, eliminates constraints"
    )
    
    if optimization.get('ignore'):
        console.print("\n[bold red]🛑 STOP DOING[/bold red]")
        for token in optimization.get('ignore', []):
            console.print(f"  • [dim]{token}[/dim]")
    else:
        console.print("\n[dim]No tokens to ignore at this time.[/dim]")
    
    # What new token should I create this week?
    if optimization.get('acquire'):
        console.print("\n[bold yellow]🆕 CREATE THIS WEEK[/bold yellow]")
        for token in optimization.get('acquire', [])[:3]:
            effort = token.get('effort', 'unknown')
            effort_color = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'red'
            }.get(effort.lower(), 'white')
            
            console.print(Panel(
                f"[bold]{token['token_to_create']}[/bold] "
                f"([{effort_color}]{effort.upper()}[/{effort_color}] effort)\n\n"
                f"{token.get('reason', 'N/A')}",
                border_style="yellow",
                box=box.ROUNDED
            ))
    else:
        console.print("\n[dim]No new tokens recommended this week.[/dim]")
    
    console.print("\n[bold green]✅ Review complete![/bold green]\n")


def self_improving_compiler(compiler: CognitiveCompiler) -> Dict:
    """The compiler optimizes its own token library"""
    
    prompt = f"""You are analyzing a token-based system.

CURRENT TOKENS:
{compiler.library.to_prompt_context()}

Identify:
1. Missing token types that would unlock new combinations
2. Redundant tokens that should be merged
3. Tokens with incorrect exponential scores
4. Token metadata that should be added

Output as JSON:
{{
  "add_token_types": ["new_type"],
  "merge": [["token1", "token2"]],
  "rescore": [{{"token": "name", "old_score": 5, "new_score": 8, "reason": "why"}}],
  "add_metadata": [{{"token": "name", "field": "field_name", "value": "value"}}]
}}
"""
    
    try:
        response = ollama.generate(
            model=compiler.model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'num_predict': 4000
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
    
    # Apply improvements
    try:
        improvements = json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from response if parsing fails
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            improvements = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
    
    # The system improves itself
    return improvements
