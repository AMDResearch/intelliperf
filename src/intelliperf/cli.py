"""Simplified CLI for IntelliPerf Cursor-style agent."""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from .agent import IntelliPerfAgent

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description='IntelliPerf - AI-Powered GPU Kernel Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  intelliperf "make this 2x faster" --dir examples/atomic_reduction
  intelliperf "optimize memory bandwidth" --dir my_kernel --max-iterations 20
  intelliperf "reduce atomic operations" --target-speedup 5.0
  intelliperf "optimize kernel" --provider anthropic --model claude-sonnet-4.5
        """
    )

    parser.add_argument('goal', type=str, help='Natural language optimization goal')
    parser.add_argument('--dir', type=str, default='.', help='Project directory (default: .)')
    parser.add_argument('--max-iterations', type=int, default=5, help='Maximum optimization attempts (default: 5)')
    parser.add_argument('--target-speedup', type=float, default=2.0, help='Target speedup (default: 2.0x)')
    parser.add_argument('--model', type=str, default='openai/gpt-5.2-codex',
                       help='LLM model (default: openai/gpt-5.2-codex)')
    parser.add_argument('--provider', type=str, default='openrouter',
                       choices=['openrouter', 'openai', 'azure', 'anthropic'],
                       help='LLM provider (default: openrouter)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed agent thinking')

    args = parser.parse_args()

    try:
        work_dir = Path(args.dir).resolve()
        if not work_dir.exists():
            console.print(f"[red]Error: Directory {args.dir} not found[/red]")
            return 1

        console.print(Panel.fit(
            f"[bold cyan]IntelliPerf - AI Kernel Optimizer[/bold cyan]\n\n"
            f"Goal: {args.goal}\n"
            f"Directory: {work_dir}\n"
            f"Provider: {args.provider}\n"
            f"Model: {args.model}",
            title="🚀 Starting Optimization"
        ))

        # Create and run agent
        agent = IntelliPerfAgent(
            work_dir=work_dir,
            goal=args.goal,
            max_iterations=args.max_iterations,
            target_speedup=args.target_speedup,
            model=args.model,
            provider=args.provider,
            verbose=args.verbose
        )

        result = agent.optimize()

        # Display results
        if result["success"]:
            goal_met = result.get("goal_met", False)
            description = result.get("description")
            speedup = result.get("speedup")
            baseline = result.get("baseline_time")
            final = result.get("final_time")

            console.print("\n" + "="*60)

            # If we have goal_achieved milestone description, use it
            if goal_met and description:
                console.print(Panel.fit(
                    f"[bold green]🎉 SUCCESS - Goal Achieved![/bold green]\n\n"
                    f"{description}\n\n"
                    f"Target: {args.target_speedup:.2f}x\n"
                    f"Optimization attempts: {result.get('iterations', 0)}",
                    title="Results"
                ))
            elif speedup and baseline and final:
                if goal_met:
                    status = "[bold green]🎉 SUCCESS - Goal Achieved![/bold green]"
                elif speedup > 1.1:
                    status = "[bold yellow]⚡ Improved (goal not fully met)[/bold yellow]"
                else:
                    status = "[bold yellow]📊 No significant improvement[/bold yellow]"

                console.print(Panel.fit(
                    f"{status}\n\n"
                    f"Baseline: {baseline:.3f} ms\n"
                    f"Final: {final:.3f} ms\n"
                    f"Speedup: [cyan]{speedup:.2f}x[/cyan]\n"
                    f"Target: {args.target_speedup:.2f}x\n"
                    f"Optimization attempts: {result.get('iterations', 0)}",
                    title="Results"
                ))
            else:
                console.print(Panel.fit(
                    "Optimization session completed.\n"
                    f"Optimization attempts: {result.get('iterations', 0)}",
                    title="Results"
                ))

            console.print("="*60 + "\n")

            return 0 if goal_met else 1
        else:
            console.print(f"\n[red]❌ Optimization failed:[/red] {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
