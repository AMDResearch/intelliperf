"""Cursor-style AI agent with explicit think-act loop."""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from rich.console import Console
from rich.panel import Panel

from .core.llm import get_llm_manager
from .tools import (
    create_shell_tools,
    create_editor_tools,
    create_profiler_tools,
    create_snapshot_tools,
)
from .tools.snapshot import restore_snapshot_impl
from .agent_context import manage_context_window

console = Console()

# Debug logging for tool calls
LOG_TOOL_CALLS = True

def log_tool(tool_name: str, arguments: Dict[str, Any]):
    """Log tool calls for debugging."""
    if LOG_TOOL_CALLS:
        console.print(f"[dim]  Tool: {tool_name}[/dim]")
        console.print(f"[dim]  Args: {json.dumps(arguments, indent=2)}[/dim]")


def _load_system_prompt() -> str:
    """
    Load system prompt from file if available, otherwise use built-in default.
    Tries in order:
    1. system_prompt.txt (user customization, not tracked)
    2. system_prompt_default.txt (default prompt, tracked in git)
    3. Built-in fallback
    """
    from pathlib import Path
    workspace_root = Path(__file__).parent.parent.parent
    
    # Try user customization first, then default
    for prompt_filename in ["system_prompt.txt", "system_prompt_default.txt"]:
        prompt_file = workspace_root / prompt_filename
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r') as f:
                    file_content = f.read()
                    # Extract the prompt string from SYSTEM_PROMPT = """..."""
                    if 'SYSTEM_PROMPT = """' in file_content:
                        file_content = file_content.split('SYSTEM_PROMPT = """', 1)[1].rsplit('"""', 1)[0]
                    return file_content
            except Exception as e:
                print(f"Warning: Failed to load system prompt from {prompt_filename}: {e}")
    
    # Fallback to simple default prompt
    return """You are an expert GPU kernel optimizer working like a senior engineer in a code editor.
You have access to tools to explore, modify, build, profile, and validate GPU kernels. Your job is to optimize them through iterative refinement.

**Available Tools:**
{tools}

**How You Work:**
1. **Explore** - Use shell to ls, cat, find files. Understand the project structure, build system, kernel code.
2. **Build** - Use shell to run make/cmake/hipcc. Find the output executable.
3. **Baseline** - Use **profile** tool on the executable to get GPU metrics, extract the kernel time in ms from the output, then call **save_snapshot(name="baseline", description="...", performance_ms=X.XXX, files=[...])** with the extracted time.
4. **Analyze** - Identify bottlenecks from profiling data.
5. **Optimize** - Edit kernel code to improve performance.
6. **Test** - Rebuild, run for correctness, use **profile** tool for accurate GPU timing, extract the time, then **save_snapshot** with descriptive name AND the performance_ms value you extracted.
7. **Iterate** - Continue trying optimizations, ALWAYS passing performance_ms to save_snapshot after profiling.
8. **Recover Best** - Use **list_snapshots** to see all attempts with their performance, then **restore_snapshot** to recover the best one.

**Tool Calling Format:**
To use a tool, output JSON wrapped in ```json ``` fences. For multi-line strings (like file content), use \n for newlines:
```json
{{
  "tool": "edit_file",
  "arguments": {{
    "filename": "test.cpp",
    "old_content": "void foo() {{\n    return;\n}}",
    "new_content": "void foo() {{\n    return 42;\n}}"
  }},
  "reasoning": "Fix return value"
}}
```

**CRITICAL**:
- Output ONLY ONE tool call per response
- For multi-line strings, use \n NOT actual newlines
- After you call a tool, you will see its result and can think about the next step
- DO NOT plan multiple tools in advance - execute one, observe the result, then decide the next action

**Key Principles:**
- **ONE TOOL AT A TIME**: Execute one tool, observe its result, then decide next action.
- Use **profile** tool to get accurate GPU timing and hardware metrics
- **CRITICAL**: After profiling, extract the kernel time and pass it as performance_ms to save_snapshot
- Read files with `cat`, search with `grep`, list with `ls`
- Build with the actual build command (make, cmake, hipcc, etc)
- Run executables directly to check correctness (if binary is in current directory, run as `./<binary>`)
- DON'T make multiple changes at once - iterate!
- DON'T hallucinate tool results - wait for actual output before proceeding
- At the end, review all snapshots and restore the best one
"""

# Load the system prompt at module level
SYSTEM_PROMPT = _load_system_prompt()


class IntelliPerfAgent:
    """Cursor-style agent with explicit think-act loop."""

    def __init__(
        self,
        work_dir: Path,
        goal: str,
        max_iterations: int = 20,
        target_speedup: float = 2.0,
        model: str = "openai/gpt-5.2-codex",
        provider: str = "openrouter",
        verbose: bool = False,
        context_window_size: int = 10
    ):
        self.work_dir = Path(work_dir)
        self.goal = goal
        self.max_iterations = max_iterations
        self.target_speedup = target_speedup
        self.verbose = verbose
        self.context_window_size = context_window_size
        # Initialize LLM
        llm_manager = get_llm_manager(model=model, provider=provider)
        self.llm = llm_manager._create_llm()

        # Initialize tools
        self.tools = self._setup_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Conversation history
        self.messages = []
        self.system_message = None
        self.initial_message = None

        # State tracking (kept across context window slides)
        self.baseline_time = None
        self.current_best_time = None
        self.best_snapshot_dir = None  # Track which snapshot had best performance
        self.iteration = 0  # Counts optimization attempts (baseline + variants)
        self.llm_calls = 0   # Tracks actual LLM invocations

        console.print(f"[green]✓[/green] Agent initialized with {model}")

    def _setup_tools(self):
        """Create all tools."""
        tools = []
        tools.extend(create_shell_tools(self.work_dir))
        tools.extend(create_editor_tools(self.work_dir))
        tools.extend(create_profiler_tools(self.work_dir))
        tools.extend(create_snapshot_tools(self.work_dir))
        return tools

    def _format_tools_description(self) -> str:
        """Format tools for prompt."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(descriptions)

    def optimize(self) -> Dict[str, Any]:
        """Run the optimization loop."""
        console.print(Panel.fit(
            f"[bold cyan]🎯 Goal:[/bold cyan] {self.goal}\n"
            f"[bold cyan]📁 Directory:[/bold cyan] {self.work_dir}\n"
            f"[bold cyan]🎯 Target:[/bold cyan] {self.target_speedup}x speedup",
            title="Starting Optimization"
        ))

        # Initial system message (always kept)
        self.system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools=self._format_tools_description())
        }

        # Initial user message (always kept)
        self.initial_message = {
            "role": "user",
            "content": f"""Goal: {self.goal}
Target speedup: {self.target_speedup}x
Working directory: {self.work_dir}

Start by exploring the project structure to understand what we're working with."""
        }

        self.messages = [self.system_message, self.initial_message]

        # Main loop - keep going until max iterations (optimization attempts) or done
        done = False
        in_baseline = True  # Track if we're establishing baseline

        while self.iteration < self.max_iterations and not done:
            try:
                # THINK: Get agent's thoughts and tool calls
                self.llm_calls += 1
                response = self._think()

                # Check if done
                if "DONE:" in response:
                    done = True
                    console.print(f"\n[yellow]Agent finished: {response.split('DONE:')[1].strip()}[/yellow]")
                    break

                # ACT: Execute all tool calls
                tool_results = self._act(response)

                # OBSERVE: Add conversation to memory
                self.messages.append({"role": "assistant", "content": response})

                if tool_results:
                    results_text = "\n\n".join([
                        f"**{tr['tool']}** result:\n{tr['result']}"
                        for tr in tool_results
                    ])
                    self.messages.append({"role": "user", "content": f"Tool results:\n\n{results_text}"})

                # Check if agent declared a milestone
                milestone = self._check_milestone(response)
                if milestone:
                    milestone_type = milestone.get("milestone")
                    description = milestone.get("description", "")

                    # Just display milestones - LLM is responsible for calling save_snapshot
                    if milestone_type == "baseline_established":
                        console.print(f"\n[bold green]✓ Baseline Established:[/bold green] {description}\n")
                        in_baseline = False
                    elif milestone_type == "optimization_attempt":
                        console.print(f"\n[bold cyan]→ Optimization {self.iteration}/{self.max_iterations}:[/bold cyan] {description}\n")
                    elif milestone_type == "goal_achieved":
                        console.print(f"\n[bold green]🎉 Goal Achieved![/bold green] {description}\n")
                        done = True

                # Manage context window
                self._manage_context()

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                break

        # Get final results
        result = self._get_final_results()

        # Save the best kernel
        self._save_best_kernel()

        return result

    def _think(self) -> str:
        """Get agent's next thought/action."""
        self.llm_calls += 1
        console.print(f"[dim]🤔 Thinking (LLM call {self.llm_calls})...[/dim]")

        # Call LLM (retry if provider returns empty/null content; OpenRouter can occasionally do this)
        last_exc: Optional[Exception] = None
        response = None
        content = None

        for attempt in range(1, 4):
            try:
                response = self.llm.invoke(self.messages)
                content = getattr(response, "content", None)
            except Exception as e:
                last_exc = e
                content = None

            # Accept non-empty text responses
            if content and str(content).strip():
                break

            # If this was an exception, or we got empty content, briefly log metadata to help debugging
            meta = getattr(response, "response_metadata", None)
            extra = getattr(response, "additional_kwargs", None)
            console.print(f"[yellow]⚠️  Empty/None LLM response (attempt {attempt}/3).[/yellow]")
            if last_exc:
                console.print(f"[yellow]  Exception: {repr(last_exc)}[/yellow]")
            if meta:
                console.print(f"[dim]  response_metadata: {meta}[/dim]")
            if extra:
                console.print(f"[dim]  additional_kwargs: {extra}[/dim]")

            # Backoff before retrying
            time.sleep(0.75 * attempt)

        # Extract content
        if content is None:
            if response is None and last_exc is not None:
                console.print(f"[red]⚠️  LLM call raised and produced no response: {repr(last_exc)}[/red]")
            content = getattr(response, "content", None) if response is not None else None
        if content is None:
            content = "" if response is None else str(response)

        # Debug: Show what we got back
        if not content:
            console.print("[red]⚠️  LLM returned None/null[/red]")
            self._debug_print_prompt()
            console.print("[yellow]Stopping optimization early...[/yellow]")
            return "DONE: LLM stopped responding, using best result so far"
        elif not content.strip():
            console.print(f"[red]⚠️  LLM returned empty/whitespace-only response (length: {len(content)})[/red]")
            console.print(f"[dim]Raw: {repr(content[:100])}[/dim]")
            self._debug_print_prompt()
            console.print("[yellow]Stopping optimization early...[/yellow]")
            return "DONE: LLM stopped responding, using best result so far"
        elif len(content.strip()) < 10:
            # Very short response - probably an error
            console.print(f"[yellow]⚠️  Unusually short LLM response ({len(content)} chars): {repr(content)}[/yellow]")

        # Always show thinking (dimmed) and separate tool calls
        self._display_thinking(content)

        return content

    def _debug_print_prompt(self):
        """Print the last few messages sent to LLM for debugging."""
        console.print("\n[yellow]═══ DEBUG: Last messages sent to LLM ═══[/yellow]")
        console.print(f"[dim]Total messages in context: {len(self.messages)}[/dim]\n")

        # Show last 3 messages
        for i, msg in enumerate(self.messages[-3:], start=len(self.messages)-2):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Truncate long content
            if len(content) > 500:
                content_preview = content[:250] + "\n...[truncated]...\n" + content[-250:]
            else:
                content_preview = content

            console.print(f"[cyan]Message {i} ({role}):[/cyan]")
            console.print(f"[dim]{content_preview}[/dim]\n")

        console.print("[yellow]═══════════════════════════════════════[/yellow]\n")

    def _display_thinking(self, content: str):
        """Display agent's full output including reasoning and tool attempts."""
        # Show the full LLM output so user can see what it's doing
        if content.strip():
            console.print(f"[dim]{content}[/dim]\n")

    def _check_milestone(self, response: str) -> Optional[Dict[str, Any]]:
        """Check if agent declared a milestone in their response."""
        # Look for milestone JSON declarations
        import re
        milestone_pattern = r'\{[^{}]*"milestone"[^{}]*\}'
        matches = re.findall(milestone_pattern, response, re.DOTALL)

        for match in matches:
            try:
                milestone = json.loads(match)
                if "milestone" in milestone:
                    return milestone
            except json.JSONDecodeError:
                continue

        return None

    def _act(self, response: str) -> List[Dict[str, Any]]:
        """Parse and execute tool calls from response."""
        tool_results = []

        # Look for JSON tool calls in the response
        import re
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)

        if not matches:
            # Also try without code fences - match complete JSON objects with "tool" key
            json_pattern = r'\{[^{}]*"tool"[^{}]*"tool"[^{}]*\}|\{"tool"[^}]*\{[^}]*\}[^}]*\}'
            # Better approach: find all JSON-like structures and validate them
            potential_jsons = []
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx >= 0:
                        potential_json = response[start_idx:i+1]
                        if '"tool"' in potential_json and '"arguments"' in potential_json:
                            potential_jsons.append(potential_json)
            matches = potential_jsons if potential_jsons else []

        # Execute only the FIRST tool call, then stop to let LLM observe and think
        # This prevents cascading failures and hallucination
        if len(matches) > 1:
            console.print(f"[yellow]⚠️  Warning: {len(matches)} tool calls detected, but only the FIRST will execute![/yellow]")
            console.print(f"[yellow]   Wait for the result before planning the next tool.[/yellow]\n")

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_name = tool_call.get("tool")
                arguments = tool_call.get("arguments", {})
                reasoning = tool_call.get("reasoning", "")

                if tool_name not in self.tool_map:
                    console.print(f"[red]Unknown tool: {tool_name}[/red]")
                    available_tools = ", ".join(sorted(self.tool_map.keys()))
                    error_msg = f"Unknown tool: {tool_name}\n\nAvailable tools: {available_tools}\n\nMake sure you use one of the available tool names."
                    tool_results.append({
                        "tool": "ERROR",
                        "arguments": {},
                        "result": error_msg
                    })
                    break

                # Execute tool - show clearly
                console.print(f"\n[bold cyan]🔧 {tool_name}[/bold cyan]", end="")
                if reasoning:
                    console.print(f" [dim]// {reasoning}[/dim]")
                else:
                    console.print()

                # Log tool call for debugging
                log_tool(tool_name, arguments)

                tool = self.tool_map[tool_name]
                result = tool.func(**arguments)

                # Show result with smart truncation
                result_lines = result.split('\n')
                if len(result_lines) > 15:
                    # Show first 10 and last 5 lines
                    shown = '\n'.join(result_lines[:10]) + '\n[dim]... (truncated) ...[/dim]\n' + '\n'.join(result_lines[-5:])
                else:
                    shown = result

                console.print(shown)
                console.print("[green]✓[/green]")

                # Track optimization attempts via snapshot saves (excluding baseline)
                if tool_name == "save_snapshot":
                    snapshot_name = arguments.get("name", "").lower()
                    # Don't count baseline snapshots
                    if "baseline" not in snapshot_name:
                        self.iteration += 1
                        console.print(f"[dim]→ Optimization attempt {self.iteration} saved[/dim]")

                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })

                # STOP after first tool - let LLM observe result before next action
                if len(matches) > 1:
                    tool_results.append({
                        "tool": "SYSTEM",
                        "arguments": {},
                        "result": f"⚠️  IMPORTANT: You output {len(matches)} tool calls, but ONLY THE FIRST ONE EXECUTED.\nThe other {len(matches)-1} tool(s) were IGNORED.\nDo NOT assume their results - wait for actual execution.\nOutput only ONE tool per response."
                    })
                break

            except json.JSONDecodeError as e:
                error_msg = f"JSON Parse Error: {e}\n\nYour JSON was malformed. Common issues:\n- Multi-line strings must use \\n not actual newlines\n- Missing quotes or commas\n- Unclosed brackets\n\nYour attempted JSON:\n{match[:500]}..."
                console.print(f"[red]❌ Parse Error:[/red] {e}")
                console.print(f"[dim]{match[:200]}...[/dim]")

                # Return error to LLM so it can fix the JSON
                tool_results.append({
                    "tool": "ERROR",
                    "arguments": {},
                    "result": error_msg
                })
                break

            except Exception as e:
                error_msg = f"Tool Execution Error: {e}"
                console.print(f"[red]❌ Tool Error:[/red] {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()

                # Return error to LLM
                tool_results.append({
                    "tool": "ERROR",
                    "arguments": {},
                    "result": error_msg
                })
                break

        return tool_results

    def _manage_context(self):
        """Manage context window to prevent overflow using LLM summarization."""
        old_len = len(self.messages)

        # Use LLM to summarize and prune
        self.messages = manage_context_window(
            self.llm,
            self.messages,
            self.system_message,
            self.initial_message,
            self.context_window_size
        )

        # Always show context management (not just in verbose mode)
        if len(self.messages) < old_len:
            console.print(f"[yellow]📝 Context summarized: {old_len} → {len(self.messages)} messages[/yellow]")


    def _save_best_kernel(self):
        """Restore the best snapshot (lowest performance_ms) if available."""
        snapshots_dir = self.work_dir / ".intelliperf" / "snapshots"
        if not snapshots_dir.exists():
            return

        best_snapshot_dir = None
        best_perf = None

        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if snapshot_dir.is_dir():
                metadata_file = snapshot_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        perf = metadata.get("performance_ms", 0.0)
                        if perf and perf > 0 and (best_perf is None or perf < best_perf):
                            best_perf = perf
                            best_snapshot_dir = snapshot_dir
                    except Exception:
                        continue

        if best_snapshot_dir:
            result = restore_snapshot_impl(self.work_dir, best_snapshot_dir.name)
            console.print(f"[green]✓ Restored best snapshot ({best_perf:.4f}ms):[/green] {best_snapshot_dir.name}")
            console.print(result)
        else:
            console.print("[yellow]⚠️  No valid snapshots found to restore.[/yellow]")

    def _get_final_results(self) -> Dict[str, Any]:
        """Extract final results from snapshot metadata."""
        import json

        # Get all snapshots from .intelliperf/snapshots directory
        snapshots_dir = self.work_dir / ".intelliperf" / "snapshots"
        baseline_perf = None
        best_perf = None
        best_snapshot_name = None

        if snapshots_dir.exists():
            for snapshot_dir in sorted(snapshots_dir.iterdir()):
                if snapshot_dir.is_dir():
                    metadata_file = snapshot_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                            perf = metadata.get("performance_ms", 0.0)
                            name = metadata.get("name", "")

                            # Track baseline (require exact "baseline")
                            name_lower = name.lower().strip()
                            if perf > 0 and name_lower == "baseline":
                                baseline_perf = perf

                            # Track best (lowest time)
                            if perf > 0:
                                if best_perf is None or perf < best_perf:
                                    best_perf = perf
                                    best_snapshot_name = name
                        except Exception:
                            continue

        if baseline_perf is None:
            return {
                "success": False,
                "goal_met": False,
                "speedup": None,
                "baseline_time": None,
                "final_time": None,
                "best_snapshot": None,
                "iterations": self.iteration,
                "error": "Baseline snapshot not found. Save a snapshot with name 'baseline' before optimization."
            }

        # Calculate results
        if baseline_perf and best_perf:
            speedup = baseline_perf / best_perf
            goal_met = speedup >= self.target_speedup

            return {
                "success": True,
                "goal_met": goal_met,
                "speedup": speedup,
                "baseline_time": baseline_perf,
                "final_time": best_perf,
                "best_snapshot": best_snapshot_name,
                "iterations": self.iteration
            }

        return {
            "success": False,
            "goal_met": False,
            "speedup": None,
            "baseline_time": None,
            "final_time": None,
            "iterations": self.iteration
        }
