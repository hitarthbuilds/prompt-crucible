import asyncio
import datetime
import json
import pathlib
import uuid
import webbrowser

import click
import pdfplumber
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .blackboard import Blackboard
from .llm import call_ollama_safe, check_model_available, check_ollama_running
from .orchestrator import Orchestrator
from .store import ExperimentStore


DEFAULT_CONFIG = {
    "model": "qwen2.5:14b",
    "judge_model": "qwen2.5:7b",
    "threshold": 0.90,
    "max_iter": 10,
    "chunk_size": 20,
    "timeout": 120,
}


console = Console()
BANNER_LINES = (
    "╔══════════════════════════════╗",
    "║   ⚗  PROMPT CRUCIBLE  ⚗    ║",
    "╚══════════════════════════════╝",
)


def _crucible_dir() -> pathlib.Path:
    return pathlib.Path.cwd() / ".crucible"


def _config_path() -> pathlib.Path:
    return _crucible_dir() / "config.yaml"


def _ensure_crucible_dir() -> pathlib.Path:
    directory = _crucible_dir()
    directory.mkdir(exist_ok=True)
    return directory


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _clock_stamp() -> str:
    return _utc_now().strftime("%H:%M:%S")


def _load_server_module():
    from . import server as server_module
    return server_module


async def _start_embedded_ui_server(
    port: int,
    open_browser: bool = False,
):
    import uvicorn

    server_module = _load_server_module()
    config = uvicorn.Config(
        server_module.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    embedded_server = uvicorn.Server(config)
    embedded_server.install_signal_handlers = lambda: None
    task = asyncio.create_task(embedded_server.serve())

    while not embedded_server.started:
        if task.done():
            raise RuntimeError("uvicorn exited before startup completed")
        await asyncio.sleep(0.05)

    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")

    return server_module, embedded_server, task


async def _stop_embedded_ui_server(
    embedded_server,
    task,
) -> None:
    if embedded_server is None or task is None:
        return
    embedded_server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=5)
    except Exception:
        pass


def _banner() -> None:
    for line in BANNER_LINES:
        console.print(f"[bold bright_cyan]{line}[/]")


def _success(message: str) -> None:
    console.print(f"[bold bright_green]✓ {message}[/]")


def _warning(message: str) -> None:
    console.print(f"[bold yellow]⚠ {message}[/]")


def _error(message: str) -> None:
    console.print(f"[bold bright_red]✗ {message}[/]")


def _muted(message: str) -> None:
    console.print(f"[dim white]{message}[/]")


def _exit_with_error(message: str) -> None:
    _error(message)
    raise click.exceptions.Exit(1)


def _confirm(message: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    response = console.input(
        f"[bold white]{message} [{suffix}]: [/]"
    ).strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def _input_files(inputs_path: pathlib.Path) -> list:
    return [
        path
        for path in sorted(inputs_path.iterdir())
        if path.suffix.lower() in {".txt", ".pdf"}
    ]


def _format_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(
        f"line {index}: {line}" for index, line in enumerate(lines, start=1)
    )


def _read_txt_chunks(path: pathlib.Path) -> list:
    raw_text = path.read_text(encoding="utf-8")
    chunks = raw_text.split("\n\n") if "\n\n" in raw_text else [raw_text]
    results = []
    for chunk in chunks:
        formatted = _format_lines(chunk)
        if formatted:
            results.append({"source": path.name, "input": formatted})
    return results


def _read_pdf_chunks(path: pathlib.Path, chunk_size: int) -> list:
    results = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for index in range(0, len(lines), chunk_size):
                group = lines[index:index + chunk_size]
                formatted = "\n".join(
                    f"line {line_number}: {line}"
                    for line_number, line in enumerate(group, start=1)
                )
                if formatted:
                    results.append(
                        {
                            "source": path.name,
                            "page": page_number,
                            "input": formatted,
                        }
                    )
    return results


def _load_chunks(inputs_path: pathlib.Path, chunk_size: int) -> list:
    chunks = []
    for path in _input_files(inputs_path):
        if path.suffix.lower() == ".txt":
            chunks.extend(_read_txt_chunks(path))
        elif path.suffix.lower() == ".pdf":
            chunks.extend(_read_pdf_chunks(path, chunk_size))
    return chunks


def _prompt_choice() -> str:
    while True:
        console.print(
            "[bright_green][a][/bright_green] Approve  "
            "[yellow][e][/yellow] Edit  "
            "[bright_red][r][/bright_red] Reject"
        )
        choice = console.input("[bold white]> [/]").strip().lower()
        if choice in {"a", "e", "r"}:
            return choice
        _warning("Enter a, e, or r.")


def _edit_json(value: dict) -> dict | None:
    edited = click.edit(json.dumps(value, indent=2))
    if edited is None:
        return None
    try:
        parsed = json.loads(edited)
    except json.JSONDecodeError:
        _error("Edited output was not valid JSON.")
        return None
    if not isinstance(parsed, dict):
        _error("Edited output must be a JSON object.")
        return None
    return parsed


def _print_history_table(rows: list) -> None:
    if not rows:
        _muted("No experiments found.")
        return

    table = Table(
        title="[bold bright_cyan]⚗ Experiment History[/]",
        box=box.SQUARE,
        border_style="bright_cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("Run ID", style="bold white")
    table.add_column("Date", style="dim white")
    table.add_column("Initial", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Δ Score", justify="right")
    table.add_column("Iters", justify="right")
    table.add_column("Reason", style="dim white")

    for row in rows:
        initial_score = float(row.get("initial_score") or 0.0)
        final_score = float(row.get("final_score") or 0.0)
        delta = final_score - initial_score
        if delta > 0:
            delta_style = "bright_green"
        elif delta < 0:
            delta_style = "bright_red"
        else:
            delta_style = "dim white"

        table.add_row(
            str(row.get("run_id", ""))[:8],
            str(row.get("started_at", "")).replace("T", " ")[:19],
            f"{initial_score:.2f}",
            f"{final_score:.2f}",
            f"[{delta_style}]{delta:+.2f}[/{delta_style}]",
            str(row.get("iterations", "")),
            str(row.get("termination_reason", "")),
        )
    console.print(table)


def _print_status_table(
    model: str,
    judge_model: str,
    threshold: float,
    max_iter: int,
    inputs_path: pathlib.Path,
    file_count: int,
) -> None:
    table = Table(show_header=False, box=box.SQUARE, border_style="bright_cyan")
    table.add_column(style="dim white", no_wrap=True)
    table.add_column(style="bold white")
    table.add_row("Model", model)
    table.add_row("Judge", judge_model)
    table.add_row("Threshold", f"{threshold:.2f}")
    table.add_row("Max Iter", str(max_iter))
    table.add_row("Inputs", f"{inputs_path} ({file_count} files)")
    console.print(table)


def _print_case_panel(index: int, total: int, chunk_input: str) -> None:
    console.print(
        Panel(
            Text(chunk_input, style="bold white"),
            title=f"Case {index} of {total}",
            title_align="left",
            border_style="bright_cyan",
            expand=False,
        )
    )


def _print_bootstrap_summary(approved: int, rejected: int) -> None:
    table = Table(
        title="[bold bright_cyan]Bootstrap Review[/]",
        box=box.SQUARE,
        border_style="bright_cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("Approved", justify="right", style="bright_green")
    table.add_column("Rejected", justify="right", style="yellow")
    table.add_row(str(approved), str(rejected))
    console.print(table)


def _score_bar(score: float) -> Text:
    filled = max(0, min(10, int(round(score * 10))))
    color = "bright_red"
    if score > 0.8:
        color = "bright_green"
    elif score > 0.5:
        color = "yellow"

    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (10 - filled), style="dim white")
    return bar


def _print_iteration_panel(iteration: dict, signal: str) -> None:
    score = float(iteration.get("score", 0.0))
    score_line = Text(f"{score:.2f}  ", style="bold white")
    score_line.append_text(_score_bar(score))
    score_line.append(f"  {int(score * 100)}%", style="bold white")

    failure_class = str(iteration.get("failure_class") or "none")
    if iteration.get("is_new_class"):
        failure_class = f"{failure_class} (new)"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim white", no_wrap=True)
    grid.add_column(style="bold white")
    grid.add_row("Score:", score_line)
    grid.add_row("Failures:", str(iteration.get("fail_count", 0)))
    grid.add_row("Failure class:", failure_class)
    grid.add_row("Change:", str(iteration.get("change_summary") or "none"))
    grid.add_row("Signal:", signal)

    console.print(
        Panel(
            grid,
            title=f"Iteration {iteration.get('iteration', '?')}",
            border_style="bright_cyan",
            expand=False,
        )
    )


def _print_final_summary(
    summary: dict,
    final_prompt_path: pathlib.Path,
) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim white", no_wrap=True)
    grid.add_column(style="bold white")
    grid.add_row("Final score:", f"{float(summary.get('final_score', 0.0)):.2f}")
    grid.add_row("Iterations run:", str(summary.get("total_iterations", 0)))
    grid.add_row("Termination reason:", str(summary.get("termination_reason", "")))
    grid.add_row("Final prompt path:", str(final_prompt_path))
    console.print(
        Panel(
            grid,
            title="[bold bright_green]Run Complete[/]",
            border_style="bright_green",
            expand=False,
        )
    )


def _print_final_prompt(final_prompt: str) -> None:
    syntax = Syntax(
        final_prompt,
        "markdown",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )
    console.print(
        Panel(
            syntax,
            title="[bold bright_cyan]Final Optimized Prompt[/]",
            border_style="bright_cyan",
            expand=False,
        )
    )


async def _bootstrap_cases(
    prompt_text: str,
    model: str,
    chunks: list,
    timeout: int,
) -> tuple[list, int]:
    approved_cases = []
    rejected_count = 0
    for index, chunk in enumerate(chunks, start=1):
        chunk_input = chunk["input"]
        output = await call_ollama_safe(
            model=model,
            prompt=f"{prompt_text}\n\n{chunk_input}",
            timeout=timeout,
        )

        console.print()
        _print_case_panel(index, len(chunks), chunk_input)
        console.print("[bold bright_cyan]Output:[/]")
        console.print(
            Syntax(
                json.dumps(output, indent=2),
                "json",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        )

        while True:
            choice = _prompt_choice()
            if choice == "r":
                rejected_count += 1
                break
            if choice == "a":
                expected_output = output
            else:
                expected_output = _edit_json(output)
                if expected_output is None:
                    continue

            approved_cases.append(
                {
                    "case_id": f"c{len(approved_cases) + 1}",
                    "input": chunk_input,
                    "expected_output": expected_output,
                }
            )
            break

    return approved_cases, rejected_count


async def _execute_orchestrator(
    orchestrator: Orchestrator,
    blackboard: Blackboard,
    model: str,
    server_module=None,
) -> dict:
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    initial_prompt = await blackboard.read("current_prompt") or ""
    iteration_history = await blackboard.read("iteration_history")
    if iteration_history is None:
        iteration_history = []
        await blackboard.write("iteration_history", iteration_history)

    if server_module is not None:
        await server_module.reset_live_state()
        await server_module.push_run_state(
            run_id=run_id,
            model=model,
            status="running",
            started_at=started_at.isoformat(),
            elapsed_seconds=0.0,
            current_prompt=initial_prompt,
        )
        await server_module.push_agent_active("idle", "idle")
        await server_module.push_agent_log(_clock_stamp(), "Run initialized", "info")

    initial_score: float | None = None
    final_score = 0.0
    termination_reason = "max_iterations"

    for iteration in range(1, orchestrator.max_iter + 1):
        test_suite = await blackboard.read("test_suite") or []
        if server_module is not None:
            await server_module.push_agent_active(
                "evaluator",
                f"0/{len(test_suite)} cases",
            )

        await orchestrator.evaluator.run(blackboard)
        eval_results = await blackboard.read("eval_results") or []
        overall_score = float(await blackboard.read("overall_score") or 0.0)
        pass_count, fail_count = orchestrator._count_cases(eval_results)

        if server_module is not None:
            for index, case in enumerate(eval_results, start=1):
                case_id = case.get("case_id", f"c{index}")
                score = float(case.get("score") or 0.0)
                passed = case.get("passed") is True
                level = "pass" if passed else "fail"
                outcome = "passed" if passed else "failed"
                await server_module.push_agent_log(
                    _clock_stamp(),
                    f"Evaluator → case {case_id} | score: {score:.2f} | {outcome}",
                    level,
                )
            await server_module.push_agent_active("diagnoser", "analyzing...")

        await orchestrator.diagnoser.run(blackboard)
        failure_reasons = await blackboard.read("failure_reasons") or {}
        failure_class = failure_reasons.get("failure_class")
        is_new_class = failure_reasons.get("is_new_class", False)

        if server_module is not None:
            class_text = failure_class or "none"
            suffix = " (NEW)" if is_new_class else ""
            await server_module.push_agent_log(
                _clock_stamp(),
                f"Diagnoser → failure_class: {class_text}{suffix}",
                "agent",
            )
            await server_module.push_agent_active("optimizer", "optimizing...")

        await orchestrator.optimizer.run(blackboard)
        prompt_candidate = await blackboard.read("prompt_candidate") or {}
        current_prompt = await blackboard.read("current_prompt") or ""

        if server_module is not None:
            confidence = float(prompt_candidate.get("confidence") or 0.0)
            await server_module.push_agent_log(
                _clock_stamp(),
                "Optimizer → "
                f"{prompt_candidate.get('approach', 'incremental_fix')} | "
                f"confidence: {confidence:.2f}",
                "agent",
            )

        if initial_score is None:
            initial_score = overall_score
        final_score = overall_score

        new_prompt = prompt_candidate.get("new_prompt", current_prompt)
        await blackboard.write("current_prompt", new_prompt)

        iteration_record = {
            "iteration": iteration,
            "prompt": new_prompt,
            "score": overall_score,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "failure_class": failure_class,
            "is_new_class": is_new_class,
            "change_summary": prompt_candidate.get("change_summary", ""),
            "what_was_avoided": prompt_candidate.get("what_was_avoided", ""),
            "approach": prompt_candidate.get("approach", "incremental_fix"),
        }

        iteration_history = list(await blackboard.read("iteration_history") or [])
        iteration_history.append(iteration_record)
        await blackboard.write("iteration_history", iteration_history)

        if overall_score >= orchestrator.threshold:
            signal = "threshold_reached"
        elif orchestrator._is_plateau(iteration_history):
            signal = "plateau"
        elif orchestrator._is_stuck(iteration_history):
            signal = "stuck"
        elif iteration >= orchestrator.max_iter:
            signal = "max_iterations"
        else:
            signal = "continue"

        termination_reason = signal
        await blackboard.write("loop_signal", signal)

        if server_module is not None:
            await server_module.push_iteration(
                {
                    "run_id": run_id,
                    "iteration": iteration,
                    "score": overall_score,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "failure_class": failure_class,
                    "change_summary": prompt_candidate.get("change_summary", ""),
                    "approach": prompt_candidate.get("approach", "incremental_fix"),
                    "signal": signal,
                    "current_prompt": new_prompt,
                    "is_new_class": is_new_class,
                }
            )
            await server_module.push_run_state(
                status="running" if signal == "continue" else signal,
                elapsed_seconds=(_utc_now() - started_at).total_seconds(),
                current_prompt=new_prompt,
            )
            await server_module.push_agent_active("idle", "idle")

    final_prompt = await blackboard.read("current_prompt") or ""
    finished_at = _utc_now()

    summary = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "initial_prompt": initial_prompt,
        "total_iterations": len(iteration_history),
        "initial_score": 0.0 if initial_score is None else initial_score,
        "final_score": final_score,
        "final_prompt": final_prompt,
        "termination_reason": termination_reason,
        "iteration_history": iteration_history,
        "model_used": model,
    }

    if server_module is not None:
        await server_module.push_run_state(
            status=termination_reason,
            elapsed_seconds=(finished_at - started_at).total_seconds(),
            current_prompt=final_prompt,
        )
        await server_module.push_agent_log(
            _clock_stamp(),
            f"Run complete → {termination_reason} | final score: {final_score:.2f}",
            "info",
        )
        await server_module.push_agent_active("idle", "idle")

    return summary


async def _run_command(
    prompt_path: pathlib.Path,
    inputs_path: pathlib.Path,
    threshold: float,
    max_iter: int,
    model: str,
    judge_model: str,
    chunk_size: int,
    timeout: int,
    no_ui: bool,
    port: int,
) -> None:
    embedded_server = None
    embedded_task = None
    server_module = None

    try:
        if not no_ui:
            server_module, embedded_server, embedded_task = await _start_embedded_ui_server(
                port=port,
                open_browser=True,
            )

        _banner()
        input_files = _input_files(inputs_path)
        _print_status_table(
            model=model,
            judge_model=judge_model,
            threshold=threshold,
            max_iter=max_iter,
            inputs_path=inputs_path,
            file_count=len(input_files),
        )

        if server_module is not None:
            _success(f"UI server running on http://127.0.0.1:{port}")

        with console.status("[bold bright_cyan]Checking Ollama...[/]"):
            ollama_running = await check_ollama_running()
        if not ollama_running:
            _exit_with_error("Ollama not found")
        _success("Ollama running")

        with console.status(f"[bold bright_cyan]Checking model {model}...[/]"):
            model_available = await check_model_available(model)
        if not model_available:
            _exit_with_error(f"Model '{model}' is not available. Run: ollama pull {model}")
        _success(f"Model ready: {model}")

        with console.status(f"[bold bright_cyan]Checking judge model {judge_model}...[/]"):
            judge_model_available = await check_model_available(judge_model)
        if not judge_model_available:
            _exit_with_error(f"Judge model '{judge_model}' is not available.")
        _success(f"Judge model ready: {judge_model}")

        if no_ui:
            _muted("Running in CLI-only mode.")
        else:
            _muted(f"Live telemetry streaming to http://127.0.0.1:{port}")

        chunks = _load_chunks(inputs_path, chunk_size)
        if not chunks:
            _exit_with_error("No .txt or .pdf chunks were found in the input folder.")

        _muted(f"Loaded {len(chunks)} chunks from {inputs_path}")
        if not _confirm("Proceed with bootstrap review?", default=True):
            raise click.Abort()

        prompt_text = prompt_path.read_text(encoding="utf-8")
        blackboard = Blackboard()
        await blackboard.write("current_prompt", prompt_text)
        await blackboard.write("test_suite", [])
        await blackboard.write("iteration_history", [])

        approved_cases, rejected_count = await _bootstrap_cases(
            prompt_text=prompt_text,
            model=model,
            chunks=chunks,
            timeout=timeout,
        )
        _print_bootstrap_summary(len(approved_cases), rejected_count)

        if len(approved_cases) < 5:
            _warning("Fewer than 5 approved cases were collected.")
        if not approved_cases:
            _warning("No approved cases. Exiting.")
            return

        crucible_dir = _ensure_crucible_dir()
        test_suite_path = crucible_dir / "test_suite.json"
        test_suite_path.write_text(
            json.dumps(approved_cases, indent=2),
            encoding="utf-8",
        )

        if server_module is not None:
            await server_module.push_review_cases(approved_cases, 0)

        await blackboard.write("test_suite", approved_cases)

        orchestrator = Orchestrator(
            model=model,
            judge_model=judge_model,
            threshold=threshold,
            max_iter=max_iter,
            timeout=timeout,
        )
        summary = await _execute_orchestrator(
            orchestrator=orchestrator,
            blackboard=blackboard,
            model=model,
            server_module=server_module,
        )

        iteration_history = summary.get("iteration_history", [])
        for index, iteration in enumerate(iteration_history, start=1):
            signal = (
                summary.get("termination_reason", "continue")
                if index == len(iteration_history)
                else "continue"
            )
            _print_iteration_panel(iteration, signal)

        final_prompt = summary.get("final_prompt", "")
        final_prompt_path = crucible_dir / "final_prompt.txt"
        final_prompt_path.write_text(final_prompt, encoding="utf-8")

        store = ExperimentStore()
        await store.init_db()
        await store.save_experiment(summary)

        for iteration in iteration_history:
            await store.save_iteration(summary["run_id"], iteration)

        final_eval_results = await blackboard.read("eval_results") or []
        saved_cases = []
        for index, result in enumerate(final_eval_results):
            merged_case = dict(approved_cases[index]) if index < len(approved_cases) else {}
            merged_case.update(result)
            saved_cases.append(merged_case)
        if saved_cases:
            await store.save_case_results(
                summary["run_id"],
                int(summary.get("total_iterations", 0)),
                saved_cases,
            )

        _print_final_summary(summary, final_prompt_path)
        _print_final_prompt(final_prompt)
    finally:
        await _stop_embedded_ui_server(embedded_server, embedded_task)


@click.group(name="crucible")
def cli() -> None:
    pass


@cli.command()
def init() -> None:
    directory = _crucible_dir()
    if directory.exists():
        _success("Already initialized.")
        return

    directory.mkdir()
    _config_path().write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
        encoding="utf-8",
    )
    _banner()
    _success("PromptCrucible initialized.")
    _muted("Edit .crucible/config.yaml to configure.")


@cli.command()
@click.option(
    "--prompt",
    "prompt_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--inputs",
    "inputs_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option("--threshold", default=0.90, show_default=True, type=float)
@click.option("--max-iter", default=10, show_default=True, type=int)
@click.option("--model", default="qwen2.5:14b", show_default=True)
@click.option("--judge-model", default="qwen2.5:7b", show_default=True)
@click.option("--chunk-size", default=20, show_default=True, type=int)
@click.option("--timeout", default=120, show_default=True, type=int)
@click.option("--no-ui", is_flag=True, default=False)
@click.option("--port", default=7777, show_default=True, type=int)
def run(
    prompt_path: pathlib.Path,
    inputs_path: pathlib.Path,
    threshold: float,
    max_iter: int,
    model: str,
    judge_model: str,
    chunk_size: int,
    timeout: int,
    no_ui: bool,
    port: int,
) -> None:
    asyncio.run(
        _run_command(
            prompt_path=prompt_path,
            inputs_path=inputs_path,
            threshold=threshold,
            max_iter=max_iter,
            model=model,
            judge_model=judge_model,
            chunk_size=chunk_size,
            timeout=timeout,
            no_ui=no_ui,
            port=port,
        )
    )


@cli.command()
@click.option("--port", default=7777, show_default=True, type=int)
def ui(port: int) -> None:
    async def _ui() -> None:
        _banner()
        server_module, embedded_server, embedded_task = await _start_embedded_ui_server(
            port=port,
            open_browser=True,
        )
        _success(f"UI server running on http://127.0.0.1:{port}")
        _muted("WebSocket endpoint: /ws")
        try:
            await server_module.push_agent_active("idle", "idle")
            await embedded_task
        finally:
            await _stop_embedded_ui_server(embedded_server, embedded_task)

    asyncio.run(_ui())


@cli.command()
def history() -> None:
    async def _history() -> None:
        store = ExperimentStore()
        await store.init_db()
        rows = await store.get_experiments()
        _print_history_table(rows)

    asyncio.run(_history())


@cli.command()
@click.option("--run", "run_id", required=True)
@click.option(
    "--output",
    default="./crucible_export/",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
)
def export(run_id: str, output: pathlib.Path) -> None:
    async def _export() -> None:
        with console.status("[bold bright_cyan]Exporting run...[/]"):
            store = ExperimentStore()
            await store.init_db()
            data = await store.resume_run(run_id)
            if data is None:
                _exit_with_error(f"Run '{run_id}' was not found.")

            output.mkdir(parents=True, exist_ok=True)
            experiment = data["experiment"]
            (output / "final_prompt.txt").write_text(
                experiment.get("final_prompt", ""),
                encoding="utf-8",
            )
            (output / "history.json").write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        _success(f"Exported to {output}")

    asyncio.run(_export())


if __name__ == "__main__":
    cli()
