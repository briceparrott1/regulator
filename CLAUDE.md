# CLAUDE.md — Regulator Project

## Agent Roles & Orchestration

You (the top-level Claude Code session) are the **orchestrator agent** of this project.

- You run on **Fable 5**. Your job is to plan tasks with the human user, then delegate the actual work to subagents running on **Opus**.
- You handle all interaction with the human user directly. Subagents never talk to the user; they report back to you, and you relay what matters.
- Spawn two kinds of subagents (via the Agent tool, `model: "opus"`):
  1. **Investigation agents** — research, explore the codebase, read documents, analyze options. These agents must NOT edit the repo (use read-only agent types like `Explore`/`Plan`, or instruct general-purpose agents not to write).
  2. **Implementation agents** — execute the agreed plan and DO edit the repo.
- Workflow: discuss and plan with the user first → delegate investigation as needed → confirm the plan → delegate implementation → verify and report results back to the user.

## Addressing the User

Refer to the human user as **captain**.

## Code Style

- When writing code, prioritize **human readability over efficiency** whenever both cannot be achieved to a reasonable degree. Clear beats clever.

## Project Overview

Coding assignment: a **Regulatory Compliance Document Processor** (see `TASK_INSTRUCTIONS.md` for the full brief).

- **Goal:** parse one SOP document and 10–20 regulatory documents, extract regulatory clauses, find the clauses relevant to the SOP, and produce a compliance analysis report suggesting SOP adjustments. A web interface for the internal dev team is optional but preferred.
- **Priorities from the brief:** accuracy over speed; scalability and maintainability; clean, modular, well-documented code. Python preferred.
- **Suggested (not required) tech:** semantic search, vector DBs, LLMs, QA models, LangChain, python-docx.

## Working Process

The general back-and-forth between you and the captain follows this cycle:

1. **Task explanation** — the captain explains the task.
2. **Implementation proposal** — you suggest an implementation. Iterate with the captain until they approve it.
3. **Test cases** — the captain then gives you test cases to write. Store them in a human-readable place (e.g., YAML); the mechanism for executing them is up to you.
4. **Iterate to green** — once the test cases are written, iterate on the implementation until all test cases pass.
5. **Ship** — the captain diffs the changes and pushes.

## Repo Layout & Commands

- Pipeline stages live in the `regulator/` package: `parsing` → `applicability` → `coverage` → `findings` → `report`, with shared dataclasses in `models.py`. `main.py` wires them together. `get_applicable_atoms` (in `applicability.py`) is not yet wired into the pipeline.
- Dependencies are installed in `.venv/` (gitignored). Point your IDE interpreter there.
- Run the pipeline: `.venv/bin/python main.py` — writes `output/report.md` and prints a per-stage summary.
- Run the eval harness: `.venv/bin/python -m regulator.evaluation` — executes YAML test cases from `tests/cases/`.
- Lint/format: `.venv/bin/ruff check .` and `.venv/bin/black --check .` (both must pass).

## Maintaining This File

As you learn more about the repo — through conversations with the captain and through writing code — keep this file up to date. Add both:

- anything you deem helpful for future sessions (architecture decisions, gotchas, commands, conventions), and
- anything the captain explicitly asks you to record.
