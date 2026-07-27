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
3. **Test cases** — the captain then gives you test cases to write (often as ground-truth data he gathered manually). Store them in a human-readable place (e.g., YAML); the mechanism for executing them is up to you.
4. **Iterate to green** — once the test cases are written, iterate on the implementation until all test cases pass. You NEVER modify the test cases or their truth data yourself — only the captain changes them. If a test seems wrong, raise it with the captain instead.
5. **Ship** — the captain diffs the changes and pushes.

## Repo Layout & Commands

- Pipeline stages live in the `regulator/` package: `parsing` → `applicability` → `coverage` → `findings` → `report`, with shared dataclasses in `models.py`. `main.py` wires them together. `get_applicable_nodes` (in `applicability.py`) is not yet wired into the pipeline.
- Document-level applicability (`applicability.py`) judges each (SOP, regulatory document) pair in one Sonnet call and records a **descriptive** `DocApplicabilityVerdict` (`verdict` ∈ applicable/not_applicable/conditional, verbatim `trigger_quote`, `reasons`, `missing_facts`, `confidence`). The **decisive** step is separate: `should_audit(verdict)` drops only `not_applicable`. The call sees both profiles plus a deterministic scope slice of the document's own text (`scope_slice`: front matter + windows around every scope/applicability/exemption line), because a verbatim quote cannot come from a profile. Failures fail OPEN (low-confidence `conditional`) so a transient error never silently drops a document.
- Dependencies are installed in `.venv/` (gitignored). Point your IDE interpreter there.
- Run the pipeline: `.venv/bin/python main.py` — writes `output/report.md` and prints a per-stage summary.
- Parse one regulatory doc: `make parse DOC=<flag>` — `<flag>` is a case-insensitive substring of a `data/regulations/*.pdf` name; writes JSONL to `output/<flag>/`. Remove that output with `make clear DOC=<flag>`. (`make help` lists targets; the flag→file resolution lives in `regulator.parse_cli`, runnable directly as `.venv/bin/python -m regulator.parse_cli <flag> --out output/<flag>`.)
- Parse the SOP: `make parse-sop` — parses `data/sop/original.docx` (deterministic atomization; one Haiku call for the profile) into `output/sop/`; `make clear DOC=sop` removes it.
- Profile-only extraction (no structure parsing): `make profile DOC=<flag>` for one reg, `make profile-sop` for the SOP, `make profiles` for all 10 regs + the SOP. Each is one cheap Haiku profile call — it skips the expensive structure passes — and writes `data/profiles/<doc_id>.jsonl` (a slim `document` record + the `profile` record, same JSONL shape as `data/parsed/`). `data/profiles/` is the artifact dir document-level applicability / matching iterates on without re-parsing structure. (CLI: `.venv/bin/python -m regulator.profiles <flag>|--sop|--all`, in `regulator.profiles`, which reuses the profile/identity machinery from `regulator.parsing`.)
- Run the eval harness: `.venv/bin/python -m regulator.evaluation` — executes the test cases in `tests/cases/`, each recognized by shape: task-keyed YAML (regulatory parse, inclusion matching), bare-list YAML (SOP full-tree exact comparison), `.json` with `sop_profile`/`reg_profiles` (profile truth), and `.jsonl` rows with `sop_id`/`doc_id`/`label` (document-level applicability truth). The applicability case gates on `label` (exact), `trigger_quote` (verbatim-in-source AND LLM-judge agreement) and LLM-judged `reasons`/`missing_facts`; `confidence` and bookkeeping fields are reported but not gated. Cheap single-case drivers live in `scratch/` (`run_profile_case.py`, `run_applicability_case.py`).
- Offline unit tests (no API key, no network): `.venv/bin/python -m unittest tests.test_applicability_offline`.
- Regulatory-document parsing stage is documented in `docs/parsing/regulation_docs.md` (pipeline, node model, worked example, failure handling).
- SOP (operating-procedure) parsing stage is documented in `docs/parsing/operating_procedure.md` (deterministic style→depth atomization, node model, worked example, edge cases).
- Lint/format: `.venv/bin/ruff check .` and `.venv/bin/black --check .` (both must pass).

## Maintaining This File

As you learn more about the repo — through conversations with the captain and through writing code — keep this file up to date. Add both:

- anything you deem helpful for future sessions (architecture decisions, gotchas, commands, conventions), and
- anything the captain explicitly asks you to record.
