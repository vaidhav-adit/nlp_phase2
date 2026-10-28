# Phase 2 Manager Instructions

These instructions govern all Phase 2 implementation and experiments.

## User authority and approvals

- Follow the user's explicit instructions. Explain a better alternative and obtain approval before adopting it.
- Ask before making research or implementation decisions not already authorized. Stop and ask when an important ambiguity affects experimental validity.
- Do not silently change the research question, conditions, metrics, split, model, or preprocessing.
- Never delete, overwrite, rename, move, or clean up existing files without explicit permission. Save new results with versioned names.
- Do not install unrelated packages or make unnecessary system-level changes.
- If a request conflicts with the research protocol, explain the exact conflict and ask before proceeding.
- For C2, present the question-only correction inventory to the user before applying any proposed correction. Do not use answers, references, qrels, or test data to construct it. If there are no defensible entries, ask whether to proceed with empty C2 or prepare an inventory.

## Transparency

Before each substantial stage, report intended actions, relevant technical/research decisions, files to read and why, files to create or modify, assumptions, dependencies, and any decision requiring approval. Maintain a record of completed work and deviations.

## Notebook-first development

- Experimental implementation and analysis live in Jupyter notebooks; do not use standalone Python experiment scripts or hide logic in helper modules.
- Environment setup commands may run in the terminal.
- Every notebook uses the `NLP` kernel (`Python (NLP)`).
- Comment important cells and non-obvious operations. Keep cells small and inspectable.
- Execute notebooks in the documented order from clean kernels and verify expected outputs. Do not proceed beyond a failed critical validation.

## Required session summary

At the end of each substantial session, state what and why, files accessed/created/modified, notebook execution status, verified outputs, unresolved issues, limitations, and next step. Do not call an experiment complete until its notebook ran and expected outputs were checked.
