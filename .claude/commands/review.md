---
description: Audit a submission pair against AGENTS.md rules (read-only review).
argument-hint: <submission-name>
---

Invoke the `vortex-submission-reviewer` subagent to audit
`submissions/$1.py` and `submissions/$1.json` against the
[AI/AGENTS.md](AI/AGENTS.md) contract.

The reviewer will produce a structured report with sections:
**Blockers**, **Warnings**, **Suggestions**, **Summary**. It does
not modify any file — if the user wants a fix, they should follow
up with `/new-submission` or ask explicitly.
