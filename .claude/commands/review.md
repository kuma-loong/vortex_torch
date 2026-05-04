---
description: Audit a submission pair against AGENTS.md rules (read-only review).
argument-hint: <submission-name>
---

Invoke the `vortex-submission-reviewer` subagent to audit a
submission pair against the [AI/AGENTS.md](AI/AGENTS.md) contract.

Resolve `$1` to a path: try `submissions/<tag>/$1.{py,json}`
first (the standard agent-tagged location), then
`submissions/$1.{py,json}` as a fallback for top-level examples.

The reviewer will produce a structured report with sections:
**Blockers**, **Warnings**, **Suggestions**, **Summary**. It does
not modify any file — if the user wants a fix, they should follow
up with `/new-submission` or ask explicitly.
