---
description: "Use when writing or editing docs/**/*.rst. Covers Sphinx/reStructuredText conventions, cross-references, toctree hygiene, and RocketPy unit/reference-frame documentation requirements."
name: "Sphinx RST Conventions"
applyTo: "docs/**/*.rst"
---
# Sphinx and RST Guidelines

- Follow existing heading hierarchy and style in the target document.
- Prefer linking to existing documentation pages instead of duplicating content.
- Use Sphinx cross-references where appropriate (`:class:`, `:func:`, `:mod:`, `:doc:`, `:ref:`).
- Keep API names and module paths consistent with current code exports.
- Document physical units and coordinate/reference-frame conventions explicitly.
- Include concise, practical examples when introducing new user-facing behavior.
- Keep prose clear and technical; avoid marketing language in development/reference docs.
- When adding a new page, update the relevant `toctree` so it appears in navigation.
- Use RocketPy docs build workflow:
  - `make build-docs` from repository root for normal validation.
  - If stale artifacts appear, clean docs build outputs via `cd docs && make clean`, then rebuild.
- Treat new Sphinx warnings/errors as issues to fix or explicitly call out in review notes.
- Keep `docs/index.rst` section structure coherent with user, development, reference, technical, and
  examples navigation.
- Do not edit Sphinx-generated scaffolding files unless explicitly requested:
  - `docs/Makefile`
  - `docs/make.bat`
- For API docs, ensure references remain aligned with exported/public objects and current module paths.

See:
- `docs/index.rst`
- `docs/development/build_docs.rst`
- `docs/development/style_guide.rst`
- `docs/reference/index.rst`
- `docs/technical/index.rst`
