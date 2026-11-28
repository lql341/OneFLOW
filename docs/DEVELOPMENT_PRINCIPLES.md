# OneFLOW Development Principles

These principles are the repository's durable engineering defaults. They are intended
to prevent previously solved defects from being reintroduced during new features,
refactoring, test maintenance, or AI-assisted development.

They are defaults, not excuses to ignore a concrete requirement. If a change must
deviate, explain why, keep the exception narrow, and add evidence for the behavior.

## 1. Preserve solved knowledge

- Before changing a subsystem, inspect its existing helpers, tests, architecture notes,
  and known regression history. Do not implement a second local solution before checking
  whether the repository already has a shared one.
- A bug fix is not complete when the immediate symptom disappears. Identify the underlying
  contract and add a regression test at the narrowest useful boundary.
- When a fix reveals a reusable lesson, update this document or the owning architecture
  page in the same logical change. Record the symptom, root cause, durable rule, and
  regression test. Do not rely on chat history or a commit message as the only record.
- When a workflow graph is built from declarative case definitions, post-run reporting and artifact-lineage collection must consume that same canonical definition list (or the typed plan). Keep report aggregation in a focused helper so the reporting contract can be tested independently of process execution.
- Prefer updating an existing authoritative document over creating overlapping notes.
  Keep chronological handoff notes concise; do not turn them into the only place where
  permanent rules live.

## 2. Keep the source tree clean and read-only during execution

- Treat checked-in source and case-definition directories as immutable inputs during
  builds, tests, GUI operations, and runtime execution.
- Put generated meshes, logs, reports, temporary files, caches, and run results in an
  explicit build or user workspace. On Windows, the normal user workspace is under
  `%USERPROFILE%\\.oneflow`; on other platforms, use an appropriate user data/cache
  location or a caller-supplied workspace.
- Use the platform's normal temporary-directory facility (for example Python
  `tempfile.TemporaryDirectory()`) for disposable data. Never move temporary data into
  the repository merely to keep it on the same drive as a source path.
- Do not assume a temporary directory and the repository are on the same filesystem or
  drive. If a tool requires same-filesystem staging, create the staging directory under
  the designated external workspace or destination filesystem, and document that
  requirement. Do not silently fall back to writing in the source tree.
- Tests must create and clean their own temporary workspaces. A test must not depend on
  stale output from a previous run.

## 3. Use portable path semantics

- Prefer cross-platform standard-library path APIs (`pathlib`, `os.path`, and
  `tempfile`) over platform-specific path manipulation.
- Do not compare paths as raw strings, use string-prefix checks for containment, or
  assume that `.resolve()` makes two aliases textually identical.
- When both paths are expected to exist and the question is whether they identify the
  same filesystem object, use `os.path.samefile(a, b)`. This handles filesystem aliases
  such as Windows 8.3 short names and symbolic links where supported.
- For normalized path policy, containment, and paths that may not yet exist, use the
  established path utility in the owning layer (currently
  `tools/oneflow_gui/path_utils.py` for Workbench/project operations). Reuse its
  `paths_equal`, `is_path_within`, and `relative_path_within` semantics instead of
  reimplementing them.
- Do not make core grid/runtime code depend on a GUI package just to reuse a helper.
  Keep utilities in the layer that owns the behavior; move one to a neutral shared
  package only when multiple independent layers genuinely need the same contract and
  the dependency direction remains valid.
- Isolate unavoidable platform-specific code behind a small, tested compatibility
  function. Do not scatter Windows API calls or OS-specific branches through business
  logic.

### Path regression to remember

A Windows CI test once compared a generated path using the normal profile path with an
expected path expressed through an 8.3 short-name alias. Both paths referred to the same
file, but direct `Path` equality (and normalization alone) did not express filesystem
identity. The test now uses `os.path.samefile`. Preserve that distinction: path spelling,
normalized path policy, containment, and filesystem identity are different questions.

A documentation-root rename (the historical `doc` directory became `docs`) once changed
the visible file links but left the directory-level path in the site-publish workflow as
a `cd` into the old name. The deploy job failed before Sphinx ran, and nothing else
noticed.
Directory-level paths live in more places than file-level links: CI `cd`/`sphinx-build`
arguments, Read the Docs configuration, the docs `Makefile`, and contract JSON that
declares documentation paths. When a directory moves, fix all of them in one change and
run `scripts/check-doc-paths.py`; it flags both a surviving reference to the old root and
a reference that does not resolve. A link that only fails at read time is not covered by
a build-only check.

## 4. Verify text and serialization boundaries

- Distinguish an actual control character from its escaped textual spelling. For
  example, Python `"\n"` is a line-feed, while `"\\n"` is the two visible
  characters backslash and `n`. When joining display lines, use the actual newline
  delimiter unless the output format explicitly requires escaped text.
- Check the final representation at the consumer boundary: plain-text widgets,
  Markdown renderers, HTML views, JSON/YAML serializers, terminal output, and log
  viewers do not interpret escape sequences in the same way. Do not assume that
  syntactically valid source produces readable rendered output.
- For multiline UI or report content, inspect section boundaries, blank lines,
  indentation, headings, and list markers. Choose a widget appropriate to the
  intended format (plain text versus rich text) rather than relying on accidental
  rendering behavior.
- Add a regression assertion against the actual displayed/exported value. For a
  newline contract, assert that expected sections are separated by real newline
  characters and that unintended literal escape sequences are absent.
- During review, inspect string escaping at both the source-code and rendered-output
  levels, especially when code is generated, templated, serialized, or edited through
  JSON/API payloads.

### Text-formatting regression to remember

The Geometry Library details were joined with the two-character literal `\\n`
instead of a real line-feed. The dialog therefore displayed visible escape sequences
and collapsed the intended multiline structure. The fix uses a real newline delimiter;
the GUI test checks both the expected newline boundary and absence of the literal
`\\nDefinition` sequence. Preserve this distinction whenever text crosses a
serialization or UI-rendering boundary.

## 5. Respect module ownership and dependency direction

- Trace the real entry point, callers, ownership, and cleanup path before changing an
  interface.
- Keep the C++ runtime independent of Python and the Workbench. Python may orchestrate
  the runtime, but must not become a hidden runtime dependency or duplicate numerical
  implementation.
- Keep GUI presentation separate from domain logic. Prefer Qt-independent services and
  data models where practical.
- Reuse an existing abstraction when its contract fits. Add a new shared abstraction only
  when at least two real consumers need the same stable behavior; avoid speculative
  frameworks and duplicate helpers.
- Keep platform-specific behavior at a narrow boundary and expose portable semantics to
  callers.

## 6. Keep examples and tests honest

- Each example should demonstrate one clearly named concept at an appropriate level of
  complexity. Do not label a multi-cell or multi-feature example as a single-cell case.
- Add complexity progressively. A more complicated topology, boundary arrangement, or
  physical model should be a separate example with a name that states what it exercises.
- A passing process exit code is not automatically proof that outputs, mesh topology,
  numerical values, or physical behavior are correct. Assert the actual contract.
- Keep the repository's C++ `tests/` tree C++-only. Place Python tests in their
  designated Python test area and keep the inventories and CI registration aligned.
- Separate unit, integration, regression, numerical, and platform-specific evidence.
  Report exactly what was run; do not infer a pass from an unrelated suite or an older CI
  run.

### Self-check regression to remember

A device-versus-host viscous face-flux self-check reported zero differing faces while the
flow field contained NaN. The check classified a face with
`relative = |a - b| / scale > tolerance`; every comparison with NaN is false, so
non-finite values took the "agrees" branch and the diagnostic silently certified the
broken arm. Non-finite values are now counted separately and always fail the check.

Two durable rules follow from this, and they apply to any comparison that gates a
numerical claim (device versus host, reference versus candidate, file versus file):

1. A comparison must classify non-finite inputs before doing arithmetic on them. A
   relative- or absolute-error test alone cannot detect NaN or Inf.
2. A diagnostic label must come from the predicate the solver itself uses. Marking every
   face below `nBFaces` as "wall" also labelled inlet, outlet and interface faces;
   wall means `bcType == SOLID_SURFACE`.

Guard: `CompareViscousFaceFlux` in `codes/uns/include/ViscousFluxSelfCheck.h` with
`tests/solver/viscous_flux_selfcheck_test.cpp`. Keep the comparison logic out of the
solver loop so the rules can be tested without a mesh, a solver, or a device.

## 7. Make each change a complete, reviewable slice

- Prefer a coherent functional slice: implementation, discriminating tests, directly
  related documentation, test inventory/CI registration, and evidence as applicable.
- Do not split one behavior change into arbitrary file-by-file commits. Do not combine
  unrelated changes merely to reduce commit count.
- Preserve existing behavior unless an intentional change is documented and tested.
- Avoid hard-coded build or test file lists when the project intentionally uses a
  discovery or glob mechanism; follow the established convention for that subsystem.
- Keep source files within the repository's encoding/style constraints, including the
  C++ ASCII-source requirement.

## 8. Required pre-change and post-change review

Before editing:

1. Confirm the target branch and current HEAD; for ongoing development, use the latest
   `hexin` unless the user explicitly specifies otherwise.
2. Read this document, `AGENTS.md`, the current `docs/AI_CONTEXT.md` handoff, and the
   owning architecture/contract page relevant to the change.
3. Search for existing helpers and regression tests before adding new path, process,
   filesystem, UI, mesh, or workflow logic.
4. State the behavior contract and the evidence that will demonstrate it.

After editing:

1. Add a regression test for the root cause, not only the visible symptom.
2. Run the narrowest relevant tests first, then broader tests appropriate to the change.
3. Check that no generated artifacts or temporary files are written into the source tree.
4. Review displayed and serialized text for actual versus escaped newlines, indentation,
   section spacing, and format-appropriate rendering; test the consumer-visible value.
5. Update the owning principle/architecture/experience record when the change teaches a
   reusable lesson.
6. Report the exact commit, test commands and outcomes, CI status for that exact commit,
   and anything not verified. Never claim success without evidence.

## 9. When to record an engineering lesson

Add or update a durable lesson when any of these occurs:

- a previously fixed defect is reintroduced;
- a platform-specific behavior causes a cross-platform regression;
- two modules independently implement the same policy;
- a test passed while failing to check the intended contract;
- a source-tree, workspace, process-lifecycle, or cleanup boundary is violated;
- an example's name or documentation misrepresents what it actually tests;
- a recurring review comment reveals a missing repository-wide convention.

Use this compact format in the owning document:

| Field | What to record |
|---|---|
| Symptom | What failed or was misleading |
| Root cause | The underlying semantic or architectural mistake |
| Rule | The reusable rule that prevents recurrence |
| Guard | The test, static check, or review step that enforces it |
| Scope | The modules or workflows to which it applies |

Do not record every transient command failure. Record the generalizable lesson, link the
fixing commit or test where useful, and update the rule if later evidence changes it.

### Recorded lesson: bisect a device-path failure before accepting a plausible root cause

| Field | What to record |
|---|---|
| Symptom | On the multi-zone viscous flagship the residual froze at its step-2 value for 20 steps while the CPU arm converged normally (gate residual 2.0e-6 / aero 3.6e-2). The fastest explanation — the gradient was never downloaded to the host — fit the first timer (`gradient_download=0`). A one-line fix forced the download and added four passing unit tests, yet the next 20-step run was worse: residual 1.4e-4 and divergence starting at step 1. |
| Root cause | The first hypothesis was never discriminated against its alternative. A three-way bisection on the same grid showed: device flux alone passes (residual 3.4e-21, job `129083701`), full host passes (residual 0, job `129082789`), and device gradient on fails in both the original and "fixed" code. The defect is in the device gradient kernel on a multi-zone field, not in the host download; forcing the download only exposed the wrong gradient earlier. The tempting fix was reverted. |
| Rule | Treat a plausible root cause as a hypothesis until a discriminating experiment separates it from the next candidate. When changing a device/host hand-off, require the numeric result to improve before calling it fixed; a positive kernel timer and a green truth-table unit test prove the path was taken, not that the values are correct. Crossed configurations (viscous × multi-zone) need their own evidence. |
| Guard | Keep `parse-xb-device-path.py` as the authoritative path-shape report (`DEVICE_FLUX_ON`/`DEVICE_GRADIENT_ON`/`TIMING_USABLE`), but do not encode an unvalidated hand-off rule into it. Record bisection jobs with their residual verdicts in the handoff so the next reader sees which arm passed. |
| Scope | Any accelerator defect with more than one plausible cause; especially device gradient / interface paths where a timer can confirm execution without confirming correctness. |

### Recorded lesson: reject a failed measurement before interpreting it

| Field | What to record |
|---|---|
| Symptom | A 200-step flagship run was reported with an absolute path timer of 0 for every category and `HIPLOG_MISSING`; earlier attempts were interpreted as OOM, private-memory exhaustion and a compiler-flag problem before the real cause — a shell comment inside a backslash continuation that swallowed every `VAR=` — was found. |
| Root cause | Measurement drivers were trusted as evidence without first proving that the launched process received its configuration and produced output. The gate itself also called `sys.exit(1)` on a numeric mismatch before emitting the device-path report, so the most diagnostic artifact was missing exactly when it was needed. |
| Rule | A run with no output, a zeroed timer table, or a missing backend banner is not a failed performance run; it is a failed measurement. Diagnose the harness before the solver. A gate must always emit its configuration and path-shape report, even when the numeric comparison fails. |
| Guard | `ci/kunshan/f2-main-solver-cross-backend-residual.sh` now records `numeric_rc` and prints `DEVICE_PATH_REPORT` on both pass and fail; `parse-xb-device-path.py` exits 2 on missing logs or mixed device/host paths. |
| Scope | Cluster gate scripts, paired A/B drivers, and any workflow where a shell wrapper assembles the runtime environment. |


## 10. Practical text-output review checklist

Use this checklist whenever a change builds, joins, serializes, exports, or renders
human-readable text. It applies to GUI details, run reports, diagnostics, generated
documentation, and command-line summaries.

- [ ] Identify the output contract first: plain text, rich text/Markdown, HTML, or
  serialized data. Do not mix the escaping rules of one format with another.
- [ ] Inspect the final value at the consumer boundary (for example,
  `QPlainTextEdit.toPlainText()`, the exported file contents, or captured terminal
  output), not only the source literal or intermediate data structure.
- [ ] Verify actual line breaks and section boundaries. For Python, `"\\n"` is a
  line-feed delimiter; `"\\\\n"` is the visible two-character escape spelling.
- [ ] Check blank lines between sections, indentation, heading underlines, list
  markers, and whether long lines remain readable.
- [ ] Check that no unintended escaped control sequences (such as visible
  `\\n`, `\\t`, or `\\r`) leak into human-readable output. Escaped spellings
  are valid only when the output format explicitly requires them.
- [ ] For JSON/YAML or other machine-readable output, parse the serialized result
  back and assert the semantic value; do not judge correctness from visual appearance
  alone.
- [ ] Add regression assertions for both the required representation and the
  forbidden representation. Prefer checking the public widget/export result over
  private formatting helpers.
- [ ] Review representative empty, single-section, and multi-section cases when
  those states are supported.

A review is not complete merely because the source compiles or the serializer emits
syntactically valid data. The rendered or round-tripped consumer value must satisfy
the declared output contract.
