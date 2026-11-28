<!-- Navigation aid and current status. Detailed history lives in the evidence
     directories and in git; do not accumulate day-by-day slice diaries here.
     Rewritten 2026-11-28 to remove ~2000 lines of historical slice records. -->

# AI Context

## Start here

This repository carries **two independent mainlines**. Decide which one a request targets first.

| Mainline | Entry | Branch |
|---|---|---|
| Runtime / case-configuration refactor (R1) | [runtime architecture](source/runtime_architecture.rst) + [runtime roadmap](plans/runtime-refactoring-roadmap.md) | `hexin` |
| **DCU accelerator mainline (3D Euler device residency)** | [development plan](plans/oneflow-development-plan.md) + [Living TODO](plans/oneflow-development-todo.md) | `dev`; `master` kept identical to upstream |

**Accelerator objective: accelerate while preserving correctness.** Establish the numerical,
physical and state/communication contracts, pass the applicable correctness gates, then
demonstrate reproducible end-to-end speedup on a comparable basis. Task ordering and timing
targets are subordinate to this objective; see the development plan §0.

Reading order for an accelerator task:
1. `AGENTS.md` — branch model, gate order, measurement discipline, workspace rules.
2. `docs/plans/oneflow-development-todo.md` — the living handoff: current status, active slice, evidence.
3. `docs/handoff/oneflow-current-handoff.md` — fixed paths/constants (§1–§3), task pointer (§4), traps (§5).
4. `docs/DEVELOPMENT_PRINCIPLES.md` — durable engineering rules and regression lessons.
5. The relevant architecture page, roadmap, focused tests and CI evidence for the exact commit.

Do not start by reading the whole repository, and do not trust a handoff note over the current
source and tests. A plan describes intent, not proof that a feature or test exists.

## Current accelerator status (2026-11-28)

- **Branch/HEAD**: `dev` at `7b97d2b52`, pushed to `origin/dev`. Local/origin/upstream `master`
  all at `701234a71`; `dev` has merged master and is clean.
- **B.4.1 boundary residual**: L1–L3 pass on **1 zone / 1 rank**. L3 full-dev 200 steps
  (`129048798`) residual `3.2e-20`, aero `0`, all three device-path flags yes. See Living TODO.
- **8 zone / 8 rank full-dev**: numerically failing, bisected to the **device gradient** chain.
  Device flux alone passes (`129083701` residual `3.4e-21`); full host passes (`129082789`
  residual `0`); both device paths on fails (`129056842` residual `2.0e-6`). A forced-download
  hypothesis was tried, disproved (residual worsened to `1.4e-4`), and reverted.
  **Do not report an 8:1 speedup until this is fixed.** The `3.6e-2` aero difference at 8 zones
  also appears in the pure-host arm (partition accumulation order), so it is not a device defect.
- **Performance caliber**: the 1-zone `2.42×` is 1 core vs 1 DCU and is **not comparable** with
  the 18.5× inviscid 8:1 figure. LU-SGS is the largest HIP-side stage (~53%) and remains host code.
- **Next action**: locate the multi-zone device-gradient defect (`HipFluxBackend::CalcGradient`
  ghost-gradient source), fix it, then re-run 8 zone 20 steps to a numeric pass before any 8:1
  performance claim.

## R1 status (hexin)

R1 case configuration and lifecycle work lives on `hexin`; see
[routing and evidence in the runtime roadmap](plans/runtime-refactoring-roadmap.md). This page no
longer tracks its per-slice state — consult that roadmap and the exact-commit CI evidence.

## Repository constraints

- Repository: [`eric2003/OneFLOW`](https://github.com/eric2003/OneFLOW); fork `lql341/OneFLOW` as `origin`.
- Keep changes merge-compatible; do not push work to `master` without an explicit PR authorization.
- Prefer small, evidence-backed changes; trace call paths and state/resource ownership before changing interfaces.
- Keep C++ source files ASCII-only. Never claim a test or CI result without evidence for the exact revision.
- Distinguish implemented, partial, planned and unverified work in all docs.

## Fixed cluster facts (see handoff §1–§3 for the full tables)

- Access: `scnet-hpc` skill; OpenAPI is the default submit backend, SSH (`kseshell`, `zzeshell`)
  is the fallback when OpenAPI is unavailable. Renewed SSH keys 2026-11-27 expire **2026-12-27**.
- Workspace: `$HOME/oneflow/` with fixed `src/ build/app/ fixtures/ evidence/ snapshots/ scripts/ logs/`.
- Source transfer: `git archive` → upload to `snapshots/` → job unpacks to `src/OneFLOW`.
- Gates: the Kunshan scripts in `ci/kunshan/`; run the CPU five-case suite and the HIP contract
  test before claiming a numerical-kernel change validated.

## Status vocabulary

Use these labels consistently: **Verified** (named test/CI evidence), **Implemented, unverified**
(code exists, no required evidence), **Partial** (acceptance criterion open), **Planned**,
**Blocked** (named missing capability), **Deferred** (postponed with reason).
