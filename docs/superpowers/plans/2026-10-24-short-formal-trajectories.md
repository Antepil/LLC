# Short Formal Excitation Trajectories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 24 long formal excitation trajectories with 120 independently seeded 45 ms trajectories while retaining 240 total excitation segments and the 11 MHz fixed grid.

**Architecture:** Keep the existing generator and schema. Change only the formal configuration, its contract test, and the published dataset-scale documentation. Each trajectory contains a 5 ms startup and two 20 ms continuous-random segments.

**Tech Stack:** MATLAB R2025b tests, JSON configuration, Markdown documentation.

**Spec:** `docs/superpowers/specs/2026-10-24-continuous-random-excitation-design.md`

## Global Constraints

- Fixed load remains 2.304 ohm.
- Fixed-grid sampling remains 11 MHz.
- Use 80 train, 20 validation, and 20 test trajectories.
- Within those splits use Vin-only/Vref-only/both counts of 20/20/40, 5/5/10, and 5/5/10.
- Every trajectory has two 20 ms excitation segments after a 5 ms startup.
- All 120 trajectory IDs and random seeds must be unique.
- Generated data and benchmark artifacts remain untracked.

## Review Focus

- The split/group cross-counts must match the approved 80/20/20 table exactly.
- The stop time must equal startup plus both segments exactly.
- The total excitation-segment count must remain 240.
- The total state-grid point count must be 59,400,120 at 11 MHz.
- The shortened configuration must still pass the existing formal generator smoke test.

---

### Task 1: Pin and implement the formal dataset scale

**Files:**
- Modify: `matlab/tests/test_llc_sinusoidal_excitation.m`
- Modify: `configs/matlab/sinusoidal_excitation/continuous_random_data_generation.json`
- Modify: `docs/superpowers/specs/2026-10-24-continuous-random-excitation-design.md`
- Modify: `docs/superpowers/plans/2026-10-24-continuous-random-excitation.md`
- Modify: `matlab/README.md`

**Interfaces:**
- Consumes: existing `llc-continuous-random-excitation-data-generation-v1` JSON schema.
- Produces: 120 deterministic formal conditions consumed unchanged by `generate_llc_sinusoidal_excitation_dataset`.

- [x] **Step 1: Change the configuration-scale test to assert the approved timing, cross-counts, 120 unique IDs/seeds, 240 segments, and 59,400,120 points.**

- [x] **Step 2: Run the focused test and verify it fails against the 24-trajectory configuration.**

Run: MATLAB `testFormalContinuousRandomConfigurationMatchesScale`

Expected: FAIL because the old stop time, segment count, condition count, split counts, and point count remain present.

- [x] **Step 3: Generate the 120 JSON condition objects with stable IDs and disjoint seed ranges for every split/group cell.**

- [x] **Step 4: Update the spec, original implementation plan, and MATLAB README to state 120 × 45 ms, the approved cross-count table, 240 segments, and 59,400,120 points.**

- [x] **Step 5: Run the focused configuration test and formal shortened smoke test.**

Expected: both PASS.

- [x] **Step 6: Run MATLAB checkcode, `git diff --check`, and the full MATLAB test suite.**

- [ ] **Step 7: Commit, merge into `master`, rerun the focused configuration test on merged `master`, then start the resumable detached formal generation job.**
