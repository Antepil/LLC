# Closed-loop Steady Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 27 zero-state closed-loop LLC startup trajectories, retain valid unsettled cases, and classify periodic steady state at 2.0–2.5 ms.

**Architecture:** Put the periodic-cycle assessment in a pure MATLAB function driven by state and applied-gate logs, so synthetic tests can verify it without Simulink. Simulate each condition through 2.5 ms once, evaluate 0.1 ms checkpoints, select the first settled checkpoint (or 2.5 ms), and export the fixed-grid prefix with assessment metadata. Process conditions independently and return a run summary so one failed case does not stop the batch.

**Tech Stack:** MATLAB function-based tests, Simulink, JSON configuration.

**Spec:** Confirmed design in the Codex conversation on 2026-10-22.

## Global Constraints

- Four physical states start at zero; controller state is reset for every trajectory.
- Vin is 360/400/440 V, R is 1.8432/2.304/2.7648 ohm, and Vref is 43.2/48/52.8 V; generate the full 27-condition Cartesian product.
- Keep the existing 1 ms reference ramp and common controller parameters.
- Assess at 2.0, 2.1, 2.2, 2.3, 2.4, and 2.5 ms using the preceding 0.4 ms of complete applied switching cycles.
- Output cycle means must all be within 2% of Vref.
- State mean limits use 1% of fixed scales [10 A, 400 V, 10 A, 48 V]. Ripple variation uses 5% of max(mean ripple, 1% of scale).
- Save numerically valid unsettled trajectories; report regulation and dynamic stability independently.
- Preserve the existing uncommitted `configs/base.yaml` change outside this worktree.

## Review Focus

- Missing or malformed gate cycles must produce an explicit non-settled assessment, never a false positive.
- A checkpoint window must exclude partial cycles crossing either window boundary.
- A regulated but drifting state must report regulated=true and dynamics_stable=false.
- A condition that never settles must be exported through 2.5 ms with settled=false.
- A failed condition must leave later conditions runnable and expose its error identifier and message in the returned summary.

---

### Task 1: Periodic steady-state assessor

**Files:**
- Create: `matlab/simulation/llc_assess_periodic_steady_state.m`
- Create: `matlab/tests/test_llc_steady_state_assessment.m`

**Interfaces:**
- Consumes: raw state times/values, gate times/values, Vref, and assessment configuration.
- Produces: `assessment` with per-checkpoint results, `settled`, `settled_by_2ms`, `first_settled_time_s`, `regulated`, `dynamics_stable`, and `selected_stop_time_s`.

- [ ] Write synthetic failing tests for settled periodic data, regulated drifting data, missing complete cycles, and an unsettled final checkpoint.
- [ ] Run the assessment test and confirm failure because the function is missing.
- [ ] Implement actual-cycle segmentation from Q1 rising edges and the agreed mean/ripple checks.
- [ ] Run the assessment test and confirm all cases pass.
- [ ] Commit the assessor and its tests.

### Task 2: Configuration and adaptive-length batch generation

**Files:**
- Modify: `configs/matlab/closed_loop_data_generation.json`
- Modify: `matlab/simulation/generate_llc_closed_loop_dataset.m`
- Modify: `matlab/tests/test_llc_closed_loop.m`

**Interfaces:**
- Consumes: the assessment function from Task 1 and the v3 generation configuration.
- Produces: 27 independently processed trajectory files plus `[files, summary]`, with the fixed grid ending at the selected checkpoint.

- [ ] Write failing configuration and generator-facing tests for 27 combinations, six checkpoints, selected-length metadata, and continue-on-error summary behavior that can be exercised without running Simulink where practical.
- [ ] Run tests and confirm the expected failures against v2 behavior.
- [ ] Add the v3 assessment schema, validate all fields, simulate each condition to 2.5 ms, assess checkpoints, resample to the selected stop, and catch errors per condition.
- [ ] Run focused and full MATLAB tests.
- [ ] Commit configuration, generator, and tests.

### Task 3: Usage documentation and final verification

**Files:**
- Modify: `matlab/README.md`

**Interfaces:**
- Consumes: final generator behavior.
- Produces: documented condition matrix, steady-state semantics, variable trajectory lengths, metadata, and failure reporting.

- [ ] Document the 27-condition matrix, checkpoint algorithm, saved unsettled cases, and `[files, summary]` API.
- [ ] Run JSON parsing, MATLAB Code Analyzer, focused tests, and the full test suite where available.
- [ ] Inspect the complete diff for unrelated changes and commit documentation.
