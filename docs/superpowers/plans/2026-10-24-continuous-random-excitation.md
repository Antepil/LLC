# Continuous Random Excitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, continuously joined, randomly refreshed Vin/Vref dataset configuration with about 54.12 million fixed-grid points while preserving existing generators.

**Architecture:** A standalone builder realizes seeded segment targets, and the existing excitation evaluator gains a backward-compatible segmented profile. The independent sinusoidal generator dispatches by configuration schema and records formal-dataset metadata without changing the shared Simulink model.

**Tech Stack:** MATLAB R2025b, Simulink/Simscape, MATLAB Unit Test, JSON configuration

**Spec:** `docs/superpowers/specs/2026-10-24-continuous-random-excitation-design.md`

## Global Constraints

- Fixed load is 2.304 ohm.
- Network inputs remain four states, four applied gates, and Vin; Vref is excluded.
- Preserve the existing fixed-condition and fixed-sinusoid workflows.
- Formal data files use MAT v7 and are not committed; v7.3 is unavailable on the repository's WSL UNC path.
- Each formal trajectory is 205 ms with 5 ms startup, ten 20 ms segments, and 2 ms transitions.

## Review Focus

- Boundary times just before, at, and after a segment update must produce a continuous value and first derivative.
- Inactive signals in single-input groups must remain exactly constant.
- Rebuilding a trajectory with the same seed must produce identical realized parameters without changing MATLAB's global random stream.
- Malformed timing, range, group, seed, and dataset-split inputs must fail explicitly.
- Existing screening and event-driven open-loop generation must retain their previous behavior.

---

### Task 1: Restore initializer isolation

**Files:**
- Modify: `matlab/models/llc_initialize_model.m`
- Modify: `matlab/tests/test_llc_data_generation.m`

**Interfaces:**
- Consumes: event-driven overrides containing `parameters`, `events`, and `solver`
- Produces: initializer behavior that does not require `control` or `excitation` for event-driven models

- [ ] **Step 1: Write the failing regression test**

Add `testOpenLoopGeneratorDoesNotRequireExcitation` using a temporary one-condition, short-stop configuration and assert that `generate_llc_dataset` writes one valid trajectory.

- [ ] **Step 2: Run the focused test and verify RED**

Run the MATLAB test selection for `testOpenLoopGeneratorDoesNotRequireExcitation`.

Expected: FAIL with `llc:InvalidSimulationConfig` because the event override has no excitation.

- [ ] **Step 3: Allow event overrides to skip excitation assignment**

Modify `llc_initialize_model(model)` so explicit excitation and closed-loop constant excitation are assigned as before, event-driven overrides proceed without `llc_excitation`, and other incomplete overrides still error.

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: PASS.

- [ ] **Step 5: Commit**

Commit the initializer fix and regression test.

### Task 2: Build and evaluate continuous random segments

**Files:**
- Create: `matlab/simulation/sinusoidal_excitation/llc_build_continuous_random_excitation.m`
- Modify: `matlab/simulation/sinusoidal_excitation/llc_evaluate_sinusoidal_excitation.m`
- Modify: `matlab/tests/test_llc_sinusoidal_excitation.m`

**Interfaces:**
- Consumes: `llc_build_continuous_random_excitation(randomization, timing, condition)` where condition has `excitation_group` and `random_seed`
- Produces: a scalar excitation structure with `profile_type`, nominal signals, `transition_time_s`, and realized `segments`; evaluator returns an N-by-2 Vin/Vref matrix for arbitrary nonnegative time values

- [ ] **Step 1: Write failing builder tests**

Add tests asserting repeatability, unique range-valid segment targets, exact inactive signals, and unchanged global RNG state.

- [ ] **Step 2: Run builder tests and verify RED**

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement the deterministic builder**

Use a local `RandStream('mt19937ar', 'Seed', condition.random_seed)`. Create ten target segments at configured starts, independent Vin/Vref draws for the combined group, and random initial phases only for active signals.

- [ ] **Step 4: Run builder tests and verify GREEN**

Expected: PASS.

- [ ] **Step 5: Write failing evaluator continuity tests**

Evaluate immediately around update boundaries and assert value continuity and matching numerical one-sided derivatives for both channels; retain the fixed-sinusoid envelope test.

- [ ] **Step 6: Run evaluator tests and verify RED**

Expected: FAIL because the evaluator does not understand the segmented profile.

- [ ] **Step 7: Implement parameter blending and analytic phase integration**

Dispatch by `profile_type`. For each segment interval use cubic smoothstep for center, amplitude, and frequency, and the analytic integral of the blended frequency for phase. Validate starts, timing, targets, and profile shape explicitly.

- [ ] **Step 8: Run sinusoidal unit tests and verify GREEN**

Expected: all non-simulation tests in `test_llc_sinusoidal_excitation` pass.

- [ ] **Step 9: Commit**

Commit the builder, evaluator, and unit tests.

### Task 3: Add the formal 24-trajectory dataset

**Files:**
- Create: `configs/matlab/sinusoidal_excitation/continuous_random_data_generation.json`
- Modify: `matlab/simulation/sinusoidal_excitation/generate_llc_sinusoidal_excitation_dataset.m`
- Modify: `matlab/tests/test_llc_sinusoidal_excitation.m`
- Modify: `matlab/README.md`

**Interfaces:**
- Consumes: formal schema `llc-continuous-random-excitation-data-generation-v1`
- Produces: 24 seeded conditions, deterministic realized excitation, formal trajectory schema and split/seed metadata, MAT v7 files

- [ ] **Step 1: Write failing formal-configuration tests**

Assert 24 trajectories, 16/4/4 split, 6/6/12 group counts, unique valid seeds, exact timing/ranges, and 54,120,024 total state-grid points including endpoints.

- [ ] **Step 2: Run configuration tests and verify RED**

Expected: FAIL because the formal configuration does not exist.

- [ ] **Step 3: Add the formal configuration**

Create the 24 explicit conditions and shared randomization/timing settings. Use output directory `data/raw/llc_sinusoidal_excitation/continuous_random`.

- [ ] **Step 4: Extend generator validation and dispatch**

Accept the existing screening schema unchanged and the new formal schema. Build formal excitations through the new builder, choose the matching output schema, save dataset split/seed/realized segments, validate expected sample count, and use `-v7` for formal files.

- [ ] **Step 5: Add and run a shortened formal smoke test**

Reduce a loaded formal configuration to one segment and one condition in a temporary JSON file. Assert simulation success, formal schema, split/seed metadata, realized segments, continuous excitation, and the unchanged network contract.

Expected: PASS.

- [ ] **Step 6: Document operation and scale**

Describe the formal command, split, timing, parameter ranges, point-count calculation, deterministic realization, output location, and the fact that full generation is intentionally separate from tests.

- [ ] **Step 7: Run all MATLAB tests**

Run `runtests('matlab/tests')` and `assertSuccess`.

Expected: all tests pass with zero failures and zero incomplete tests.

- [ ] **Step 8: Commit**

Commit the formal configuration, generator integration, tests, and documentation.
