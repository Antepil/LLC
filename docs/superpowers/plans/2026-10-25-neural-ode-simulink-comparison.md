# Neural ODE Simulink Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable Simulink model that runs the reference LLC circuit and Neural ODE in parallel with independent PI/PFM controllers, shared excitation, cold/warm initialization, and directly logged waveform differences.

**Architecture:** Derive `llc_neural_ode_comparison.slx` from the existing closed-loop circuit model, then add a continuous-state Level-2 MATLAB S-Function Neural ODE branch and a second PFM controller. Keep trained weights outside the `.slx`; a MATLAB runner injects the validated network, initial states, controller state, excitation, and solver configuration for each run.

**Tech Stack:** MATLAB R2025a, Simulink, Simscape Electrical, Level-2 MATLAB S-Functions, `matlab.unittest`, JSON validation configuration.

**Spec:** `docs/superpowers/specs/2026-10-25-neural-ode-simulink-comparison-design.md`

## Global Constraints

- Preserve the existing reference `.slx` files and the existing `llc_run_neural_ode_validation` backend.
- Use `ode23t` and the existing solver configuration; do not introduce hidden solver defaults.
- Keep checkpoint/model artifacts out of Git and reject placeholder Neural ODE runtime data at simulation time.
- Fixed load remains exactly `2.304 Ω`.
- Cold and warm startup must both work; warm startup clones the reference handoff state into the Neural ODE and restores complete, independent controller states.
- All missing, malformed, nonfinite, uncovered, or mismatched data must fail explicitly with `llc:*` identifiers.

## Review Focus

- A warm controller restored at local `t=0` must continue the already elapsed reference ramp instead of restarting it; Task 1 adds a continuity test.
- A scheduled gate/control event exactly at handoff must remain due at local `t=0`, while a future event must not be consumed early; Task 1 tests both boundaries.
- The saved `.slx` must not silently simulate with placeholder weights; Task 2 tests explicit rejection.
- Neural PFM feedback must come only from Neural ODE `v_o`, not the reference voltage sensor; Task 3 checks line sources programmatically.
- Simulink logs must cover every requested output time without extrapolation or NaN masking; Task 4 tests shortened and incomplete coverage.

---

### Task 1: Warm-safe controller runtime state

**Files:**
- Create: `matlab/validation/llc_validate_closed_loop_controller_state.m`
- Create: `matlab/validation/llc_shift_closed_loop_controller_state.m`
- Modify: `matlab/validation/llc_initialize_closed_loop_controller.m`
- Modify: `matlab/validation/llc_advance_closed_loop_controller.m`
- Modify: `matlab/validation/llc_reconstruct_closed_loop_controller.m`
- Modify: `matlab/modulation/llc_closed_loop_pfm.m`
- Modify: `matlab/tests/test_llc_neural_ode_validation.m`

**Interfaces:**
- Consumes: existing control structures and reconstructed controller states.
- Produces: `llc_validate_closed_loop_controller_state(state, control)` and `localState = llc_shift_closed_loop_controller_state(state, handoffTimeS, toleranceS)`; every controller state includes `reference_ramp_elapsed_time_s`.

- [ ] **Step 1: Write failing controller-state tests**

Add tests named `testControllerRampElapsedTimeSurvivesLocalTimeShift`, `testControllerShiftKeepsBoundaryEventDue`, `testControllerShiftKeepsFutureEventFuture`, and `testControllerStateValidationRejectsIncompleteOrNonfiniteState`. Assert that warm local time preserves the handoff reference level and that all scheduled times are shifted exactly once.

- [ ] **Step 2: Run the focused tests and verify RED**

Run the named procedures from `matlab/tests/test_llc_neural_ode_validation.m`. Expected: failure because the new field/functions do not exist.

- [ ] **Step 3: Implement controller state validation, ramp elapsed time, and local-time shifting**

Cold initialization sets `reference_ramp_elapsed_time_s = 0`; every advance accumulates elapsed physical time before computing the ramp. The shift helper sets `last_time_s = 0`, subtracts handoff from finite future event fields, preserves `Inf`, and retains ramp elapsed time.

- [ ] **Step 4: Make the PFM S-Function accept optional `control.initial_state`**

Keep one dialog parameter and all existing cold behavior. If `initial_state` exists, validate it and populate every DWork value, including ramp elapsed time and the next hit.

- [ ] **Step 5: Run controller and existing Neural ODE validation tests**

Run `runtests('matlab/tests/test_llc_neural_ode_validation.m')`. Expected: all pass.

- [ ] **Step 6: Commit**

Commit message: `feat: support warm controller state in Simulink`

### Task 2: Continuous Neural ODE S-Function

**Files:**
- Create: `matlab/validation/llc_neural_ode_sfun.m`
- Create: `matlab/tests/test_llc_neural_ode_simulink_comparison.m`

**Interfaces:**
- Consumes: `llc_neural_runtime.model`, `llc_neural_runtime.initial_state`, a 5-vector `[Q1 Q2 Q3 Q4 Vin]`, and existing `llc_neural_ode_forward` scaling conventions.
- Produces: a four-continuous-state Level-2 S-Function whose output is `[i_Lr v_Cr i_Lm v_o]`.

- [ ] **Step 1: Write a failing temporary-harness test**

Create a temporary Simulink harness in the test, inject a constant-derivative fixture network, simulate with `ode23t`, and assert the four states match the analytic linear trajectory at requested times. Add separate tests for placeholder runtime, malformed initial state, invalid gates, and nonfinite Vin.

- [ ] **Step 2: Run the new test file and verify RED**

Run `runtests('matlab/tests/test_llc_neural_ode_simulink_comparison.m')`. Expected: failure because `llc_neural_ode_sfun` is missing.

- [ ] **Step 3: Implement `llc_neural_ode_sfun`**

Use four continuous states, one five-element input, one four-element output, continuous sample time, explicit runtime validation, and the existing normalized network forward function. Reject `runtime.is_placeholder == true` during initialization.

- [ ] **Step 4: Run the new tests and verify GREEN**

Expected: analytic trajectory and all rejection tests pass without warnings from the implementation.

- [ ] **Step 5: Commit**

Commit message: `feat: add continuous Neural ODE Simulink block`

### Task 3: Programmatic dual-branch model builder

**Files:**
- Create: `matlab/models/build_llc_neural_ode_comparison_model.m`
- Create: `matlab/models/llc_neural_ode_comparison.slx`
- Modify: `matlab/tests/test_llc_neural_ode_simulink_comparison.m`

**Interfaces:**
- Consumes: `llc_full_bridge_closed_loop.slx`, `llc_closed_loop_pfm`, `llc_neural_ode_sfun`, and model-workspace variables `llc_control`, `llc_excitation`, `llc_neural_runtime`.
- Produces: `modelFile = build_llc_neural_ode_comparison_model(modelDirectory)` and stable block/log names defined by the design.

- [ ] **Step 1: Write failing structural tests**

Generate into a temporary directory and assert the Neural ODE, independent Neural PFM, input/state muxes, error calculation, Scopes, and all four Neural log blocks exist. Inspect port line handles to prove Neural PFM feedback originates from the Neural state selector and shares only `Vref`/`Vin` with the reference branch.

- [ ] **Step 2: Run structural tests and verify RED**

Expected: failure because the builder is missing.

- [ ] **Step 3: Implement the idempotent builder**

Copy the reference closed-loop model, add the Neural branch in a separate visual region with annotations, create placeholder runtime data tagged `is_placeholder = true`, wire all signals, and save without modifying the source model.

- [ ] **Step 4: Generate and inspect the committed model artifact**

Run the builder against `matlab/models`, reopen the generated `.slx`, verify block diagram update succeeds without simulation, and confirm the source model hash is unchanged.

- [ ] **Step 5: Run structural tests and verify GREEN**

Expected: all structural and source-preservation assertions pass.

- [ ] **Step 6: Commit**

Commit message: `feat: build dual-branch Neural ODE Simulink model`

### Task 4: Simulink comparison simulation API

**Files:**
- Create: `matlab/validation/llc_simulate_neural_ode_comparison.m`
- Modify: `matlab/tests/test_llc_neural_ode_simulink_comparison.m`

**Interfaces:**
- Consumes: validated Neural ODE model, hardware, comparison `.slx`, reference and Neural initial states, one controller state cloned internally, output times, Vin/Vref signals, control, and solver.
- Produces: a result struct containing common `times_s`, `reference_states`, `predicted_states`, `state_error`, both gate logs, both control logs, and final controller evidence.

- [ ] **Step 1: Write failing cold and warm simulation tests**

Use a short duration and fixture network. Assert finite outputs, exact requested output times, independent gate/control logs, `state_error == reference_states - predicted_states`, and warm `t=0` states/controller outputs equal the injected handoff values. Add explicit missing-log and insufficient-coverage tests.

- [ ] **Step 2: Run the tests and verify RED**

Expected: failure because the simulation API is missing.

- [ ] **Step 3: Implement input validation and `SimulationInput` injection**

Set hardware/solver/excitation overrides, physical component initial states, `llc_neural_runtime`, and separate copies of the warm controller configuration. Run the model, validate every log, and interpolate only within verified coverage using linear states and previous-value gates/control.

- [ ] **Step 4: Run comparison and regression tests**

Run the new test file plus `test_llc_neural_ode_validation.m`. Expected: all pass.

- [ ] **Step 5: Commit**

Commit message: `feat: simulate Neural ODE comparison model`

### Task 5: Configured cold/warm entry point

**Files:**
- Create: `matlab/validation/llc_run_neural_ode_simulink_comparison.m`
- Modify: `configs/matlab/neural_ode_validation.json`
- Modify: `matlab/validation/llc_load_neural_ode_validation_config.m`
- Modify: `matlab/tests/test_llc_neural_ode_validation.m`
- Modify: `matlab/tests/test_llc_neural_ode_simulink_comparison.m`

**Interfaces:**
- Consumes: `paths.simulink_comparison_model_file`, an enabled independent closed-loop `experiment_id`, existing model/hardware/control/steady-state/sampling configuration, and Task 4's simulation API.
- Produces: `[result, context] = llc_run_neural_ode_simulink_comparison(configPath, experimentId)` for cold or warm experiments.

- [ ] **Step 1: Write failing config and cold-entry tests**

Assert the new model path is mandatory, resolved relative to the project root, and must exist. Run a short cold fixture config and assert the selected experiment and result schema.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: failure because the path and entry point are unsupported.

- [ ] **Step 3: Implement config resolution and cold orchestration**

Select exactly one enabled `independent_closed_loop` experiment by ID, load the model export/hardware, create cold initial states, build the fixed output grid, and invoke Task 4.

- [ ] **Step 4: Write failing warm-entry tests**

Use a short reference preheat fixture and assert mandatory steady-state assessment, reference-state cloning, full controller reconstruction, local-time shifting, and no reference-ramp restart.

- [ ] **Step 5: Implement warm orchestration**

Run reference preheat with enough post-handoff scheduler coverage, assess settling, reconstruct at handoff, shift the controller state, clone reference state into both evaluation branches, and invoke Task 4 with evaluation-local signals.

- [ ] **Step 6: Run entry-point and regression tests**

Run both MATLAB test files. Expected: all pass.

- [ ] **Step 7: Commit**

Commit message: `feat: run configured Simulink Neural ODE comparisons`

### Task 6: Documentation, screenshot, and final verification

**Files:**
- Modify: `matlab/README.md`
- Modify: `README.md`
- Create outside repository: `/tmp/llc_neural_ode_comparison.png`

**Interfaces:**
- Consumes: the completed builder, saved `.slx`, and configured runner.
- Produces: user-facing run instructions and an actual exported Simulink diagram screenshot.

- [ ] **Step 1: Write documentation for build, injection, cold/warm execution, outputs, and limitations**

Clearly distinguish the Simulink comparison backend from the existing formal MATLAB validation backend and state that real waveform accuracy still requires a real exported checkpoint.

- [ ] **Step 2: Rebuild, open, fit, and export the actual model diagram**

Export `/tmp/llc_neural_ode_comparison.png`; visually inspect that both branches, both PFM blocks, state comparison, and logs are legible.

- [ ] **Step 3: Run static checks and complete test suites**

Run `checkcode` on new/modified `.m` files, all MATLAB tests in manageable batches, and `PYTHONPATH=src .../.venv-gpu/bin/python -m unittest discover -s tests`. Expected: all pass; report environmental warnings separately.

- [ ] **Step 4: Review diff and request whole-branch code review**

Run `git diff --check`, inspect binary/model and source changes, then obtain an independent Standards and Spec review.

- [ ] **Step 5: Commit and merge according to repository workflow**

Commit message: `docs: document Neural ODE Simulink comparison`; after all checks and review pass, merge the task branch into `master`, verify the merged critical suites, and preserve unrelated untracked files.

