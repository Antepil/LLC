# Checkpoint Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the checkpoint configuration and add a strict standalone PyTorch checkpoint manager with deterministic Top-K retention, `last.ckpt`, and complete training-state restoration.

**Architecture:** A frozen configuration object validates the `checkpoint` mapping and resolves its directory from an explicit project root. A stateful manager serializes versioned trusted-run payloads with PyTorch, tracks ranked files in an atomic JSON manifest, and strictly restores supplied training objects without owning a training loop.

**Tech Stack:** Python 3.11+, PyTorch 2.14.1, standard-library `dataclasses`, `json`, `pathlib`, `random`, `tempfile`, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-24-checkpoint-manager-design.md`

## Global Constraints

- Restore the legacy values `test_mean_score`, `max`, `5`, the original filename template, `save_last_ckpt: true`, and `save_last_snapshot: false`.
- Resolve `artifacts/checkpoints` against an explicit project root, never the process working directory.
- Do not add a YAML parser, training loop, evaluator, data loader, or third-party training framework.
- Reject missing, illegal, incompatible, or non-finite data explicitly; do not substitute defaults.
- Treat checkpoint files as trusted local artifacts because PyTorch serialization is pickle-based.
- Preserve existing changes in the user's primary checkout; all work stays on `codex/checkpoint-manager` until verified.

## Review Focus

- A filename template that renders an absolute path or `..` traversal must be rejected by the configuration tests in Task 1.
- Boolean values passed where integer epoch/global-step/Top-K values are expected must be rejected by tests in Tasks 1 and 2.
- A manifest referencing missing files or containing malformed JSON must fail explicitly in Task 2 rather than silently resetting retention state.
- A save failure must leave the previous `last.ckpt` and manifest usable, exercised by atomic-write failure tests in Task 2.
- Loading a scheduler state without a supplied scheduler, or supplying a scheduler when none was saved, must fail in Task 3.

---

### Task 1: Strict checkpoint configuration

**Files:**
- Create: `src/llc_dynamics/support/__init__.py`
- Create: `src/llc_dynamics/support/checkpointing.py`
- Create: `tests/unit/test_checkpointing.py`
- Modify: `configs/base.yaml`

**Interfaces:**
- Consumes: a root configuration `Mapping[str, Any]` and explicit `project_root: Path`.
- Produces: `TopKConfig`, `CheckpointConfig`, and `CheckpointConfig.from_mapping(root_config, project_root=...) -> CheckpointConfig`.

- [ ] **Step 1: Write failing configuration tests**

Add `CheckpointConfigTests` covering the exact restored values and resolved path, missing/unknown keys, `k <= 0`, boolean `k`, invalid mode, empty monitor key, unsafe/absolute/render-failing filename templates, non-boolean save flags, and snapshot enablement.

- [ ] **Step 2: Run the configuration tests to verify failure**

Run: `PYTHONPATH=src .venv-gpu/bin/python -m unittest tests.unit.test_checkpointing.CheckpointConfigTests -v` using the primary checkout's existing `.venv-gpu` interpreter.

Expected: FAIL because `llc_dynamics.support.checkpointing` does not exist.

- [ ] **Step 3: Add the exact configuration block**

Append the documented `checkpoint` mapping to `configs/base.yaml`, including `directory: artifacts/checkpoints` and the six legacy values.

- [ ] **Step 4: Implement strict configuration objects**

Implement frozen dataclasses and:

```python
@classmethod
def from_mapping(
    cls,
    root_config: Mapping[str, Any],
    *,
    project_root: Path,
) -> CheckpointConfig: ...
```

Require exact checkpoint/topk keys and types, resolve a relative safe directory beneath the resolved project root, preflight the template with `epoch=0` and the monitor key set to `0.0`, require a plain `.ckpt` basename other than `last.ckpt`, and reject snapshot enablement with `NotImplementedError`.

- [ ] **Step 5: Run configuration tests**

Run the Task 1 test command.

Expected: all `CheckpointConfigTests` pass.

- [ ] **Step 6: Commit the configuration boundary**

```bash
git add configs/base.yaml src/llc_dynamics/support tests/unit/test_checkpointing.py
git commit -m "feat: restore checkpoint configuration"
```

---

### Task 2: Atomic save and deterministic retention

**Files:**
- Modify: `src/llc_dynamics/support/checkpointing.py`
- Modify: `tests/unit/test_checkpointing.py`

**Interfaces:**
- Consumes: `CheckpointConfig` from Task 1; a model, optimizer, optional scheduler, nonnegative epoch/global step, finite metrics, and complete run configuration.
- Produces: `CheckpointManager.__init__(config)`, `CheckpointManager.save(...) -> Path`, ranked `.ckpt` files, `last.ckpt`, and schema-versioned `index.json`.

- [ ] **Step 1: Write failing save/retention tests**

Add `CheckpointSaveTests` for directory creation, complete payload keys, missing/non-finite monitor metrics, invalid epoch/step types and ranges, max-mode ordering, min-mode ordering, ties preferring later epoch then step, nonqualifying saves updating only `last.ckpt`, filename collisions, manager restart from `index.json`, malformed/stale manifests, and `save_last_ckpt: false`.

- [ ] **Step 2: Write an atomic-failure regression test**

Patch the final replace used for a new save to fail and assert the previously readable `last.ckpt` and `index.json` contents are unchanged and temporary files are cleaned up.

- [ ] **Step 3: Run save tests to verify failure**

Run: `PYTHONPATH=src /home/antep/projects/LLC_Neural_ODE_rebuild/.venv-gpu/bin/python -m unittest tests.unit.test_checkpointing.CheckpointSaveTests -v`

Expected: FAIL because `CheckpointManager` is not implemented.

- [ ] **Step 4: Implement the versioned payload and validation**

Implement:

```python
def save(
    self,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    run_config: Mapping[str, Any],
    scheduler: Any | None = None,
) -> Path: ...
```

Use schema version `1`; store model/optimizer/optional scheduler state, counters, a plain metrics dict, a deep-copied run configuration, Python RNG state, CPU PyTorch RNG state, and all CUDA RNG states when CUDA is initialized.

- [ ] **Step 5: Implement Top-K ranking and manifest validation**

Use `(metric direction, epoch, global_step)` so newer counters win ties. Store manifest version `1` plus filename, metric, epoch, and global step for every ranked entry. On startup reject malformed schemas, duplicate filenames, unsafe names, missing files, non-finite metrics, and more than `k` entries.

- [ ] **Step 6: Implement atomic writes and pruning**

Write each file to a same-directory temporary file, flush and `fsync`, then call `os.replace`. Publish checkpoint files before the new manifest; update `last.ckpt` independently; only delete the pruned ranked file after the manifest commits. Clean temporary files on every failure.

- [ ] **Step 7: Run save/retention tests**

Run the Task 2 test command.

Expected: all `CheckpointSaveTests` pass.

- [ ] **Step 8: Commit saving and retention**

```bash
git add src/llc_dynamics/support/checkpointing.py tests/unit/test_checkpointing.py
git commit -m "feat: save ranked training checkpoints"
```

---

### Task 3: Strict restoration and user documentation

**Files:**
- Modify: `src/llc_dynamics/support/checkpointing.py`
- Modify: `src/llc_dynamics/support/__init__.py`
- Modify: `tests/unit/test_checkpointing.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: a trusted checkpoint path or configured `last.ckpt`, destination model/optimizer/optional scheduler, expected run configuration, explicit `map_location`, and `restore_rng` flag.
- Produces: `CheckpointMetadata`, `CheckpointManager.load(...) -> CheckpointMetadata`, and `CheckpointManager.load_last(...) -> CheckpointMetadata`.

- [ ] **Step 1: Write failing round-trip tests**

Add `CheckpointLoadTests` proving model, optimizer, scheduler, epoch, step, metrics, run config, Python RNG, and PyTorch RNG round-trip through an explicit path and `last.ckpt`, including explicit CPU `map_location`.

- [ ] **Step 2: Write failing strictness tests**

Cover missing files, malformed/unsupported payload schema, absent state keys, incompatible model state, optimizer load failure, both scheduler mismatch directions, and model/normalization contract mismatches while permitting unrelated run-config differences.

- [ ] **Step 3: Run load tests to verify failure**

Run: `PYTHONPATH=src /home/antep/projects/LLC_Neural_ODE_rebuild/.venv-gpu/bin/python -m unittest tests.unit.test_checkpointing.CheckpointLoadTests -v`

Expected: FAIL because load APIs and `CheckpointMetadata` do not exist.

- [ ] **Step 4: Implement strict load APIs**

Implement:

```python
def load(
    self,
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_run_config: Mapping[str, Any],
    scheduler: Any | None = None,
    map_location: torch.device | str,
    restore_rng: bool = True,
) -> CheckpointMetadata: ...

def load_last(...same destination arguments...) -> CheckpointMetadata: ...
```

Load only explicit paths, validate the entire payload before mutating destination objects, compare only `model` and `normalization` contract sections, require symmetric scheduler presence, call strict model loading, then restore optimizer/scheduler/RNG state and return immutable metadata.

- [ ] **Step 5: Export the public API and document its boundary**

Export `CheckpointConfig`, `CheckpointManager`, and `CheckpointMetadata` from `llc_dynamics.support`. Update README with configuration location, future-trainer call boundary, explicit/load-last behavior, full state contents, trusted-file warning, Top-K/last semantics, and unsupported snapshots/automatic resume.

- [ ] **Step 6: Run checkpoint tests and the full Python suite**

Run:

```bash
PYTHONPATH=src /home/antep/projects/LLC_Neural_ODE_rebuild/.venv-gpu/bin/python -m unittest tests.unit.test_checkpointing -v
PYTHONPATH=src /home/antep/projects/LLC_Neural_ODE_rebuild/.venv-gpu/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass; no failures or errors.

- [ ] **Step 7: Commit restoration and documentation**

```bash
git add README.md src/llc_dynamics/support tests/unit/test_checkpointing.py
git commit -m "feat: restore checkpoint training state"
```

---

### Task 4: Final verification and integration

**Files:**
- Verify: all task files and commits

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: a reviewed, verified task branch ready to merge into `master`.

- [ ] **Step 1: Inspect branch-only changes**

Run `git diff --check master...HEAD`, `git diff --stat master...HEAD`, and `git status --short --branch`; verify there are no unrelated files or whitespace errors.

- [ ] **Step 2: Run the full suite once more from a clean tree**

Run the full discovery command from Task 3 and record the exact pass/skip counts.

- [ ] **Step 3: Request whole-branch code review**

Review `master...HEAD` against the design and repository conventions. Fix every confirmed issue and rerun affected plus full tests.

- [ ] **Step 4: Merge and verify `master`**

Merge `codex/checkpoint-manager` into `master` without disturbing the user's dirty primary checkout. Verify the resulting commit graph and working-tree status; rerun the full Python suite against merged `master` from a clean checkout.
