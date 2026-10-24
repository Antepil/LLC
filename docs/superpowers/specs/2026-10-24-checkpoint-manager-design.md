# Checkpoint Manager Design

## Intent

Restore the checkpoint configuration that was present in the legacy
`configs/base.yaml` and implement the checkpoint layer needed by a future
training loop. The current repository has no Python training loop, data loader,
or evaluator, so this change provides a strict standalone API rather than
inventing those missing components.

## Goals

- Restore the original Top-K, last-checkpoint, and snapshot settings.
- Add an explicit project-relative checkpoint directory.
- Save enough state for deterministic training continuation: model, optimizer,
  optional scheduler, epoch, global step, metrics, relevant configuration, and
  Python/PyTorch random-number generator states.
- Maintain deterministic Top-K retention and a separate `last.ckpt`.
- Load either an explicit checkpoint path or `last.ckpt`, with strict state and
  compatibility validation.
- Prevent partial checkpoint and manifest files through atomic replacement.
- Cover configuration, retention, save/load, error handling, and RNG restoration
  with CPU-only unit tests.

## Non-goals

- Implementing data loading, evaluation, a training loop, or automatic resume at
  process startup.
- Defining or computing `test_mean_score`; callers supply recorded metrics.
- Implementing workspace snapshots. `save_last_snapshot: true` is rejected until
  snapshot contents and restoration semantics are designed.
- Supporting multiple processes writing the same checkpoint directory
  concurrently.
- Loading untrusted checkpoint files. PyTorch checkpoint files use pickle-based
  serialization and are only accepted from trusted project runs.

## Configuration

`configs/base.yaml` regains the original block with one explicit path:

```yaml
checkpoint:
  directory: artifacts/checkpoints
  topk:
    monitor_key: test_mean_score
    mode: max
    k: 5
    format_str: 'epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'
  save_last_ckpt: true
  save_last_snapshot: false
```

The directory is resolved against a required project-root argument, never the
caller's current working directory. Configuration validation rejects missing or
unknown keys, invalid types, nonpositive `k`, unsupported modes, unsafe filename
templates, and `save_last_snapshot: true`.

## Public API and payload

Add `llc_dynamics.support.checkpointing` with:

- `CheckpointConfig.from_mapping(...)` for strict configuration parsing and path
  resolution.
- `CheckpointManager(config)` for saving, retention, and loading.
- `save(...)`, requiring a model, optimizer, epoch, global step, metrics, and the
  complete run configuration; a scheduler is optional.
- `load(path, ...)` and `load_last(...)`, which restore supplied model and
  optimizer objects, restore a scheduler when one was saved, optionally restore
  RNG state, and return the saved epoch, global step, metrics, and run config.

The serialized payload has an explicit schema version. Loading rejects an
unsupported version, malformed payloads, missing optimizer state, scheduler
mismatches, non-finite metrics, model state incompatibility, and differences in
the model/normalization configuration contract. Device relocation is explicit
through `map_location`; incompatibilities are not silently ignored.

## Retention and files

Every save validates that `monitor_key` exists and is finite. A qualifying
checkpoint is written using `format_str`; `last.ckpt` is updated independently
when enabled. Top-K ranking uses the configured `min`/`max` direction, then
prefers greater epoch and global step when monitored values tie. Once more than
`k` ranked files exist, the worst entry is deleted.

`index.json` records ranked filenames and ranking metadata so a restarted
manager does not depend on parsing filenames. Checkpoint files and the manifest
are written to sibling temporary files, flushed, and atomically replaced.
Unsafe paths and filename collisions with a different saved step are errors.

## Approaches considered

1. **Standalone `CheckpointManager` with a manifest — selected.** It gives the
   future trainer a complete, testable boundary without choosing a training
   framework prematurely.
2. Stateless save/load functions that reconstruct ranking from filenames. This
   is smaller but brittle when templates change and cannot reliably recover
   arbitrary metric precision.
3. A third-party training framework callback. This supplies more features than
   needed while binding a still-undesigned training architecture and adding
   dependencies.

## Documentation and verification

The root README will describe the restored configuration, trusted-file rule,
manager API boundary, and the fact that automatic resume remains future trainer
work. Tests will first demonstrate failures for invalid configuration, Top-K
ordering and ties, `last.ckpt`, state/RNG round trips, strict compatibility,
missing/non-finite metrics, snapshot rejection, and restart from `index.json`.
The full Python unit suite will run on the existing GPU environment before the
branch is merged to `master`.
