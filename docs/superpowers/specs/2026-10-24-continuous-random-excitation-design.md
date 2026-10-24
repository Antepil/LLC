# Continuous Random Excitation Dataset Design

## Intent

Build a formal LLC training-data configuration that follows the paper's pattern of periodically refreshing sinusoidal-excitation parameters while keeping the excitation waveform continuous. Preserve the existing constant-condition generators and the six-trajectory fixed-sinusoid screening configuration.

## Goals

- Generate 24 closed-loop trajectories with a fixed 2.304 ohm load.
- Run each trajectory for 205 ms: 5 ms startup followed by ten 20 ms excitation segments.
- Smooth each parameter update over the first 2 ms of its segment.
- Keep excitation value and first derivative continuous at every segment boundary.
- Use deterministic per-trajectory random seeds and save every realized segment parameter.
- Keep the neural-network contract at four states, four applied gates, and Vin. Save Vref only as a control/audit signal.
- Produce approximately 54.12 million state-grid points at the existing 11 MHz fixed-grid rate.

## Non-goals

- Do not generate or commit the full dataset as part of this change.
- Do not change the shared Simulink power-stage topology or controller tuning.
- Do not add load excitation or feed Vref into the neural network.
- Do not remove or rewrite the fixed-sinusoid screening workflow.

## Dataset layout

The formal configuration contains 24 trajectories:

| Excitation group | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Vin only | 4 | 1 | 1 | 6 |
| Vref only | 4 | 1 | 1 | 6 |
| Vin and Vref | 8 | 2 | 2 | 12 |
| Total | 16 | 4 | 4 | 24 |

Each trajectory has a unique stored integer seed. Random draws use a local stream so dataset construction does not alter MATLAB's global random state.

## Excitation definition

For either Vin or Vref,

$$
u(t)=C(t)+A(t)\sin\theta(t),
$$

$$
\theta(t)=\phi_0+2\pi\int_{t_0}^{t} f(\tau)\,\mathrm{d}\tau.
$$

At a segment boundary, each parameter $p\in\{C,A,f\}$ moves from its old value to its newly sampled target using

$$
p(t)=p_\mathrm{old}+\left(p_\mathrm{new}-p_\mathrm{old}\right)w(s),
$$

$$
w(s)=3s^2-2s^3,\qquad s=\frac{t-t_j}{T_\mathrm{blend}}.
$$

Because $w'(0)=w'(1)=0$ and phase is obtained by integrating the blended frequency, both $u(t)$ and $\dot u(t)$ are continuous. The first segment transitions from zero amplitude and the nominal center value, so excitation also enters continuously after startup.

The target ranges are:

| Signal | Center | Peak amplitude | Frequency choices |
|---|---:|---:|---:|
| Vin | 380--420 V | 10--20 V | 100, 300, 1000 Hz |
| Vref | 45.6--50.4 V | 1.2--2.4 V | 100, 300, 1000 Hz |

Inactive signals remain at 400 V for Vin or 48 V for Vref with zero amplitude. Vin and Vref use independent draws in the combined group.

## Architecture

The existing sinusoidal dataset generator remains the external-excitation entry point. A focused builder converts a formal-condition seed and randomization ranges into a fully realized segmented excitation structure. The shared evaluator supports the existing fixed-sinusoid structure and the new segmented structure. The shared Simulink model continues to call that evaluator.

The formal JSON configuration has its own schema version and output directory. Generated formal trajectories have their own trajectory schema, include the dataset split, seed, realized segment table, and the unchanged network-input contract. Formal files use MAT v7 because MATLAB's HDF5-based v7.3 writer cannot create or reopen files on the repository's WSL UNC path; each trajectory remains well below the v7 per-variable size limit.

The shared initializer must allow the old event-driven open-loop model to run without an excitation object. Closed-loop runs still require either an explicit excitation or enough constant Vin/Vref control data to construct one.

## Validation

- Unit tests verify deterministic draws, configured ranges, group behavior, and preservation of the global random stream.
- Evaluator tests verify value and numerical first-derivative continuity around every update boundary.
- Configuration tests verify the 16/4/4 split, group counts, timing, unique seeds, and expected fixed-grid point count.
- A shortened simulation smoke test exercises the formal schema through the shared Simulink model and checks saved metadata and the nine-input network contract.
- Existing screening and project tests remain green.

## Approaches considered

1. Blend complete old and new sinusoidal waveforms. This can keep values continuous when carefully initialized, but generally does not preserve the derivative and makes phase behavior difficult to audit.
2. Reset phase to solve value continuity at every update. This does not generally preserve the derivative and introduces artificial phase decisions.
3. Smooth center, amplitude, and frequency while integrating frequency into a continuous phase. This is selected because it directly enforces the stated continuity requirement and stores parameters with clear physical meaning.
