# ApolloX2

ApolloX2 is a GPU-accelerated toolkit for constructing fixed-site lattice models for multicomponent alloy systems. It optimizes atomic occupations on a fixed lattice using target short-range order (SRO) data and can export optimized structures for VASP workflows.

## Installation With micromamba

```bash
micromamba env create -f env.yaml
micromamba activate pyhea_env
pip install .
```

For development:

```bash
pip install -e .
```

Check the installation:

```bash
apollox2 --version
```

## Requirements

- Python 3.9+
- micromamba
- CUDA-capable GPU
- CUDA Toolkit with `nvcc`
- CMake 3.12+
- C++ compiler

## Quick Start

ApolloX2 currently supports GPU execution. Make sure CUDA is available:

```bash
nvcc --version
```

Run the global search:

```bash
cd example
apollox2 global config.yaml
```

Run the detail search:

```bash
apollox2 detail config.yaml
```

## Configuration

Example `config.yaml`:

```yaml
device: gpu
solutions: 200
total_iter: 10
parallel_task: 256
converge_depth: 10
weight:
  - 1.0
structure: POSCAR
target_sro: sro
cut_iter: 9
cutoff:
  - 5
mc_num_lattice_per_batch: 3
mc_batch_num: 1
mc_num_tasks: 256
mc_search_depth: 10
mc_fitness_threshold: 1
mc_initial_temperature: 0.01
mc_cooling_rate: 0.9
mc_annealing_steps: 10
```

## Configuration Parameters

### `device`

Computation device.

```yaml
device: gpu
```

ApolloX2 currently supports GPU execution.

### `solutions`

Number of candidate lattice configurations used during the global search.

```yaml
solutions: 200
```

Larger values explore more candidates but require more GPU memory and time.

### `total_iter`

Number of global optimization iterations.

```yaml
total_iter: 10
```

Increasing this value gives the global search more chances to improve the result.

### `parallel_task`

Number of parallel GPU search tasks used during global optimization.

```yaml
parallel_task: 256
```

Larger values may improve GPU utilization but can increase memory usage.

### `converge_depth`

Number of recent iterations used to judge convergence.

```yaml
converge_depth: 10
```

If the result does not improve over this depth, the search may be considered converged.

### `weight`

Fitness weights for SRO shells.

```yaml
weight:
  - 1.0
```

Each value corresponds to one SRO shell. More values include more shells in the fitness calculation.

Example:

```yaml
weight:
  - 1.0
  - 0.5
  - 0.25
```

### `structure`

Path to the input structure file.

```yaml
structure: POSCAR
```

Usually this is a VASP POSCAR file. ApolloX2 keeps the lattice fixed and optimizes the atomic occupations.

### `target_sro`

Path to the target SRO file.

```yaml
target_sro: sro
```

ApolloX2 optimizes the lattice occupations to match these target SRO values.

### `cut_iter`

Iteration cutoff used during the global search stage.

```yaml
cut_iter: 9
```

This controls filtering or cutoff behavior during global optimization.

### `cutoff`

Neighbor cutoff values used for SRO and neighbor calculations.

```yaml
cutoff:
  - 5
```

The cutoff determines which neighboring atoms are included in SRO-related calculations.

### `mc_num_lattice_per_batch`

Number of lattice configurations sampled in each Monte Carlo batch.

```yaml
mc_num_lattice_per_batch: 3
```

Larger values explore more configurations per batch.

### `mc_batch_num`

Number of Monte Carlo batches in the detail search.

```yaml
mc_batch_num: 1
```

Increasing this value runs more refinement batches.

### `mc_num_tasks`

Number of parallel GPU tasks used during Monte Carlo search.

```yaml
mc_num_tasks: 256
```

This controls GPU parallelism during the detail search stage.

### `mc_search_depth`

Search depth for each Monte Carlo task.

```yaml
mc_search_depth: 10
```

Larger values allow each task to explore more trial moves.

### `mc_fitness_threshold`

Fitness threshold used during detail search.

```yaml
mc_fitness_threshold: 1
```

This value can be used to decide whether a candidate structure is good enough during refinement.

### `mc_initial_temperature`

Initial temperature for simulated annealing.

```yaml
mc_initial_temperature: 0.01
```

Higher values allow more uphill moves early in the search.

### `mc_cooling_rate`

Cooling factor for simulated annealing.

```yaml
mc_cooling_rate: 0.9
```

Values closer to `1.0` cool more slowly. Smaller values cool faster.

### `mc_annealing_steps`

Number of annealing steps in each Monte Carlo search.

```yaml
mc_annealing_steps: 10
```

Increasing this value gives the detail search more refinement steps.

## Exporting Best Structures

Use `generate_vasp.py` to export the best structures from `evolution_data.txt`:

```bash
python generate_vasp.py config.yaml evolution_data.txt -n 100 --outdir best_structures --outcsv top_results.csv
```

This generates ranked `.vasp` files and a summary CSV.

## License

This project is licensed under the LGPL-3.0 License. See `LICENSE` for details.
