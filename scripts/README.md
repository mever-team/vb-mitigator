# Reproducibility scripts

This directory holds scripts used to **reproduce results from publications**.
They are intentionally kept separate from the user-facing API (`vbm-train`,
`vbm-eval`, `vbm-ui`) because they encode paper-specific experiment grids.

Unlike the previous iteration of this repo, cluster-specific launchers
(SLURM/HUA jobs, one-off sweeps, and temporary debugging scripts) are **not**
included here — the supported way to run experiments is the CLI and the configs
under [`../configs`](../configs).

## Layout

Each script is a thin loop over `vbm-train` invocations with the relevant
configs. See [`benchmark_utkface.sh`](benchmark_utkface.sh) for a template you
can copy for a new dataset/benchmark.

## Adding a reproducibility script

1. Add the configs your experiment needs under `configs/<dataset>/<method>/`.
2. Write a shell script here that loops over them with `vbm-train --cfg ...`.
3. Document what paper/table it reproduces at the top of the script.
