# Held-out synthetic-spot benchmark for spatial deconvolution

This repository holds method and code only. The full description is in [index.md](index.md), which is also the GitHub Pages home page.

- `code/resolvability.py` builds held-out synthetic Visium spots and runs DestVI on them.
- `code/score.py` scores the predictions against the known truth and writes the decision table.
- `code/run_example.sh` is an example driver. `code/requirements.txt` lists the packages.
- `tests/make_toy_inputs.py` makes fake tables so you can check that the scorer runs.
