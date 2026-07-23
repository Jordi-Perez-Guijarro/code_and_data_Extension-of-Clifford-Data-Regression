# Extension of Clifford Data Regression Methods for Quantum Error Mitigation

This repository contains the code and data used in the paper:

> J. Pérez-Guijarro, A. Pagès-Zamora, and J. R. Fonollosa, "Extension of Clifford Data Regression Methods for Quantum Error Mitigation," *IEEE Transactions on Quantum Engineering*, vol. 7, 2026.
>
> Preprint: [arXiv:2411.16653](https://arxiv.org/abs/2411.16653)

The work studies two extensions of Clifford Data Regression (CDR), a supervised-learning-based quantum error mitigation technique: one that uses multiple copies of the original circuit, and one that adds a layer of single-qubit rotations. The scripts and data here reproduce the numerical experiments and figures reported in the paper.

If you use this code or data, please cite the paper above (see [Citation](#citation)).

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Data](#data)
  - [Important Remark on Variable Names](#important-remark-on-variable-names)
  - [Figure 1](#figure-1)
  - [Figures 2 and 3a](#figures-2-and-3a)
  - [Figure 3b](#figure-3b)
  - [Figure 5a](#figure-5a)
  - [Figure 5b](#figure-5b)
- [Scripts](#scripts)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Repository Structure

```
.
├── DATA_paper_CDR/        # All data used to generate the paper's figures, organized by figure
├── library_QEM.py         # Core functions imported by all other scripts
├── script_figure_*.py     # One script per figure, for data generation and/or plotting
└── README.md
```

## Data

All data used to produce the paper's figures is stored in the `DATA_paper_CDR` folder, organized into one subfolder per figure.

### Important Remark on Variable Names

The stored data was generated with earlier versions of the scripts. As a result, some variable names have since changed — for example, a variable now called `J` may appear in the data files as `R`, or under a similar older name. The data itself is consistent between old and current versions, so this naming mismatch does not affect plot generation.

### Figure 1

- **Data files:** `N_1K_MERGED_0_1000.spydata` and `N_infty_MERGED_0_1000.spydata` contain the final data, stored in the variable `loss_circuits`.
- **Auxiliary files:** The remaining files in this subfolder are intermediate outputs from parallel threads used during the numerical experiments; they were later combined into the two merged files above.
- **Note:** Opening this data may require an older version of Spyder (e.g., 4.2).

### Figures 2 and 3a

- **Data file:** `All_data_figure_2_3a.spydata` contains all data needed for both figures.
- **Auxiliary files:** As with Figure 1, additional files come from parallel-thread runs and are not needed directly.

### Figure 3b

- Results are split by the number of shots, `N`. For example, `data_N_100K.spydata` stores errors for `N = 100K`.
- **Exception:** For the ZNE-insertion method with `J_2 = 6`, results are stored separately in `data_N_100K_J_2_6.spydata`.
- The final values across different numbers of shots are assembled in `script_figure_3b_plot.py`.

### Figure 5a

- `N_1k_QFT.spydata` — data for `N = 1k`.
- `N_inf_QFT.spydata` — data for `N → ∞`.

### Figure 5b

- Results are split by noise level `p`. For example, `p_inf_0_01.spydata` contains estimates for `p = 0.01` with `N = ∞` shots.
- **Files containing `_non_corrected` in the name** hold results for the non-corrected method. These had to be generated separately because the original version of `script_figure_5b.py` did not yet support the non-corrected option.

## Scripts

- **`library_QEM.py`** — Core module with the main functions, imported by all other scripts.
- **`script_figure_*.py`** — One script per figure, used either to generate the underlying data or to produce the corresponding plot. Each script is commented to be self-explanatory.

## Requirements

- Python 3 with standard scientific packages (NumPy, Matplotlib, etc.)
- [Spyder](https://www.spyder-ide.org/) to open `.spydata` files. For **Figure 1** specifically, an older version (e.g., 4.2) may be required for the data to load correctly.

## Citation

If you use this code or data, please cite:

```bibtex
@article{perez2026extension,
  title   = {Extension of Clifford Data Regression Methods for Quantum Error Mitigation},
  author  = {P{\'e}rez-Guijarro, Jordi and Pag{\`e}s-Zamora, Alba and Fonollosa, Javier R.},
  journal = {IEEE Transactions on Quantum Engineering},
  volume  = {7},
  year    = {2026},
  note    = {Preprint available at arXiv:2411.16653}
}
```
