# Longitudinal treatment-response prediction in treated Hashimoto thyroiditis

This repository contains the analysis code and reproducibility materials for a longitudinal clinical prediction study in treated Hashimoto thyroiditis.

## Study overview

The locked longitudinal cohort contains:

- **458 patients**
- **1,022 exact longitudinal transitions**
- **564 consecutive within-patient transition pairs**
- **329 patients with at least 2 transitions**
- **39 locked model features**
- **20,000 patient-cluster bootstrap replicates**

The primary analysis uses **patient-grouped out-of-fold (OOF) prediction** so that all visits from a given patient remain within the same fold.

The primary locked endpoint is:

`outcome_log_excess_reduction`

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── .gitignore
├── environment.yml
├── requirements.txt
├── CODE_AVAILABILITY.md
├── DATA_AVAILABILITY.md
├── docs/
│   └── analysis_workflow.md
├── scripts/
│   ├── v35_4_clinical_benchmark_trajectory_decoupling.py
│   ├── run_v35_4_windows.ps1
│   └── README.md
├── data/
│   └── README.md
├── results/
│   ├── README.md
│   └── v35_4_locked_summary.csv
├── manifests/
│   └── locked_analysis_manifest_public.json
└── figures/
    └── README.md
```

## Main reproducibility analyses

The current public analysis script reproduces the V35.4 clinical benchmark and trajectory-decoupling analyses, including:

1. final locked ML performance on the 1,022-transition cohort;
2. TSH-only clinical benchmark using patient-grouped OOF prediction;
3. paired patient-cluster bootstrap comparisons;
4. direct next-visit prediction within consecutive transitions;
5. raw change-change association, reported as descriptive because of the shared current-response term;
6. innovation correlation after removal of the current-response effect; and
7. patient-cluster-robust cross-lag regression.

### Locked V35.4 results

Final locked ML, all 1,022 transitions:

- R² = **0.378221**
- RMSE = **0.755986**
- MAE = **0.509698**
- Pearson r = **0.615550**
- Spearman ρ = **0.484154**

Incremental value over TSH-only linear prediction:

- ΔR² = **0.148373** (95% CI 0.102228–0.193021)
- ΔRMSE = **0.085378** (95% CI 0.056441–0.113470)
- ΔMAE = **0.021858** (95% CI 0.001824–0.041779)
- ΔPearson r = **0.135601** (95% CI 0.090429–0.174665)
- ΔSpearman ρ = **0.102750** (95% CI 0.060257–0.144118)

Trajectory-decoupling analysis, 564 consecutive transitions from 329 patients:

- direct next-visit Pearson r = **0.580797** (95% CI 0.485349–0.666854)
- innovation Pearson r = **0.417059** (95% CI 0.318857–0.513924)
- cross-lag coefficient for predicted next response: **β = 0.901097** (95% CI 0.665502–1.136691; P = 6.56×10⁻14)
- cross-lag coefficient for current response: **β = −0.070091** (95% CI −0.188579–0.048397; P = 0.246)

## Analysis safeguards

The locked workflow follows these pre-specified safeguards:

- no synthetic dates;
- no automatic time transformation;
- no first-observed-as-onset assumption;
- no model output used as ground truth;
- patient-grouped validation;
- chronological ordering preserved; and
- no patient-ID, target, or future-information leakage.

## Data access

**No individual-level clinical data are included in this repository.**

The underlying clinical data contain protected health information and cannot be publicly released through GitHub. See `DATA_AVAILABILITY.md`.

## Reproduction

A compatible Python environment can be created using:

```bash
conda env create -f environment.yml
conda activate hashimoto-longitudinal
```

On Windows, the V35.4 analysis can be launched with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_v35_4_windows.ps1
```

The analysis script expects the authorized local clinical-analysis files to remain outside the Git repository.

## Citation

Please cite the associated manuscript when using this code. A `CITATION.cff` file is included for GitHub citation support.

## License

No open-source license is assigned in this scaffold. Select an institutional/IP-approved license before making the repository public.
