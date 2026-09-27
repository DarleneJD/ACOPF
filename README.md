# ACOPF — autonomous voltage management in MV/LV distribution networks

This repository contains the reproducibility material for a study intended for
submission to *IEEE Transactions on Smart Grid*. It supports two connected
computational workflows:

1. selection of seasonal representative days from 2019 meteorological data;
2. identification, synthesis and chronological evaluation of autonomous local
   voltage-control settings in an unbalanced MV/LV distribution network.

The two Google Colab notebooks are the canonical entry points. The first
creates the representative-day dictionary consumed by the second.

> **Reproducibility status.** The repository currently documents the code and
> data used by the study. Before archival publication, add the manuscript
> citation, the repository release/DOI and the applicable code and data
> licenses.

## Repository structure

```text
ACOPF/
├── data/
│   ├── meteorology/
│   │   ├── profiles_10min/
│   │   │   ├── perfis_10min_2019_01.csv
│   │   │   ├── ...
│   │   │   └── perfis_10min_2019_12.csv
│   │   ├── dicionario_8_dias_representativos.json
│   │   └── manifesto_perfis_10min.json
│   └── network/
│       ├── branches_1.xlsx
│       └── buses_1.xlsx
├── notebooks/
│   ├── DiasRepresentativos.ipynb
│   └── OPF_MVLV_autonomo_v4_9.ipynb
├── src/
│   └── acopf.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated results are written outside the input-data folders and are ignored by
Git.

## Quick start in Google Colab

These links work after this directory structure has been committed to the
`main` branch of `DarleneJD/ACOPF`.

1. [Open `DiasRepresentativos.ipynb` in Colab](https://colab.research.google.com/github/DarleneJD/ACOPF/blob/main/notebooks/DiasRepresentativos.ipynb), then run all cells in order.
2. Inspect the generated representative-day files.
3. [Open `OPF_MVLV_autonomo_v4_9.ipynb` in Colab](https://colab.research.google.com/github/DarleneJD/ACOPF/blob/main/notebooks/OPF_MVLV_autonomo_v4_9.ipynb), edit only the `CFG` cell and run the required campaign.

Both notebooks clone the public repository automatically. No manual upload of
the network spreadsheets, monthly profiles or representative-day JSON is
required. Colab must have internet access, and the repository must remain
public or be cloned with authenticated credentials.

The OPF notebook additionally requires a licensed IBM ILOG CPLEX installation;
see [Solver requirement](#solver-requirement).

## Canonical workflow

### 1. `notebooks/DiasRepresentativos.ipynb`

**Purpose.** Reduce the 365-day meteorological series to two observed days per
season while retaining the associated occurrence frequencies.

**Inputs.** The twelve files matching
`data/meteorology/profiles_10min/perfis_10min_2019_MM.csv`.

**Main operations.**

1. clone or locate the repository;
2. validate the 2019 calendar and the 10-minute time grid;
3. retain days that meet the minimum daily coverage criterion;
4. construct one daily feature vector from irradiance and ambient temperature;
5. standardize the features separately within each season;
6. apply K-means with two clusters per season (`random_state=42`, `n_init=50`);
7. select the observed day closest to each centroid;
8. export the selected days, their frequencies, diagnostic tables and figures.

The notebook was rewritten from the corrected portion of the original
prototype. The preceding experimental cells are not part of this repository.

**Primary output.** `dicionario_8_dias_representativos.json`, which is the
input used by the OPF notebook. The version distributed in
`data/meteorology/` contains:

| Season | Representative dates |
|---|---|
| Summer (`Verao`) | 15/02/2019 and 06/12/2019 |
| Autumn (`Outono`) | 13/05/2019 and 24/03/2019 |
| Winter (`Inverno`) | 17/06/2019 and 13/07/2019 |
| Spring (`Primavera`) | 12/10/2019 and 15/11/2019 |

The notebook writes its Colab results to
`/content/drive/MyDrive/ACOPF/dias_representativos_2019`. After verification, a
new JSON can replace the tracked file in `data/meteorology/`.

### 2. `notebooks/OPF_MVLV_autonomo_v4_9.ipynb`

**Purpose.** Identify implementable local inverter settings and evaluate them
with an autonomous step-voltage regulator (SVR), including chronological
multiday transfer.

**Inputs.**

- `data/network/buses_1.xlsx`;
- `data/network/branches_1.xlsx`;
- `data/meteorology/dicionario_8_dias_representativos.json`;
- the monthly meteorological profiles, including
  `perfis_10min_2019_02.csv` for the current consecutive-week experiment.

**Computational stages.**

| Stage | Operation | Main product |
|---|---|---|
| Data preparation | Parse the phase-explicit MV/LV network and 144-point daily profiles | Validated network and time series |
| E1 identification | BFM–SOCP with continuous tap relaxation | Candidate tap trajectory and training target `Q*` |
| E2 identification | Causal integer projection and conditioned optimization | Discrete identification candidate |
| Physical reconstruction | Fix taps and re-solve with the exact SVR ratio | Physically closed voltage–reactive-power points |
| Device synthesis | Fit implementable local settings | A/B/C assignments and individualized Volt–VAR curves |
| Closed-loop replay | Let each scenario's native LDC determine its own taps | Daily S1–S4 electrical metrics |
| Multiday transfer | Freeze settings and carry relay state across midnight | Chronological taps, DRP, DRC and `FD95` |

`Q*` is an identification target only; it is not dispatched during the final
closed-loop evaluations. Likewise, the final scenarios do not receive the tap
trajectory found during identification. They use the same autonomous,
phase-wise line-drop compensation logic and the exact fixed-tap SVR relation.

The scenarios are:

| Scenario | Inverter operation | Role |
|---|---|---|
| S1 | Unity power factor at all inverters | Autonomous-LDC reference |
| S2 | Uniform capacitive power factor of 0.96 | Uniform-support counterfactual |
| S3-RL | Sparse A/B/C settings selected by contextual reward-based Q-learning | Proposed implementation |
| S3-direct | Direct maximization of the same immediate rewards | Classifier ablation |
| S4 | Individualized OPF-derived Volt–VAR curve at every inverter | Non-sparse comparison |

The notebook embeds the complete model implementation; it does not import
`src/acopf.py`. Its `CFG` cell controls the source date, campaign identifier,
output directory, solver limits, enabled experiments and reproducibility seed.
The default Colab output root is
`/content/drive/MyDrive/ACOPF/public_v4_9`.

## Input-data formats

### Meteorological CSV files

File naming convention: `perfis_10min_2019_MM.csv`, where `MM` is the month
from `01` to `12`.

Each file contains one row every 10 minutes and uses these columns:

| Column | Type/unit | Description |
|---|---|---|
| `timestamp` | ISO-8601 datetime | Physical timestamp in 2019 |
| `source_year_index` | integer | Calendar year; must be `2019` |
| `month` | integer | Month number |
| `day` | integer | Day of month |
| `hour` | integer | Hour from 0 to 23 |
| `minute` | integer | Minute in `{0,10,...,50}` |
| `irradiance_pu` | p.u. | Global irradiance normalized by 1000 W/m² |
| `ambient_temperature_c` | °C | Ambient temperature |
| `panel_temperature_c` | °C | Panel temperature calculated from irradiance and NOCT |
| `generation_pu` | p.u. | Temperature-corrected available PV generation |

A complete non-leap year has 52,560 rows: 144 samples per day for 365 days.
The formulas, interpolation rule, source worksheet metadata and monthly
statistics are recorded in
`data/meteorology/manifesto_perfis_10min.json`.

### Representative-day JSON

The top level is organized by the keys `Verao`, `Outono`, `Inverno` and
`Primavera`. Each season contains two dates in `DD/MM/YYYY` format. Each date
contains exactly 144 records:

```json
{
  "Verao": {
    "15/02/2019": [
      {
        "hora": "00:00",
        "irradiance_pu": 0.0,
        "ambient_temperature_c": 22.4
      }
    ]
  }
}
```

The example is abbreviated; the tracked file contains all seasons, dates and
10-minute samples.

### Network workbook: `buses_1.xlsx`

The OPF loader reads the following worksheets. Column names are part of the
input contract and should not be changed without updating the loader.

| Worksheet | Required columns | Content |
|---|---|---|
| `MT` | `N`, `name`, `zb`, `tb`, `v_nom_kv`, `phases` | MV buses and phase availability |
| `Load_MT` | `name`, `v_nom_kv`, `phases`, `P_D`, `Q_D`, `Conn`, `Model` | MV loads |
| `BT` | `N`, `name`, `zb`, `tb`, `v_nom_kv`, `phases`, `P_D`, `Q_D` | LV buses and loads |
| `PV` | `Name`, `Bus`, `kV`, `kva`, `pf`, `phases`, `p_pv_1`–`p_pv_3`, `q_pv_1`–`q_pv_3` | PV inverter placement and ratings |

### Branch workbook: `branches_1.xlsx`

| Worksheet | Required columns | Content |
|---|---|---|
| `MT` | `l`, `k`, `R`, `X`, `Imax`, `q_rt`, `m`, `phase` | MV branch endpoints and phase parameters |
| `BT` | `l`, `k`, `R`, `X`, `Imax`, `q_rt`, `m`, `phase` | LV branch endpoints and phase parameters |
| `Trafos` | `trafo_id`, `phases`, `mv_bus`, `lv_bus`, `kv_mv`, `kv_lv`, `kva`, `R_ohm`, `X_ohm`, `conn_mv`, `conn_lv` | Distribution transformers |
| `Reg` | `trafo_id`, `phases`, `mv_bus`, `lv_bus`, `kv_mv`, `kv_lv`, `kva`, `R_ohm`, `X_ohm`, `ptratio_V`, `ctprim_A`, `r_LDC_V`, `X_LDC_V`, `vreg_V`, `band_V`, `Delay_s`, `maxtapchange`, `tapdelay_s`, `TapNum`, `Step`, `Taps` | SVR and LDC parameters |

The distributed workbooks are the authoritative templates for units, phase
encoding and optional columns. Preserve their sheet names and headers when
creating a new case.

## Calendar correction

The former meteorological branch used 2001 as an auxiliary calendar even
though the data represented 2019. In this repository:

- filenames use `perfis_10min_2019_MM.csv`;
- timestamps and `source_year_index` use 2019;
- the manifest declares `calendar_year: 2019`;
- both notebooks operate directly on 2019 without a year-index conversion.

Both 2001 and 2019 are non-leap years, so the correction preserves the
month/day/time correspondence and the number of samples.

## Solver requirement

The OPF notebook uses IBM ILOG CPLEX. By default it expects the Linux installer
at:

```text
/content/drive/MyDrive/Solvers/cplex_studio2212.linux_x86_64.bin
```

The installer and license are not distributed. The user is responsible for a
valid IBM license and for complying with its terms. Update the installation
cell and `CFG.cplex_executable` if a different CPLEX release or path is used.

## Local execution

Python dependencies are listed in `requirements.txt`:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When running `DiasRepresentativos.ipynb` locally, start Jupyter in the
repository root so that `Path.cwd()` resolves to this directory.

`src/acopf.py` is a standalone implementation retained from the earlier
script-based workflow:

```bash
python src/acopf.py --cplex-exe /path/to/cplex
```

It is not imported by either notebook and is not the canonical implementation
for the V4.9 experiments reported by the current study.

## Reproducibility checklist

Before using results in the manuscript:

1. record the Git commit or immutable release tag;
2. keep the notebook `campaign_id` with the exported result folder;
3. retain `frozen_parameters.json` and its SHA-256 hash;
4. verify that all physical-closure and operational guards pass;
5. verify that final scenarios use the exact fixed-tap SVR relation;
6. verify chronological continuity of the relay state in multiday runs;
7. report the CPLEX and Python-package versions and computational hardware;
8. archive the exact result bundle used to generate each table and figure.

## Scope and interpretation

The implementation uses a phase-explicit, phase-diagonal branch-flow model.
Consequently, the reported voltage-unbalance measure is a voltage-magnitude
proxy rather than a full sequence-component VUF obtained from a mutually
coupled three-phase model. Transferability claims must be based on the frozen,
state-continuous chronological experiment, not solely on representative-day
identification.

If contextual Q-learning and direct reward maximization yield identical
assignments and electrical outcomes, the result should be described as
contextual reward-based classification rather than as evidence of a specific
reinforcement-learning advantage.

## Citation and license

Citation metadata will be added when the manuscript or its preprint receives a
persistent identifier. A `CITATION.cff` file and explicit licenses for code and
data should be included before the repository is formally released as the
article's reproducibility package.
