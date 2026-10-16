# Monza 2026: Antonelli Race Strategy Reconstruction

An independent educational project reconstructing the pace estimates and strategy analysis
Ruth Buscombe published for Kimi Antonelli before the 2026 Italian Grand Prix
([*The best strategy depends on who you are*](https://ruthbuscombe.substack.com/p/the-best-strategy-depends-on-who)).
It combines a public-timing reconstruction with a deterministic, full-field (22-car, 53-lap) race simulator.
The simulator uses the lap-wise methodology of Heilmeier, Graf & Lienkamp (2018),
*A Race Simulation for Strategy Decisions in Circuit Motorsports*.

This is an independent reconstruction of disclosed rules, not Ruth's original simulator or a validated race
forecast. Unspecified details are completed with documented assumptions and additional evidence.

## Articles on Medium

A two-part write-up of this project is coming soon on Medium. Links will be added here after publication.

| Part | Article | Medium link |
|---|---|---|
| 1 | **How Do F1 Strategists Estimate Race Pace?** | Coming soon |
| 2 | **How Do F1 Strategists Choose a Race Strategy?** | Coming soon |

Part 1 reconstructs Ruth Buscombe's Monza pace estimates from public timing. Part 2 explores three strategies
across five Safety-Car scenarios and the assumptions that change the answer.

## Scientific objective

Ruth published a 3 × 5 grid of modelled finishing positions for Antonelli: three strategies (a lap-1 stop, a
long first stint on the hard, a conventional medium–hard plan) in five Safety-Car "worlds". She disclosed the
main parameters of her model but not the model itself. This project asks:

> Can the published pace estimates be reproduced from public timing, and can her disclosed rules, completed
> with explicit implementation choices, produce similar strategy results? Which assumptions explain the
> results and their differences from hers?

## The notebooks

### Standalone FP2 long-run reconstruction

[`fp2_long_run_reconstruction.ipynb`](fp2_long_run_reconstruction.ipynb) implements the long-run extraction
without importing the simulator or its pace functions. It records every lap's retention or rejection reason,
compares the selected runs with Ruth's table, tests alternative readings of the filter, and investigates the
Bearman discrepancy. Its saved outputs are in `outputs/fp2_long_run_reconstruction/`.

This notebook estimates FP2 long-run gaps only. It does not compute the Friday-weighted or Saturday-rebased
race-pace inputs, which are reconstructed in the main notebook. Agreement between the two implementations
is a within-project cross-check, not an independent external replication.

### Main pace and strategy reconstruction

[`monza_2026_antonelli_strategy_reconstruction.ipynb`](monza_2026_antonelli_strategy_reconstruction.ipynb)
contains the complete pace-to-strategy analysis:

| Sections | Content |
|---|---|
| 1–3 | research question, sources and methodology, data and its provenance |
| 4 | reconstruction of Ruth's two race-pace readings from FP2 and qualifying timing |
| 5 | the mathematical model, element by element, with the reason for each rule |
| 6–7 | the simulator code (written by the notebook itself), analytic checks, a worked example of the Safety Car on a shared race clock, the test suite, and the invariants checked in every scenario |
| 8–9 | the 15 scenarios: results, mechanisms, comparison with Ruth, figures |
| 10 | sensitivity analysis (tyre degradation, pace, starting assumptions, traffic, rival pit schedules, Safety-Car timing) and a robustness summary |
| 11–12 | discussion, limitations and conclusions |
| 13 | a quarantined post-race epilogue (the actual race, never used to build the model) |
| 14 | reproducibility record |
| final | **Project Roadmap and Future Work**: the five phases of the project |

## Main findings

- **FP2 reconstruction.** All 18 published long-run estimates match at the displayed precision, including
  compound and clean-lap count. The literal method also finds a six-clean-lap medium run for Bearman, whom
  Ruth lists as having no long run. The standalone extraction keeps this unresolved discrepancy.
- **Race-pace inputs.** All 22 published Friday-weighted values and the ten published Saturday-rebased values
  are reproduced at their displayed precision. Matching Friday-weighted inputs retains Ruth's qualifying-only
  treatment of Bearman as an explicit source-matching exception. Completing the Saturday set requires eight
  additional reconstructions and four assumed drop-offs; it is not a 22-driver published match.
- **Baseline choice.** The simulator uses Ruth's published, rounded Friday-weighted values, not the unrounded
  reconstructions. Her article does not unambiguously identify which pace reading produced the Antonelli grid.
  Alternative pace sets are tested separately.
- **Agreement with her 15 positions.** Five exact matches and ten within one place, including exact matches;
  position mean absolute error 1.60 places. Winner-gap mean absolute error is about 6.4 seconds. The model was
  developed with her predictions known, so this is not a blind test or validation against a real race.
- **Quiet-race preference.** A and B both finish P3, but B is 5.19 seconds quicker. Ranking by finishing
  position, then winner gap, B is preferred in four of the five baseline scenarios. C narrowly takes the
  lap-5 scenario, 0.04 seconds ahead of A. A is not the quiet-race winner in this reconstruction.
- **Qualitative pattern.** Early or mid-race Safety Cars hurt A; its position survives a lap-35 Safety Car;
  B wins with that late deployment; C finishes P8 throughout the baseline matrix. These are conditional model
  results, not universal strategy rules.
- **Unresolved differences.** B finishes P3 after both lap-1 and lap-20 Safety Cars, versus Ruth's P9 and P8.
  Starting offsets, passing and restart mechanics, tyre-life thresholds and pace inputs affect these results.
  Their sensitivity does not establish how Ruth implemented her model.
- **Hard degradation.** Raising the shared hard-tyre degradation rate from 0.015 to 0.05 s/lap moves quiet-race
  A from P3 to P8 and B from P3 to P5; B remains preferred. Fits to Lawson's, Antonelli's and Russell's Friday
  hard runs give approximately 0.07–0.10 s/lap with an assumed 0.07 s/lap fuel correction. Those fits cannot
  separate degradation from fuel burn and other time trends without further assumptions.
- **Verification versus validation.** The saved notebook execution records 113 passing test cases, including
  34 Safety-Car timeline cases. Internal checks and hand-calculated examples support implementation
  consistency; the simulator has **not** been independently validated against observed race outcomes.

### Baseline finishing positions

| Antonelli plan | No Safety Car | SC lap 1 | SC lap 5 | SC lap 20 | SC lap 35 |
|---|---:|---:|---:|---:|---:|
| A: medium, stop lap 1, hard to the flag | P3 | P8 | P8 | P8 | P3 |
| B: hard, planned medium stop lap 36 | P3 | P3 | P10 | P3 | P1 |
| C: medium, planned hard stop lap 20 | P8 | P8 | P8 | P8 | P8 |

Source: [saved scenario table](outputs/tables/antonelli_15_scenarios.csv). Safety-Car opportunities can bring
planned stops forward. The [comparison with Ruth](outputs/tables/comparison_with_ruth.csv) includes her
positions, both sets of winner gaps and mechanism explanations.

### Important modelling limits

Rivals use fixed pit schedules apart from the cheap-stop rule. Tyre degradation is linear; thermal effects,
incidents, retirements, red flags and VSCs are not simulated. All cars complete the model's full 53-lap distance.
The Safety Car uses one shared deployment/restart timeline, but progress within a lap is uniform, queue
interactions are resolved at line crossings, and racing pace resumes at a common restart timestamp. Some cars
may not catch the queue before the fixed three-lap period ends. This is a declared abstraction, not a detailed
recreation of real Safety-Car procedures.

## Reproduce

Install Python 3.13 separately; the recorded notebook environment used Python 3.13.2. The single
`requirements.txt` pins dependencies for both notebooks, tests, timing downloads and PDF extraction.

```bash
git clone https://github.com/Nahid-ahmdv/monza-2026-strategy-reconstruction.git
cd monza-2026-strategy-reconstruction
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name race-sim-ruth --display-name "Python (race-sim-ruth .venv)"
```

These commands use the default folder name of a fresh clone. The existing local project folder can remain
`race_simulator_Ruth`; if working there, skip the clone step and run the setup commands from that folder.

Open either notebook in Jupyter or VS Code, select the `race-sim-ruth` kernel, and run all cells. For headless
execution, use the commands below. They update notebook outputs in place. The main notebook also regenerates
code, tests and outputs, so use a separate project copy if you need to preserve the current artefacts.

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=race-sim-ruth fp2_long_run_reconstruction.ipynb

.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=race-sim-ruth monza_2026_antonelli_strategy_reconstruction.ipynb
```

- Installing dependencies and re-downloading timing need internet access. **Notebook execution can run
  offline** using the stored inputs under `data/`. If its stored FP2 CSV is missing, the standalone notebook
  instead attempts to fetch the session through FastF1.
- The saved clean-room main-notebook run took 87 seconds; runtime varies by machine. Its
  [reproducibility record](outputs/reproducibility/cleanroom_check.txt) documents that historical run, not a
  fresh verification of subsequent edits or the newly combined dependency file.
- The run is deterministic: the race engine uses no random numbers, and the randomised rival pit schedules use
  fixed seeds.
- Run the notebook from the project root with the `race-sim-ruth` kernel. A generic `python3` kernel may resolve
  to a different Python installation.
- The main notebook **stops** if the test suite fails, or if the configuration, the code or the effective model
  specification differs from the fingerprints in `config/FINGERPRINT.txt`.

The main notebook **writes** the `racesim/` package and the `tests/` suite with `%%writefile` cells, then imports and
runs them, so the code you read in the notebook is exactly the code that runs. **Edit the notebook, not the
generated `.py` files**; they are overwritten on every run, and any change to them requires re-recording the
code fingerprint. With the generated files present, the tests can also be run on their own:

```bash
.venv/bin/python -m pytest tests -q        # saved notebook run: 113 passing cases
```

## Repository structure

| Path | Content |
|---|---|
| `fp2_long_run_reconstruction.ipynb` | standalone FP2 long-run reconstruction, lap audit, filter variants and Bearman investigation |
| `monza_2026_antonelli_strategy_reconstruction.ipynb` | the executed research notebook (the primary artefact) |
| `config/monza_2026.yaml` | every model parameter and modelling choice, each with `value`, `status` (published / reconstructed / evidence / assumed), `role` (operative, or fixed and validated), `source` and reasoning |
| `config/FINGERPRINT.txt` | SHA-256 of the configuration, the simulator code and the effective model specification |
| `racesim/` | the simulator package: `config`, `strategies`, `lap_model`, `traffic`, `safety_car`, `engine`, `scenarios`, `pace` (generated by the notebook) |
| `tests/` | pytest suite (generated by the notebook) |
| `data/published/` | values transcribed from Ruth's article and its charts |
| `data/timing/` | official 2026 Italian GP FP2 laps and qualifying (FastF1); **no race data** |
| `data/historical/` | 2025 Monza race stints; end-of-lap-1 gaps at Monza 2023–2025 (OpenF1, FastF1) |
| `data/sources/` (local only) | downloaded third-party reference articles, graphics, research paper and regulations; excluded by `.gitignore` unless already tracked; see the original sources cited below |
| `data/post_race/` | the **actual** 2026 race outcome; read **only** in the notebook's epilogue |
| `outputs/figures/`, `outputs/tables/`, `outputs/lap_by_lap/` | every figure and table, and the full 22-car × 53-lap record and event log of each of the 15 scenarios |
| `outputs/fp2_long_run_reconstruction/` | standalone notebook's lap audit, stint summary, published comparison, method variants and figures |
| `outputs/reproducibility/` | record of the offline clean-room run |
| `scripts/` | one-off download scripts for the input data (dependencies are in `requirements.txt`; downloads need internet) |
| `requirements.txt` | combined pinned dependencies for both notebooks, tests, data downloads and PDF extraction |
| `articles/` (local only) | publication drafts; ignored by `.gitignore` unless files are already tracked |

## Data integrity

The project retrospectively reconstructs **pre-race** analysis:

- construction uses practice, qualifying, published inputs and historical evidence, not the 2026 race session;
- actual race data (`data/post_race/`) are read only in the clearly labelled epilogue, separately from model
  construction, calibration and selection;
- historical (2023–2025) Monza data are used only as evidence for unpublished inputs (tyre life, lap-1 spread).

The notebook acknowledges that the headline race result was encountered during source review, so the overall
development is not claimed to be outcome-blind. The single post-race epilogue is an illustration, not predictive
validation. Broader strategy optimisation, Monte Carlo race simulation and independent historical validation
remain future work; the existing one-stop scan and schedule permutations are narrower experiments.

## Sources

- Ruth Buscombe, *The best strategy depends on who you are*, The Race Strategy Society (Substack), Monza 2026.
- A. Heilmeier, M. Graf, M. Lienkamp, *A Race Simulation for Strategy Decisions in Circuit Motorsports*, IEEE ITSC
  2018, doi:10.1109/ITSC.2018.8570012.
- TUMFTM/race-simulation, https://github.com/TUMFTM/race-simulation (LGPL-3.0). Its methodology was reviewed and
  its fitted Monza parameters are cited as evidence; no code was copied.
- Formula1.com, *The insider's guide to… F1 strategists* (M. Youson, 2020).
- FIA 2026 Formula 1 Sporting Regulations, Section B, Issue 07 (25 June 2026).
- Official F1 timing via FastF1 3.8.3; OpenF1 API (https://openf1.org).
