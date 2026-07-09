# FR-LTC Model Validation

**Factor-Residual Local Turning Calculus for Model Validation and Stress-Gated Usage Control**

This repository contains a reproducible demonstration implementation of Factor-Residual Local Turning Calculus (FR-LTC). The code is intended as a **diagnostic model-validation demonstration**, not as a trading strategy.

The implementation follows the conservative residual-diagnostic workflow described in the accompanying manuscript:

```text
observed matrix R_W
-> teacher residual E_W
-> low-rank residual modes by SVD
-> scalar oscillatory-envelope fit for residual-mode scores
-> stability and stress diagnostics
-> accept / downgrade / fallback / reject usage gate
```

## Repository contents

```text
FRLTC_Model_Validation.py           Main reproducible Python implementation
FRLTC_Model_Validation.pdf          Manuscript PDF
sample_outputs/                     Deterministic outputs generated with seed=8, rank=2
docs/IMPLEMENTATION_NOTES.md        Method and implementation notes
docs/REPRODUCIBILITY.md             Reproducibility checklist
examples/run_demo.sh                Example command-line run
tests/test_smoke.py                 Smoke test for the demo pipeline
requirements.txt                    Runtime dependencies
pyproject.toml                      Project metadata and pytest configuration
LICENSE                             MIT License
CITATION.cff                        Citation metadata
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python FRLTC_Model_Validation.py --output-dir outputs --seed 8 --rank 2 --strict-audit
```

The script writes CSV diagnostics, PNG figures, and a JSON-style run summary to the selected output directory.

## Expected outputs

The default run creates:

```text
FR-LTC_gate_metrics.csv
FR-LTC_singular_values.csv
FR-LTC_mode_fit.csv
FR-LTC_parameter_stability.csv
FR-LTC_numerical_audit.csv
FR-LTC_run_summary.txt
FR-LTC_residual_singular_values.png
FR-LTC_mode_fit.png
FR-LTC_mode_derivatives.png
```

For the default deterministic demonstration, the base window is accepted and the severe boundary-stress test falls back:

```text
base_window            -> accept
stress_boundary_test   -> fallback
```

## Run tests

```bash
pip install -r requirements.txt pytest
pytest
```

## Manuscript

The manuscript is included in the repository root as:

```text
FRLTC_Model_Validation.pdf
```

## Scope and limitations

This repository implements the safe residual-diagnostic version of FR-LTC. It includes a compact parameter-stability report, but it intentionally does not calibrate every possible stress family and does not enable augmented-regression usage by default. Any real-data deployment would require separate out-of-window validation, leakage controls, and domain-specific gate calibration.

## License

This project is released under the MIT License.

## Author

David Hongkai Shen
