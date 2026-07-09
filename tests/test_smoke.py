from pathlib import Path
import importlib.util
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "FRLTC_Model_Validation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("frltc_model_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_pipeline_generates_expected_outputs(tmp_path):
    module = load_module()
    out_dir = tmp_path / "outputs"

    module.run_demo(out_dir, seed=8, r=2, strict_audit=True)

    expected_files = [
        "FR-LTC_gate_metrics.csv",
        "FR-LTC_singular_values.csv",
        "FR-LTC_mode_fit.csv",
        "FR-LTC_parameter_stability.csv",
        "FR-LTC_numerical_audit.csv",
        "FR-LTC_run_summary.txt",
        "FR-LTC_residual_singular_values.png",
        "FR-LTC_mode_fit.png",
        "FR-LTC_mode_derivatives.png",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name

    metrics = pd.read_csv(out_dir / "FR-LTC_gate_metrics.csv")
    assert metrics.loc[0, "gate_decision"] == "accept"
    assert metrics.loc[1, "gate_decision"] == "fallback"

    audit = pd.read_csv(out_dir / "FR-LTC_numerical_audit.csv")
    thresholded = audit["tolerance"].notna()
    assert bool(audit.loc[thresholded, "passed"].all())
