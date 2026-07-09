# Reproducibility

The sample outputs in this repository were generated with:

```bash
python FRLTC_Model_Validation.py --output-dir sample_outputs --seed 8 --rank 2 --strict-audit
```

The deterministic default configuration uses:

- seed: `8`
- selected residual rank: `2`
- number of assets: `40`
- number of local-time grid points: `96`
- local time range: `[0.05, 1.0]`

Expected gate decisions:

```text
base_window            accept
stress_boundary_test   fallback
```

Expected best O-E fit summary:

```text
m = 0.000
beta = -0.200
omega = 9.400
relative_rmse ≈ 0.0595
```

All thresholded audit checks should pass when `--strict-audit` is used.
