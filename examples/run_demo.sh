#!/usr/bin/env bash
set -euo pipefail

python FRLTC_Model_Validation.py --output-dir outputs --seed 8 --rank 2 --strict-audit
