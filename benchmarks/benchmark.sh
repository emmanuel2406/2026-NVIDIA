#!/usr/bin/env bash
#
# LABS Benchmark Runner
#
# Runs different methods for values of N specified as input.
# Uses eval_util.py from tutorial_notebook/evals to validate results.
# Writes output to benchmarks/results.csv.
#
# Usage:
#   ./benchmark.sh                    # default: N = 3,4,...,25
#   ./benchmark.sh 3 4 5 10 20        # specific N values
#   ./benchmark.sh 3-10               # range 3..10
#   ./benchmark.sh 3-10 20 25         # range + additional values
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running LABS benchmarks..."
python3 run_benchmark.py "$@"
echo "Done. Results in benchmarks/results.csv"
