#!/bin/bash
# Pull results back, then: python scripts/03_evaluate.py --figures
cd "$(dirname "$0")/.." && rsync -az sanganak:battery/results/ results/ && rsync -az sanganak:battery/cluster/logs/ cluster/logs/
