# -*- coding: utf-8 -*-
"""Run the four pipeline stages in order."""
import subprocess
import sys

STEPS = [
    ["-m", "pipeline.train", "--data", "data", "--out", "artifacts"],
    ["-m", "pipeline.monitor", "--data", "data",
     "--artifact", "artifacts/forecast.npz", "--out", "artifacts"],
    ["-m", "pipeline.rootcause", "--data", "data",
     "--artifact", "artifacts/forecast.npz",
     "--monitor", "artifacts/monitor_events.csv", "--out", "artifacts"],
    ["-m", "pipeline.agent", "--data", "data",
     "--rootcause", "artifacts/rootcause_details.json",
     "--knowledge", "knowledge/faults.json", "--out", "artifacts"],
]

for i, args in enumerate(STEPS, 1):
    print("\n=== step %d/%d: %s ===" % (i, len(STEPS), " ".join(args)))
    subprocess.run([sys.executable] + args, check=True)
print("\nall stages finished.")
