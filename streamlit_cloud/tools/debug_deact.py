# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
al = pd.read_csv(os.path.join(D, "alarms.csv"))
ev = pd.read_csv(os.path.join(D, "events.csv"))

for r in ["R06", "R07"]:
    print("=== alarms", r, "===")
    print(al[al.run_id == r][["tag", "kind", "raised_time", "is_nuisance",
                              "is_invalid", "nuisance_reason",
                              "root_event_id"]].to_string(index=False))
    print("=== events", r, "===")
    print(ev[ev.run_id == r][["event_id", "fault_type", "root_alarm_time",
                              "root_alarm_valid", "quality_alarm_time",
                              "quality_limit_time_true", "quality_limit_time_meas",
                              "lead_root_min", "dcs_lag_min", "detected"]]
          .to_string(index=False))
