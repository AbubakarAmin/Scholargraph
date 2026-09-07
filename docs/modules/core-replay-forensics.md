# Replay and Forensics

`core/replay.py` replays companion experiments in the sandbox or in a newly created clean virtual environment. The clean mode is invoked with `python replay_run.py PATH --clean-env`.

`core/forensics.py` reconstructs durable events, claims, artifacts, unresolved claims, and claim lineage. `forensic_report.py` handles one run; `historical_report.py` selects the latest failed and completed runs.

Full-text source retrieval remains open-access only and is cached with source URL, license, hash, and retrieval status. A missing license or unavailable source is reported as unavailable, never treated as evidence.