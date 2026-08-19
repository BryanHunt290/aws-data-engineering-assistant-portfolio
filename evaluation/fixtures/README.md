# Monitoring evaluation fixture

`monitoring_events.jsonl` is deterministic synthetic demonstration data. It
contains no real users, customer documents, prompts, credentials, or provider
responses. The fixture data is dedicated to the public domain under CC0 1.0.

Verify it from the repository root with the reviewed fixed seed:

```powershell
python -m evaluation.generate_monitoring_fixture
```

Add `--force` only to replace the fixture intentionally. Then regenerate
aggregate evidence and Matplotlib charts:

```powershell
python -m evaluation.run_monitoring_report
```

Review both the fixture and generated evidence before committing a new
snapshot. Do not place private or production telemetry in this directory.
