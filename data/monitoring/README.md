# Local monitoring data

`events.jsonl` is the default append-only local monitoring path. Every JSONL
file in this directory is ignored by Git and excluded from Docker build
contexts.

The reviewed fixture is
`evaluation/fixtures/monitoring_events.jsonl`. It is deterministically
generated from a fixed seed, contains no real user or customer data, and is
dedicated to the public domain under CC0 1.0. Regenerate it from the repository
root:

```powershell
python -m evaluation.generate_monitoring_fixture --force
```

Do not copy production telemetry, raw prompts, retrieved document text,
credentials, secrets, or private feedback into this directory for publication.
The application code and reporting tools remain under the repository MIT
License.
