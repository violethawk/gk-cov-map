# Human annotation benchmark protocol

The `<90 seconds after warm-up` criterion cannot be established by an automated test because it is a human interaction latency requirement.

Use the included synthetic clip for five warm-up trials, then time ten complete annotations:

1. Load the already-selected clip and JSONL destination.
2. Mark strike and crossing frames.
3. Mark four goal references, ball crossing, keeper strike, and keeper crossing.
4. Select outcome/dive direction and append.
5. Reset for the next trial.

Pass condition: median of the ten measured trials is below 90 seconds, with no invalid records and no coordinate/timing regression against `synthetic_expected.json`.

Record results in a dated file under `docs/benchmarks/` before declaring Phase 1 complete.
