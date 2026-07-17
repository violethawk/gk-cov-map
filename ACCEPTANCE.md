# Phase 1 Acceptance Status

| Criterion | Status | Evidence |
|---|---|---|
| Annotate one penalty in <90 s after warm-up | **Pending human benchmark** | Protocol in `docs/annotation-benchmark.md`; cannot be honestly automated |
| Synthetic round-trip within 5 cm and 1 frame | **Passed** | Python and browser geometry tests; MP4 metadata test; included synthetic clip |
| 200-shot simulator recovers asymmetry within CI ≥90% over 20 runs | **Passed** | 20/20 interval coverage; 20/20 sign recovery; 99/100 over a wider 100-run sweep |
| Prior-dominated corner visibly differs from data-rich center | **Passed** | Posterior/prior SD ratio test plus in-map fading and `///` hatching |

Phase 2 is blocked until the human timing benchmark is run and recorded. No acceptance criterion has been silently weakened.
