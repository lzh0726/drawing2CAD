# A3 fixed-test evaluation

| Metric | A2 | A3 |
|---|---:|---:|
| Valid solids | 35/44 | 36/44 |
| Raw IoU, all 44 | 0.515334 | 0.532643 |
| Translation-aligned IoU, all 44 | 0.667306 | 0.699065 |
| Raw IoU, valid outputs | 0.647848 | 0.651008 |
| Translation-aligned IoU, valid outputs | 0.838899 | 0.854413 |
| XYZ bounding dimensions within 1% | 29 | 31 |
| XYZ bounding dimensions within 5% | 31 | 33 |

All-44 means assign zero to invalid CAD; unresolved metric errors leave the mean unavailable.
Exact serial OCC Boolean IoU, also measured after bbox-center translation; no rotation or scale fitting.
50 fixed test cases, 6 excluded for invalid ground truth, 44 evaluated. This previously diagnosed set is a regression test.
A3 checkpoint1225. Same input and generation settings as A2: 16384 new tokens, 600 seconds, greedy, no repairs.
GPU admission is stricter (<15% utilization, >42 GiB free); model weights are CPU-merged in memory and cast to BF16.
All four projectors are checked against the saved A3 tensors on every model load.
Dimension pass checks bounding extents, not every engineering dimension or feature topology.

Newly valid relative to A2: test-00354124, test-00118368, test-00442545, test-00361382
Regressed relative to A2: test-00966577, test-00152464, test-00002718
Unresolved metric errors: 
Original source hashes unchanged: True
