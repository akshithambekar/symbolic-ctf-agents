# Partial Nemotron benchmark results

The initial three-trial run scheduled 960 outcomes across eight model aliases and 40
instructions. It measured three responses before unavailable routes and a rate limit
stopped progress. These samples are too small for a model comparison.

| Shortcut | Measured | Observed route result |
| --- | ---: | --- |
| `nano-9b-v2` | 0 | Free and paid routes returned 404. OpenRouter said no provider serves either route. |
| `nano-12b-v2-vl` | 0 | Free and paid routes returned 404. OpenRouter said no provider serves either route. |
| `nano` | 1 | Free route returned 404. The paid route returned one response. |
| `nano-omni` | 1 | Free route returned one response. |
| `super` | 0 | Free route returned 404. The paid route returned 429 with `Provider returned error`. |
| `ultra` | 1 | Free route returned one response. |
| `lightning` | 0 | No measured responses. |
| `super-vl` | 0 | No measured responses. |

The single responses do not produce a semantic accuracy rate. A full comparison needs
all 40 instructions for each available model and three trials. Run the remaining
single-trial checks and the full benchmark as described in [the run guide](NEMOTRON.md).

The runner stores detailed outputs, provider errors, and usage data in the local
`benchmarks/results/nemotron/` directory. Git ignores that directory. The first run
was written to `benchmarks/results/nemotron/20261103T044915Z`. A separate retest of
`nano-9b-v2` was written to `benchmarks/results/nemotron/20261103T050550Z`.
