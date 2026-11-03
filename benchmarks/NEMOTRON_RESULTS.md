# Nemotron benchmark results

The completed run measured 480 model, instruction, and trial outcomes across four
free routes. Each route handled 40 instructions over three trials. The routes were
run sequentially, with at least 3.1 seconds between provider requests.

| Shortcut | Model ID | Measured | Syntax valid | Compilable | Semantically correct | Mean latency | Estimated cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nano-omni` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 120/120 | 98.96% | 98.96% | 54.17% | 20.90 s | $0.00 |
| `ultra` | `nvidia/nemotron-3-ultra-550b-a55b:free` | 120/120 | 100% | 100% | 95.83% | 24.08 s | $0.00 |
| `lightning` | `nvidia/nemotron-3.5-lightning:free` | 120/120 | 93.75% | 93.75% | 66.67% | 60.14 s | $0.00 |
| `super-vl` | `nvidia/nemotron-3.5-super-vl:free` | 120/120 | 100% | 100% | 88.54% | 15.28 s | $0.00 |

Semantic correctness compares each formula with the project-authored reference under
the benchmark's finite-trace semantics. The references still need independent human
review. These metrics show agreement with those references, not proof that they cover
every reasonable interpretation of the English instructions.

The run recorded 679 provider requests. Seven requests failed transiently during the
run: five connection errors, one chunked response error, and one HTTP 502. Resume runs
recovered every scheduled outcome. OpenRouter returned no HTTP 404 or 429 for these
four routes. The results include only free-route attempts, so estimated cost is $0.00.

Separate route checks found that OpenRouter no longer serves the 9B v2 or 12B v2 VL
routes. Nano and Super free routes returned 404. In an earlier run, Nano's paid route
produced a response and Super's paid route returned 429. This completed comparison
excludes those four aliases. Review their route availability before adding them to a
future comparison.

The runner stores full raw outputs and usage data in the local,
gitignored `benchmarks/results/nemotron/20261103T073211Z/` directory. Use the commands
in [the run guide](NEMOTRON.md) to repeat or extend the benchmark.
