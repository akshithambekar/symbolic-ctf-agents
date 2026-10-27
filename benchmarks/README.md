# English-to-LTL benchmark

This directory is a starting point for evaluating LTLCodeGen.
It currently contains one example case and an offline case-listing command.
It does not call a model or score translations yet.

From the repository root, inspect the cases with:

```bash
python -m benchmarks.run --list
```

Use `--cases PATH` to inspect a different JSON case file.
Each case records an ID, an English instruction, proposition definitions, a reference formula, and its intended meaning.
The example uses the translator's existing navigation vocabulary and is a smoke test, not an accuracy benchmark.

The reference formula treats `reach(object_7)` as one atomic proposition.
Its notation is illustrative and has not been checked against a particular LTL parser.

## Next steps

1. Configure an explicit model and provider, with credentials supplied through the environment.
2. Make generated-code handling suitable for live model responses and address the risks of executing generated code.
3. Add a runner that records configuration, output, failures, retries, and latency for every attempt.
4. Add parsing and semantic equivalence checks, then expand the reviewed, held-out case set.
5. Save run results locally without committing credentials or generated outputs.
