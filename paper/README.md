# Working manuscript

`manuscript.tex` is the authoritative working paper supplied by Yahya on October 12, 2026 and updated with the runtime-only study. `manuscript.pdf` is the compiled nine-page revision. `originals/submitted-20261012.tex` preserves the supplied manuscript unchanged.

The revision separates the initial deployment-only shield proposal from implemented training-time guidance, clarifies bounds-gate limitations, and reports the completed exploratory 360-episode paired comparison. Both arms had zero captures/pickups; the gate substituted no actions in 43,200 checks and all 180 pairs had identical recorded trajectories. This does not estimate intervention benefits in a task-capable attacker.

The matching bibliography and FLAIRS styles were copied from the user's Downloads files; asset_sources.json records their origin. The architecture figure is reproducible with scripts/render_paper_architecture.py. build_info.json records source/output hashes and compilation checks. CHANGELOG.md describes the revisions.

## Compile

From paper/:

```text
pdflatex -output-directory=build manuscript.tex
bibtex build/manuscript
pdflatex -output-directory=build manuscript.tex
pdflatex -output-directory=build manuscript.tex
```

Create build/ first. Required dependencies include the FLAIRS template styles, accessibility, and the Times/Helvetica/Courier LaTeX font packages. The built-in standalone compiler cannot load these project files; the local MiKTeX compiler was used.

The tactical-attacker follow-up adds two imitation-initialized checkpoints, actual bounds-gate substitutions in an exploratory moving-start test, higher-speed deadlock evidence, and scripted per-player role screening. See the new study report; this is not a general superiority claim.

The offensive follow-up reports the completed five-seed selected-condition replication, controlled role screen, carrier conflicts and stopped chase continuation. The nine-page PDF was rebuilt and visually checked; the report preserves selection reuse and reward/task mismatch limitations.
