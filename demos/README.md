# PyQuaticus sample-agent demo

This demo runs existing PyQuaticus policies in maritime capture the flag.
The default game is 1v1: blue attacks red's flag while red defends it. It opens
a live game window, prints final counters, and exits. Behavior comes from
handwritten upstream rules; RL training is a later task. The demo creates no
recordings, run directories, or result files.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
On macOS, Windows PowerShell, or Linux, run from the repository root:

```bash
cd demos
uv sync --locked
uv run --locked python run_sample_agents.py
```

For a short live preview:

```bash
uv run --locked python run_sample_agents.py --seconds 12
```

From the repository root, you can also use:

```bash
uv run --locked --project demos python demos/run_sample_agents.py
```

Close the window or press Escape to stop early. Desktop rendering needs a
local graphical session; restricted agent shells may lack display access.
A headless check runs the same game and prints its results:

```bash
uv run --locked python run_sample_agents.py --render none
```

## What happens

1. PyQuaticus creates the field, flags, agents, and movement physics.
2. Blue's `BaseAttacker` in `medium` mode moves toward red's flag while avoiding
   the opponent. Carrying the flag changes its target to home.
3. Red's `BaseDefender` in `medium` mode moves toward its flag when too far away
   and away when too close. This can produce back-and-forth movement.
4. The simulator checks tagging, flag pickup, capture, and recovery as the
   policies choose movement actions.
5. At the episode limit, the window closes and the terminal prints capture,
   grab, and tag totals in blue/red order.

The default seed-7 run completed 120 simulated seconds with tags `[0, 1]`,
captures `[0, 0]`, and grabs `[0, 0]` during verification. It demonstrates
execution and tagging. Successful capture and learned-agent performance need
separate experiments.

| Visual | Meaning |
| --- | --- |
| Blue agent labeled 0 | Attacker |
| Red agent labeled 1 | Defender |
| Stationary colored circles | Team flags |
| Vertical center line | Division between team territories |
| Green ring around an agent | Tagged agent in recovery |
| Black arc around an agent | Tagging cooldown |
| Yellow ring around an agent | Out of bounds |

## Configuration and other examples

`configs/demo.json` supplies the defaults: `attack-defend`, live (`human`)
rendering, 120 simulated seconds, and seed 7. Command-line settings override
those values. A custom config supports the same four keys.

```bash
uv run --locked python run_sample_agents.py --example random
uv run --locked python run_sample_agents.py --example combined
uv run --locked python run_sample_agents.py --config configs/demo.json --seed 8
```

Random uses 1v1 and sampled discrete actions. Combined uses the upstream
`Heuristic_CTF_Agent` in `hard` mode in 2v2 with `si` dynamics. Attacker/defender
uses the default `heron` dynamics. All examples use the default non-GPS field,
a simulation speedup factor of 10, and discrete actions. Live playback duration
depends on the renderer as well as simulated time.

## Dependencies and portability

PyQuaticus is installed directly from competition commit
`475d670650ec5d49691ba707cd1054c4e45d8053` by uv. The source is unmodified and
managed in uv's cache. `pyproject.toml` declares dependencies; `uv.lock` pins
the resolved versions. No local source checkout, setup script, or tracked
patch is needed.

The demo uses PyQuaticus's bundled observation-based policies from
`base_policies.deprecated`. They consume dictionaries of unnormalized
observations for every agent. This grants access to both teams' observations;
keep that information assumption in mind when designing a future RL policy.
The newer upstream policies import the MOOS hardware bridge during module
loading. The bundled legacy policies avoid that dependency. Before switching,
all 240 attacker/defender actions and all 480 combined-policy actions matched
our previous demo over the seed-7, 120-step runs.

The uv project excludes `pymoos`, which is the unused hardware-bridge package
and has no matching Apple Silicon distribution for the pinned version. This
is a simulation-only dependency selection. Hardware deployment needs its own
environment. SciPy 1.14.1 is constrained on ARM macOS because 1.15.3 failed
native-library loading on the tested macOS 27 machine.

Python 3.10 is required by this PyQuaticus revision. `.python-version` selects
3.10.20, the patch version used for verification. uv downloads Python when
needed and creates `demos/.venv` to isolate installed packages. That environment
is gitignored and needs no manual activation.

Live and headless execution were verified on macOS 27 ARM64. The Windows x64
locked installation plan passed a dry run; Windows runtime verification remains
pending a teammate's machine. Commit dependency changes together with the
updated lockfile and recheck the demo.
