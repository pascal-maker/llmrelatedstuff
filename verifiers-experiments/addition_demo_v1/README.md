# Addition Demo V1

A tiny v1 verifiers environment for checking the basic taskset/reward loop.

Each task asks one addition question. The reward is `1.0` when the model replies with
exactly the expected integer and `0.0` otherwise.

## Run From The Repo Root

```bash
uv pip install -e environments/addition_demo_v1
uv run validate addition-demo-v1 -n 5 --runtime.type subprocess --rich False
uv run eval addition-demo-v1 -n 3 --dry-run True --push False --rich False
```

Use `--runtime.type subprocess` for local validation unless Docker is running.

- `addition_demo_v1/taskset.py` — the task (`@reward` scoring + behavior) and the taskset: `load` (data + prompts).

Tune the number of generated questions with `--taskset.num-tasks 10`.
