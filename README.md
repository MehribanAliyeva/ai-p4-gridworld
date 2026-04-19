# GridWorld Q-Learning Agent

This project trains a tabular Q-learning agent against the NoteXponential GridWorld API.

## Files

- [main.py](/Users/mehriban/ada/ai-p4-gridworld/main.py): training entrypoint, API client, parser, store, learner, and agent orchestration.
- [tests/test_main.py](/Users/mehriban/ada/ai-p4-gridworld/tests/test_main.py): parser and learner regression tests.
- [q_data](/Users/mehriban/ada/ai-p4-gridworld/q_data): saved Q-tables and metadata.

## Environment

Create a `.env` file with:

```env
GRIDWORLD_TEAM_ID=...
GRIDWORLD_API_KEY=...
GRIDWORLD_USER_ID=...
```

Install dependencies:

```bash
python3 -m pip install numpy python-dotenv
```

## Run

Basic run:

```bash
python3 main.py --episodes 5
```

Useful options:

```bash
python3 main.py \
  --episodes 20 \
  --max-steps 300 \
  --rows 40 \
  --cols 40 \
  --move-delay 2.0 \
  --enter-delay 600 \
  --location-retries 3 \
  --location-retry-delay 1.0 \
  --history-limit 200
```

Key flags:

- `--rows`, `--cols`: grid dimensions used for state indexing and Q-table shape.
- `--history-limit`: caps stored episode summaries in metadata.
- `--enter-delay`: defaults to `600` seconds to respect the assignment rule of no more than one `enter` call every 10 minutes.
- `--location-retries`, `--location-retry-delay`: retry `location` after incomplete move responses before falling back to a self-loop.
- `--strict-missing-state`: fail the episode if a move response and fallback location lookup both omit the next state. By default the agent assumes a self-loop and records a warning.
- `--quiet`: suppresses per-step logging.

## Outputs

Per world, the agent writes:

- `q_data/world_<id>_q.npy`: Q-table
- `q_data/world_<id>_meta.json`: epsilon, history, totals, and rolling metrics

The metadata file stores:

- grid shape
- episode summaries
- best reward
- rolling reward and rate metrics
- cumulative action/state counters
- recent warnings, including inferred self-loops caused by incomplete API responses

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```

## Notes

- Saved Q-table shape must match the configured or previously saved grid shape.
