# GridWorld Q-Learning Agent

This project runs a resumable tabular Q-learning campaign against the NoteXponential GridWorld API.

## Files

- [main.py](/Users/mehriban/ada/ai-p4-gridworld/main.py): campaign runner, API client, parser, persistence, learner, and per-world agent logic.
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
python3 main.py
```

Useful options:

```bash
python3 main.py \
  --max-steps 300 \
  --campaign-world-start 1 \
  --campaign-world-end 10 \
  --traversals-per-world 5 \
  --rows 40 \
  --cols 40 \
  --epsilon-min 0.2 \
  --epsilon-decay 0.999 \
  --exploration-bonus 1.0 \
  --state-novelty-bonus 0.25 \
  --move-delay 2.0 \
  --enter-delay 600 \
  --location-retries 3 \
  --location-retry-delay 1.0 \
  --null-state-terminal-reward 1000 \
  --history-limit 200
```

Key flags:

- `--rows`, `--cols`: grid dimensions used for state indexing and Q-table shape.
- `--campaign-world-start`, `--campaign-world-end`, `--traversals-per-world`: control the resumable exploration campaign. Defaults are worlds `1..10`, `5` completed traversals each.
- `--history-limit`: caps stored episode summaries in metadata.
- `--epsilon-min`, `--epsilon-decay`: keep more exploration active for longer during training.
- `--exploration-bonus`: adds a count-based bonus to less-visited actions during greedy selection, which improves map coverage.
- `--state-novelty-bonus`: adds intrinsic reward for reaching rarely visited states, which pushes exploration toward new areas instead of repeating familiar loops.
- `--enter-delay`: defaults to `600` seconds to respect the assignment rule of no more than one `enter` call every 10 minutes.
- `--location-retries`, `--location-retry-delay`: retry `location` after incomplete move responses before falling back to a self-loop.
- `--null-state-terminal-reward`: if `nextState` is `null` and `abs(reward)` reaches this threshold, the move is treated as an assumed terminal outcome.
- `--eval-mode`: disables exploration and learning updates so you can run the current policy greedily for measurement.
- `--strict-missing-state`: fail the episode if a move response and fallback location lookup both omit the next state. By default the agent assumes a self-loop and records a warning.
- `--quiet`: suppresses per-step logging.

## Outputs

Per world, the agent writes:

- `q_data/world_<id>_q.npy`: Q-table
- `q_data/world_<id>_visits.npy`: state-action exploration counts
- `q_data/world_<id>_state_visits.npy`: state visit counts used for novelty bonuses
- `q_data/world_<id>_meta.json`: epsilon, history, totals, and rolling metrics
- `q_data/campaign_progress.json`: resumable campaign status for worlds `1..10`

The metadata file stores:

- grid shape
- episode summaries
- best reward
- rolling reward and rate metrics
- cumulative action/state counters
- assumed terminal event counts for `null`-state high-magnitude rewards
- recent warnings, including inferred self-loops caused by incomplete API responses

The move parser intentionally does not infer blocked, invalid, or terminal status from response message text. It only uses explicit structured fields such as `reward`, `newState`, and `worldId`.

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```

## Notes

- Saved Q-table shape must match the configured or previously saved grid shape.
