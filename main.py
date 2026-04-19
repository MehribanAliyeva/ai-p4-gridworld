import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

GW_URL = "https://www.notexponential.com/aip2pgaming/api/rl/gw.php"
SCORE_URL = "https://www.notexponential.com/aip2pgaming/api/rl/score.php"
DEFAULT_ACTIONS = ["N", "S", "E", "W"]
DEFAULT_ROWS = 40
DEFAULT_COLS = 40


@dataclass(frozen=True)
class GridSpec:
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS

    @property
    def num_states(self) -> int:
        return self.rows * self.cols

    def contains_state(self, state: Optional[int]) -> bool:
        return state is not None and 0 <= state < self.num_states

    def to_state(self, row: int, col: int) -> Optional[int]:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row * self.cols + col
        return None


@dataclass
class QConfig:
    team_id: int
    api_key: str
    user_id: str
    world_id: int = 0
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    max_episodes: int = 5
    max_steps_per_episode: int = 300
    move_delay_sec: float = 15.5
    enter_delay_sec: float = 600.0
    autosave_every: int = 1
    storage_dir: str = "q_data"
    actions: Tuple[str, ...] = tuple(DEFAULT_ACTIONS)
    verbose: bool = True
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS
    history_limit: int = 200
    infer_self_loop_on_missing_state: bool = True
    location_retry_attempts: int = 3
    location_retry_delay_sec: float = 1.0


@dataclass(frozen=True)
class MoveOutcome:
    reward: float
    next_state: Optional[int]
    done: bool
    blocked: bool
    invalid_move: bool
    raw: Dict


class GridWorldAPIError(Exception):
    pass


def run_curl(command: List[str], retries: int = 2, retry_delay: float = 2.0) -> Dict:
    last_error = None

    for attempt in range(retries + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode == 0:
            if not stdout:
                last_error = GridWorldAPIError(
                    f"Curl returned empty response. returncode=0, stderr={stderr!r}"
                )
            else:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError as exc:
                    print("RAW CURL STDOUT:")
                    print(stdout)
                    print("RAW CURL STDERR:")
                    print(stderr)
                    raise GridWorldAPIError(f"Invalid JSON response: {stdout}") from exc
        else:
            last_error = GridWorldAPIError(
                f"Curl failed. returncode={result.returncode}, stderr={stderr!r}, stdout={stdout!r}"
            )

        if attempt < retries:
            print(f"Curl attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise last_error


class GridWorldClient:
    def __init__(self, team_id: int, api_key: str, user_id: str):
        self.team_id = team_id
        self.api_key = api_key
        self.user_id = user_id

    def _headers(self) -> List[str]:
        return [
            "-H",
            f"x-api-key: {self.api_key}",
            "-H",
            f"userId: {self.user_id}",
            "-H",
            "Content-Type: application/x-www-form-urlencoded",
        ]

    def _check_response(self, data: Dict) -> Dict:
        code = str(data.get("code", "")).upper()
        if code != "OK":
            raise GridWorldAPIError(data.get("message", f"API failed: {data}"))
        return data

    def get_location(self) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{GW_URL}?type=location&teamId={self.team_id}",
            *self._headers(),
        ]
        return self._check_response(run_curl(cmd))

    def enter_world(self, world_id: int) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "POST",
            GW_URL,
            *self._headers(),
            "-d",
            f"type=enter&worldId={world_id}&teamId={self.team_id}",
        ]
        return self._check_response(run_curl(cmd))

    def move(self, world_id: int, move: str) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "POST",
            GW_URL,
            *self._headers(),
            "-d",
            f"type=move&teamId={self.team_id}&move={move}&worldId={world_id}",
        ]
        return self._check_response(run_curl(cmd))

    def get_runs(self, count: int = 10) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{SCORE_URL}?type=runs&teamId={self.team_id}&count={count}",
            *self._headers(),
        ]
        return self._check_response(run_curl(cmd))

    def get_score(self) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{SCORE_URL}?type=score&teamId={self.team_id}",
            *self._headers(),
        ]
        return self._check_response(run_curl(cmd))


class GridWorldResponseParser:
    def __init__(self, grid: GridSpec):
        self.grid = grid

    @staticmethod
    def _extract_first_int(obj: Dict, keys: Sequence[str]) -> Optional[int]:
        for key in keys:
            if key in obj:
                try:
                    return int(obj[key])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _extract_first_float(obj: Dict, keys: Sequence[str]) -> Optional[float]:
        for key in keys:
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
        return None

    def parse_state_value(self, value) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.lstrip("-").isdigit():
                return int(value)
            if ":" in value:
                parts = value.split(":")
                if len(parts) == 2:
                    try:
                        row = int(parts[0].strip())
                        col = int(parts[1].strip())
                    except ValueError:
                        return None
                    return self.grid.to_state(row, col)
        return None

    def parse_location_state(self, data: Dict) -> Optional[int]:
        possible_keys = ["state", "location", "currentState", "newState", "stateId"]

        def try_extract(obj: Dict) -> Optional[int]:
            if not isinstance(obj, dict):
                return None
            for key in possible_keys:
                if key in obj:
                    parsed = self.parse_state_value(obj[key])
                    if parsed is not None:
                        return parsed
            return None

        state = try_extract(data)
        if state is not None:
            return state

        for parent in ("data", "result", "details", "payload"):
            nested = data.get(parent)
            if isinstance(nested, dict):
                state = try_extract(nested)
                if state is not None:
                    return state
        return None

    def parse_move_result(self, data: Dict, current_state: int) -> MoveOutcome:
        reward = self._extract_first_float(data, ["reward", "score", "points"])
        world_id = self._extract_first_int(data, ["worldId", "world"])
        next_state = None
        done = False

        for key in ("newState", "state", "location", "currentState"):
            if key in data:
                next_state = self.parse_state_value(data[key])
                if next_state is not None:
                    break

        for parent in ("data", "result", "details", "payload"):
            nested = data.get(parent)
            if isinstance(nested, dict):
                if reward is None:
                    reward = self._extract_first_float(nested, ["reward", "score", "points"])
                if world_id is None:
                    world_id = self._extract_first_int(nested, ["worldId", "world"])
                if next_state is None:
                    for key in ("newState", "state", "location", "currentState"):
                        if key in nested:
                            next_state = self.parse_state_value(nested[key])
                            if next_state is not None:
                                break

        if reward is None:
            raise GridWorldAPIError(f"Could not parse reward from move response: {data}")

        message_parts = [str(data.get("message", ""))]
        for parent in ("data", "result", "details", "payload"):
            nested = data.get(parent)
            if isinstance(nested, dict) and "message" in nested:
                message_parts.append(str(nested["message"]))
        message = " ".join(message_parts).lower()

        invalid_move = any(
            token in message for token in ("invalid", "illegal", "not allowed", "forbidden")
        )
        blocked = any(
            token in message for token in ("blocked", "wall", "obstacle", "cannot move", "can't move")
        )

        if world_id == -1 or next_state == -1:
            done = True

        if any(token in message for token in ("exit", "terminal", "finished", "goal", "win")):
            done = True

        if not done and next_state is None and (blocked or invalid_move):
            next_state = current_state

        return MoveOutcome(
            reward=reward,
            next_state=next_state,
            done=done,
            blocked=blocked,
            invalid_move=invalid_move,
            raw=data,
        )


class QTableStore:
    def __init__(self, storage_dir: str, world_id: int):
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.q_path = self.storage / f"world_{world_id}_q.npy"
        self.meta_path = self.storage / f"world_{world_id}_meta.json"

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as handle:
                np.save(handle, array)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def load_q_table(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.q_path.exists():
            q_table = np.load(self.q_path)
            if q_table.shape != shape:
                raise ValueError(f"Saved Q table shape {q_table.shape} does not match expected {shape}")
            return q_table
        return np.zeros(shape, dtype=np.float32)

    def load_meta(self, cfg: QConfig) -> Dict:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        else:
            meta = {}

        meta.setdefault("episodes_completed", 0)
        meta.setdefault("epsilon", cfg.epsilon)
        meta.setdefault("best_episode_reward", None)
        meta.setdefault("last_enter_ts", 0.0)
        meta.setdefault("last_move_ts", 0.0)
        meta.setdefault("history", [])
        meta.setdefault("grid", {"rows": cfg.rows, "cols": cfg.cols})
        meta.setdefault("metrics", {})
        meta.setdefault("totals", {})
        meta.setdefault("warnings", [])
        return meta

    def save(self, q_table: np.ndarray, meta: Dict) -> None:
        self._atomic_write_npy(self.q_path, q_table)
        self._atomic_write_json(self.meta_path, meta)


class QLearner:
    def __init__(self, alpha: float, gamma: float, num_states: int, num_actions: int):
        self.alpha = alpha
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions

    def choose_action(self, q_table: np.ndarray, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        row = q_table[state]
        best_value = np.max(row)
        best_indices = np.flatnonzero(np.isclose(row, best_value))
        return int(random.choice(best_indices.tolist()))

    def update_q(
        self,
        q_table: np.ndarray,
        state: int,
        action_idx: int,
        reward: float,
        next_state: Optional[int],
        done: bool,
    ) -> float:
        current_q = float(q_table[state, action_idx])
        next_value = 0.0
        if not done and next_state is not None and 0 <= next_state < self.num_states:
            greedy_action = int(np.argmax(q_table[next_state]))
            next_value = float(q_table[next_state, greedy_action])
        target = reward + self.gamma * next_value
        td_error = target - current_q
        q_table[state, action_idx] = current_q + self.alpha * td_error
        return float(td_error)


class QLearningAgent:
    def __init__(self, config: QConfig):
        self.cfg = config
        self.grid = GridSpec(rows=config.rows, cols=config.cols)
        self.actions = list(config.actions)
        self.num_actions = len(self.actions)
        self.store = QTableStore(config.storage_dir, config.world_id)
        self.meta = self.store.load_meta(config)
        saved_grid = self.meta.get("grid", {})
        saved_rows = int(saved_grid.get("rows", self.grid.rows))
        saved_cols = int(saved_grid.get("cols", self.grid.cols))
        self.grid = GridSpec(rows=saved_rows, cols=saved_cols)
        self.meta["grid"] = {"rows": self.grid.rows, "cols": self.grid.cols}
        self.q = self.store.load_q_table((self.grid.num_states, self.num_actions))
        self.client = GridWorldClient(config.team_id, config.api_key, config.user_id)
        self.parser = GridWorldResponseParser(self.grid)
        self.learner = QLearner(config.alpha, config.gamma, self.grid.num_states, self.num_actions)

    def save(self) -> None:
        self.store.save(self.q, self.meta)

    def _respect_rate_limit(self, key: str, min_delay: float) -> None:
        last_ts = float(self.meta.get(key, 0.0))
        elapsed = time.time() - last_ts
        wait = max(0.0, min_delay - elapsed)
        if wait > 0:
            print(f"Waiting {wait:.1f}s to respect API rate limit...")
            time.sleep(wait)

    def _mark_call(self, key: str) -> None:
        self.meta[key] = time.time()

    def _safe_get_location(self) -> Optional[int]:
        attempts = max(1, self.cfg.location_retry_attempts)
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                state = self.parser.parse_location_state(self.client.get_location())
                if self.grid.contains_state(state):
                    return state
                last_error = f"location API returned an invalid state: {state}"
            except Exception as exc:
                last_error = str(exc)

            if attempt < attempts:
                print(
                    f"Warning: location lookup attempt {attempt}/{attempts} failed: {last_error}. "
                    f"Retrying in {self.cfg.location_retry_delay_sec:.1f}s..."
                )
                time.sleep(self.cfg.location_retry_delay_sec)

        print(f"Warning: could not read current location after {attempts} attempt(s): {last_error}")
        return None

    def _bump_total(self, key: str, delta: int = 1) -> None:
        totals = self.meta.setdefault("totals", {})
        totals[key] = int(totals.get(key, 0)) + delta

    def _record_warning(self, message: str) -> None:
        warnings = self.meta.setdefault("warnings", [])
        warnings.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": message,
            }
        )
        if len(warnings) > self.cfg.history_limit:
            del warnings[:-self.cfg.history_limit]

    def _update_running_metrics(self, episode_summary: Dict) -> None:
        metrics = self.meta.setdefault("metrics", {})
        rewards = [item["total_reward"] for item in self.meta.get("history", [])]
        window = rewards[-10:]
        metrics["moving_avg_reward_10"] = float(sum(window) / len(window)) if window else 0.0
        metrics["episodes_recorded"] = len(rewards)
        history = self.meta.get("history", [])
        if history:
            metrics["terminal_rate"] = float(sum(1 for item in history if item["ended"]) / len(history))
            metrics["blocked_move_rate"] = float(
                sum(item["blocked_moves"] for item in history) / max(1, sum(item["steps"] for item in history))
            )
            metrics["invalid_move_rate"] = float(
                sum(item["invalid_moves"] for item in history) / max(1, sum(item["steps"] for item in history))
            )
        metrics["last_episode_reward"] = float(episode_summary["total_reward"])
        metrics["best_episode_reward"] = self.meta.get("best_episode_reward")

    def enter_world_once(self) -> int:
        self._respect_rate_limit("last_enter_ts", self.cfg.enter_delay_sec)
        try:
            enter_response = self.client.enter_world(self.cfg.world_id)
            self._mark_call("last_enter_ts")

            print("ENTER RESPONSE:")
            print(json.dumps(enter_response, indent=2))

            state = self.parser.parse_location_state(enter_response)
            if state is None:
                loc = self.client.get_location()
                print("LOCATION RESPONSE:")
                print(json.dumps(loc, indent=2))
                state = self.parser.parse_location_state(loc)

            if state is None:
                raise GridWorldAPIError(
                    "Could not determine initial state after entering the world. "
                    "See ENTER RESPONSE / LOCATION RESPONSE above."
                )

            return state

        except GridWorldAPIError as exc:
            message = str(exc).lower()
            if "already in a world" in message or "inconsistent state" in message:
                loc = self.client.get_location()
                print("RECOVERY LOCATION RESPONSE:")
                print(json.dumps(loc, indent=2))
                state = self.parser.parse_location_state(loc)
                if self.grid.contains_state(state):
                    return state
            raise

    def get_current_state(self) -> int:
        state = self._safe_get_location()
        if self.grid.contains_state(state):
            return state
        raise GridWorldAPIError("Could not determine current state from location API.")

    def run_episode(self, episode_index: int, initial_state: int) -> Dict:
        epsilon = float(self.meta.get("epsilon", self.cfg.epsilon))
        state = initial_state
        total_reward = 0.0
        total_td_error = 0.0
        done = False
        blocked_moves = 0
        invalid_moves = 0
        visited_states = [state]
        action_counts = {action: 0 for action in self.actions}
        step = 0

        for step in range(1, self.cfg.max_steps_per_episode + 1):
            if not self.grid.contains_state(state):
                raise GridWorldAPIError(f"Invalid state index: {state}")

            self._bump_total(f"state_{state}_visits")
            action_idx = self.learner.choose_action(self.q, state, epsilon)
            action = self.actions[action_idx]
            action_counts[action] += 1
            self._bump_total(f"action_{action}_count")

            self._respect_rate_limit("last_move_ts", self.cfg.move_delay_sec)
            move_data = self.client.move(self.cfg.world_id, action)
            self._mark_call("last_move_ts")

            outcome = self.parser.parse_move_result(move_data, current_state=state)
            reward = outcome.reward
            next_state = outcome.next_state
            done = outcome.done
            blocked_moves += int(outcome.blocked)
            invalid_moves += int(outcome.invalid_move)

            if next_state is None and not done:
                next_state = self._safe_get_location()
                if next_state is None:
                    if not self.cfg.infer_self_loop_on_missing_state:
                        raise GridWorldAPIError(
                            "Move response did not include next state, and fallback location lookup also failed."
                        )
                    next_state = state
                    self._bump_total("inferred_self_loops")
                    warning = (
                        "Move response did not include next state, and fallback location lookup failed. "
                        f"Assuming self-loop at state {state}. Raw response: {json.dumps(outcome.raw, sort_keys=True)}"
                    )
                    self._record_warning(warning)
                    print(f"Warning: {warning}")

            td_error = self.learner.update_q(self.q, state, action_idx, reward, next_state, done)
            total_td_error += abs(td_error)
            total_reward += reward

            if next_state is not None:
                visited_states.append(next_state)

            if self.cfg.verbose:
                print(
                    f"Episode {episode_index} | Step {step:03d} | State {state:4d} | "
                    f"Action {action:>5s} | Reward {reward:8.3f} | "
                    f"Next {str(next_state):>4s} | Done={done} | "
                    f"Blocked={outcome.blocked} | Invalid={outcome.invalid_move}"
                )

            if done:
                self._bump_total("terminal_episodes")
                break

            state = next_state

        epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)
        self.meta["epsilon"] = epsilon
        self.meta["episodes_completed"] = int(self.meta.get("episodes_completed", 0)) + 1

        best = self.meta.get("best_episode_reward")
        if best is None or total_reward > best:
            self.meta["best_episode_reward"] = total_reward

        episode_summary = {
            "episode": episode_index,
            "total_reward": float(total_reward),
            "steps": step,
            "ended": done,
            "epsilon_after": float(epsilon),
            "unique_states": len(set(visited_states)),
            "blocked_moves": blocked_moves,
            "invalid_moves": invalid_moves,
            "mean_abs_td_error": float(total_td_error / max(1, step)),
            "action_counts": action_counts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        history = self.meta.setdefault("history", [])
        history.append(episode_summary)
        if len(history) > self.cfg.history_limit:
            del history[:-self.cfg.history_limit]

        self._update_running_metrics(episode_summary)
        return episode_summary

    def train(self) -> None:
        current_state = self.enter_world_once()
        for episode in range(1, self.cfg.max_episodes + 1):
            try:
                summary = self.run_episode(episode, current_state)
                print("-" * 90)
                print("Episode summary:", json.dumps(summary, indent=2))
                print("Metrics:", json.dumps(self.meta.get("metrics", {}), indent=2))
                print("-" * 90)
                if summary["ended"]:
                    print("Episode reached a terminal state. Stopping training because re-enter is disabled.")
                    break
                current_state = self.get_current_state()
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as exc:
                print(f"Episode {episode} failed: {exc}")
                self.save()
                raise

            if episode % self.cfg.autosave_every == 0:
                self.save()

        self.save()
        print(f"Saved Q-table to {self.store.q_path}")
        print(f"Saved metadata to {self.store.meta_path}")


def parse_args() -> QConfig:
    parser = argparse.ArgumentParser(
        description="Q-learning agent for the NoteXponential grid world API"
    )
    parser.add_argument("--team-id", type=int, default=int(os.getenv("GRIDWORLD_TEAM_ID", "0")))
    parser.add_argument("--api-key", type=str, default=os.getenv("GRIDWORLD_API_KEY", ""))
    parser.add_argument("--user-id", type=str, default=os.getenv("GRIDWORLD_USER_ID", ""))
    parser.add_argument("--world-id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--move-delay", type=float, default=2.0)
    parser.add_argument("--enter-delay", type=float, default=600.0)
    parser.add_argument("--storage-dir", type=str, default="q_data")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument("--location-retries", type=int, default=3)
    parser.add_argument("--location-retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--strict-missing-state",
        action="store_true",
        help="Fail the episode if a move response and fallback location lookup both omit the next state.",
    )
    parser.add_argument(
        "--actions",
        nargs=4,
        default=DEFAULT_ACTIONS,
        help="Exactly 4 action strings, e.g. N S E W",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.team_id or not args.api_key or not args.user_id:
        parser.error("team-id, api-key, and user-id are required, either as flags or environment variables.")

    return QConfig(
        team_id=args.team_id,
        api_key=args.api_key,
        user_id=args.user_id,
        world_id=args.world_id,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        max_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        move_delay_sec=args.move_delay,
        enter_delay_sec=args.enter_delay,
        storage_dir=args.storage_dir,
        actions=tuple(args.actions),
        verbose=not args.quiet,
        rows=args.rows,
        cols=args.cols,
        history_limit=args.history_limit,
        infer_self_loop_on_missing_state=not args.strict_missing_state,
        location_retry_attempts=args.location_retries,
        location_retry_delay_sec=args.location_retry_delay,
    )


def main() -> None:
    cfg = parse_args()
    agent = QLearningAgent(cfg)
    agent.train()


if __name__ == "__main__":
    main()
