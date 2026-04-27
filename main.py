import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from collections import deque
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
RESET_URL = "https://www.notexponential.com/aip2pgaming/api/rl/reset.php"
DEFAULT_ACTIONS = ["N", "S", "E", "W"]
ACTION_TO_IDX = {action: idx for idx, action in enumerate(DEFAULT_ACTIONS)}
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
    epsilon_min: float = 0.2
    epsilon_decay: float = 0.999
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
    null_state_terminal_reward_abs: float = 1000.0
    exploration_bonus: float = 1.0
    state_novelty_bonus: float = 0.25
    unvisited_state_bonus: float = 5.0
    frontier_bonus: float = 1.5
    eval_mode: bool = False
    goal_optimization_mode: bool = False
    frontier_search_prob: float = 0.85
    loop_penalty_weight: float = 0.1
    hazard_penalty_multiplier: float = 3.0
    goal_short_path_bonus: float = 300.0
    goal_q_bias_multiplier: float = 2.0
    goal_reward_gradient_weight: float = 1.0
    goal_backtrack_penalty: float = 0.3
    campaign_world_start: int = 1
    campaign_world_end: int = 10
    traversals_per_world: int = 5
    allow_world_switch: bool = False
    reset_otp: str = ""


@dataclass(frozen=True)
class LocationInfo:
    world_id: int
    state: Optional[int]


@dataclass(frozen=True)
class MoveOutcome:
    reward: float
    next_state: Optional[int]
    done: bool
    blocked: bool
    invalid_move: bool
    assumed_terminal: bool
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

    def reset_team(self, otp: str) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{RESET_URL}?teamId={self.team_id}&otp={otp}",
            *self._headers(),
        ]
        return self._check_response(run_curl(cmd))

class GridWorldResponseParser:
    def __init__(self, grid: GridSpec, null_state_terminal_reward_abs: float = 1000.0):
        self.grid = grid
        self.null_state_terminal_reward_abs = abs(null_state_terminal_reward_abs)

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

    def parse_location_world_id(self, data: Dict) -> Optional[int]:
        possible_keys = ["world", "worldId", "currentWorld"]

        def try_extract(obj: Dict) -> Optional[int]:
            if not isinstance(obj, dict):
                return None
            return self._extract_first_int(obj, possible_keys)

        world_id = try_extract(data)
        if world_id is not None:
            return world_id

        for parent in ("data", "result", "details", "payload"):
            nested = data.get(parent)
            if isinstance(nested, dict):
                world_id = try_extract(nested)
                if world_id is not None:
                    return world_id
        return None

    def parse_location_info(self, data: Dict) -> LocationInfo:
        world_id = self.parse_location_world_id(data)
        state = self.parse_location_state(data)
        if world_id is None:
            raise GridWorldAPIError(f"Could not parse world id from location response: {data}")
        return LocationInfo(world_id=world_id, state=state)

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

        invalid_move = False
        blocked = False

        if world_id == -1 or next_state == -1:
            done = True

        assumed_terminal = False

        if next_state is None and abs(reward) >= self.null_state_terminal_reward_abs:
            done = True
            assumed_terminal = True

        return MoveOutcome(
            reward=reward,
            next_state=next_state,
            done=done,
            blocked=blocked,
            invalid_move=invalid_move,
            assumed_terminal=assumed_terminal,
            raw=data,
        )


class QTableStore:
    def __init__(self, storage_dir: str, world_id: int):
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.q_path = self.storage / f"world_{world_id}_q.npy"
        self.visits_path = self.storage / f"world_{world_id}_visits.npy"
        self.state_visits_path = self.storage / f"world_{world_id}_state_visits.npy"
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

    def load_action_visits(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.visits_path.exists():
            visits = np.load(self.visits_path)
            if visits.shape != shape:
                raise ValueError(f"Saved visit table shape {visits.shape} does not match expected {shape}")
            return visits
        return np.zeros(shape, dtype=np.float32)

    def load_state_visits(self, num_states: int) -> np.ndarray:
        if self.state_visits_path.exists():
            visits = np.load(self.state_visits_path)
            if visits.shape != (num_states,):
                raise ValueError(
                    f"Saved state visit table shape {visits.shape} does not match expected {(num_states,)}"
                )
            return visits
        return np.zeros((num_states,), dtype=np.float32)

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
        meta.setdefault("bad_transitions", [])
        meta.setdefault("safe_edges", {})
        meta.setdefault("tried_transitions", [])
        meta.setdefault("discovered_states", [])
        meta.setdefault("known_terminal_events", [])
        meta.setdefault("state_reward_avg", {})
        meta.setdefault("goal_entry", None)  # {"state": pre-terminal state, "action": action that reaches goal}
        meta.setdefault("best_known_goal_path", [])
        return meta

    def save(
        self,
        q_table: np.ndarray,
        action_visits: np.ndarray,
        state_visits: np.ndarray,
        meta: Dict,
    ) -> None:
        self._atomic_write_npy(self.q_path, q_table)
        self._atomic_write_npy(self.visits_path, action_visits)
        self._atomic_write_npy(self.state_visits_path, state_visits)
        self._atomic_write_json(self.meta_path, meta)


class CampaignStore:
    def __init__(self, storage_dir: str):
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.storage / "campaign_progress.json"

    def load(self, cfg: QConfig) -> Dict:
        if self.progress_path.exists():
            with open(self.progress_path, "r", encoding="utf-8") as handle:
                progress = json.load(handle)
        else:
            progress = {}

        worlds = progress.setdefault("worlds", {})
        for world_id in range(cfg.campaign_world_start, cfg.campaign_world_end + 1):
            world_progress = worlds.setdefault(str(world_id), {})
            legacy_traversals_completed = world_progress.get("traversals_completed")
            world_progress.setdefault("goal_found", False)
            world_progress.setdefault("goal_reward", None)
            world_progress.setdefault("hazards_found", 0)
            legacy_goal_hits_completed = world_progress.get("optimization_runs_completed")
            world_progress.setdefault("goal_hits_completed", 0)
            world_progress.setdefault("optimization_runs_completed", 0)
            world_progress.setdefault("terminals_found", 0)
            world_progress.setdefault("last_terminal_kind", None)
            world_progress.setdefault("last_terminal_reward", None)
            world_progress.setdefault("last_known_state", None)
            # Backward compatibility with older progress files.
            if legacy_goal_hits_completed is not None and int(world_progress["goal_hits_completed"]) == 0:
                world_progress["goal_hits_completed"] = int(legacy_goal_hits_completed)
            if legacy_traversals_completed is not None and int(world_progress["goal_hits_completed"]) == 0:
                world_progress["goal_hits_completed"] = int(legacy_traversals_completed)
            if legacy_traversals_completed is not None and int(world_progress["optimization_runs_completed"]) == 0:
                world_progress["optimization_runs_completed"] = int(legacy_traversals_completed)
            world_progress.pop("traversals_completed", None)
        progress.setdefault("active_world", None)
        progress.setdefault("last_enter_ts", 0.0)
        progress.setdefault("last_summary", None)
        progress.setdefault("campaign_world_start", cfg.campaign_world_start)
        progress.setdefault("campaign_world_end", cfg.campaign_world_end)
        progress.setdefault("traversals_per_world", cfg.traversals_per_world)
        return progress

    def save(self, progress: Dict) -> None:
        QTableStore._atomic_write_json(self.progress_path, progress)


class QLearner:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        num_states: int,
        num_actions: int,
        exploration_bonus: float = 0.0,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        unvisited_state_bonus: float = 0.0,
        frontier_bonus: float = 0.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.exploration_bonus = exploration_bonus
        self.rows = rows
        self.cols = cols
        self.unvisited_state_bonus = unvisited_state_bonus
        self.frontier_bonus = frontier_bonus

    def _neighbor_state(self, state: int, action_idx: int) -> Optional[int]:
        """Predict neighbor using the API's observed action mapping.

        From your logs:
        - N increases state by +1  => column + 1
        - S decreases state by -1  => column - 1
        - E increases state by +40 => row + 1
        - W decreases state by -40 => row - 1
        """
        row = state // self.cols
        col = state % self.cols
        if action_idx == ACTION_TO_IDX["N"]:
            col += 1
        elif action_idx == ACTION_TO_IDX["S"]:
            col -= 1
        elif action_idx == ACTION_TO_IDX["E"]:
            row += 1
        elif action_idx == ACTION_TO_IDX["W"]:
            row -= 1

        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row * self.cols + col
        return None

    def choose_action(
        self,
        q_table: np.ndarray,
        state: int,
        epsilon: float,
        action_visits: Optional[np.ndarray] = None,
        state_visits: Optional[np.ndarray] = None,
    ) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        row = q_table[state].astype(np.float64, copy=True)
        if action_visits is not None and self.exploration_bonus > 0:
            row += self.exploration_bonus / np.sqrt(action_visits[state] + 1.0)
        if state_visits is not None:
            for action_idx in range(self.num_actions):
                neighbor_state = self._neighbor_state(state, action_idx)
                if neighbor_state is None:
                    row[action_idx] -= 10_000.0
                    continue
                if state_visits[neighbor_state] == 0 and self.unvisited_state_bonus > 0:
                    row[action_idx] += self.unvisited_state_bonus
                elif self.frontier_bonus > 0:
                    row[action_idx] += self.frontier_bonus / np.sqrt(state_visits[neighbor_state] + 1.0)
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
        self.action_visits = self.store.load_action_visits((self.grid.num_states, self.num_actions))
        self.state_visits = self.store.load_state_visits(self.grid.num_states)
        self.client = GridWorldClient(config.team_id, config.api_key, config.user_id)
        self.parser = GridWorldResponseParser(
            self.grid,
            null_state_terminal_reward_abs=config.null_state_terminal_reward_abs,
        )
        self.learner = QLearner(
            config.alpha,
            config.gamma,
            self.grid.num_states,
            self.num_actions,
            exploration_bonus=config.exploration_bonus,
            rows=self.grid.rows,
            cols=self.grid.cols,
            unvisited_state_bonus=config.unvisited_state_bonus,
            frontier_bonus=config.frontier_bonus,
        )

    def save(self) -> None:
        self.store.save(self.q, self.action_visits, self.state_visits, self.meta)

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

    def get_location_info(self) -> LocationInfo:
        return self.parser.parse_location_info(self.client.get_location())

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

    def _state_novelty_reward(self, next_state: Optional[int], done: bool) -> float:
        if self.cfg.eval_mode or done or next_state is None or not self.grid.contains_state(next_state):
            return 0.0
        if self.cfg.state_novelty_bonus <= 0:
            return 0.0
        return float(self.cfg.state_novelty_bonus / np.sqrt(self.state_visits[next_state] + 1.0))

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
        metrics["last_episode_policy_reward"] = float(episode_summary["total_policy_reward"])
        metrics["best_episode_reward"] = self.meta.get("best_episode_reward")
        metrics["exploration_bonus"] = float(self.cfg.exploration_bonus)
        metrics["state_novelty_bonus"] = float(self.cfg.state_novelty_bonus)
        metrics["unvisited_state_bonus"] = float(self.cfg.unvisited_state_bonus)
        metrics["frontier_bonus"] = float(self.cfg.frontier_bonus)
        metrics["epsilon_min"] = float(self.cfg.epsilon_min)
        metrics["eval_mode"] = bool(self.cfg.eval_mode)
        metrics["goal_optimization_mode"] = bool(self.cfg.goal_optimization_mode)
        metrics["frontier_search_prob"] = float(self.cfg.frontier_search_prob)
        metrics["known_bad_transitions"] = len(self.meta.get("bad_transitions", []))
        metrics["known_safe_edges"] = len(self.meta.get("safe_edges", {}))
        metrics["discovered_states"] = len(self.meta.get("discovered_states", []))

    def enter_world_once(self) -> int:
        # self._respect_rate_limit("last_enter_ts", self.cfg.enter_delay_sec)
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

    def _transition_key(self, state: int, action: str) -> str:
        return f"{state}:{action}"

    def _remember_discovered_state(self, state: Optional[int]) -> None:
        if state is None or not self.grid.contains_state(state):
            return
        discovered = self.meta.setdefault("discovered_states", [])
        state_int = int(state)
        if state_int not in discovered:
            discovered.append(state_int)

    def _remember_safe_edge(self, state: int, action: str, next_state: Optional[int]) -> None:
        if next_state is None or not self.grid.contains_state(next_state):
            return
        safe_edges = self.meta.setdefault("safe_edges", {})
        safe_edges[self._transition_key(state, action)] = int(next_state)
        tried = self.meta.setdefault("tried_transitions", [])
        key = self._transition_key(state, action)
        if key not in tried:
            tried.append(key)
        self._remember_discovered_state(state)
        self._remember_discovered_state(next_state)

    def _remember_bad_transition(self, state: int, action: str, reward: float) -> None:
        bad = self.meta.setdefault("bad_transitions", [])
        key = self._transition_key(state, action)
        if key not in bad:
            bad.append(key)
        events = self.meta.setdefault("known_terminal_events", [])
        events.append({
            "state": int(state),
            "action": action,
            "reward": float(reward),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(events) > self.cfg.history_limit:
            del events[:-self.cfg.history_limit]

    def _update_state_reward_avg(self, next_state: Optional[int], reward: float) -> None:
        """Track average immediate reward for each state as a local gradient signal."""
        if next_state is None or not self.grid.contains_state(next_state):
            return
        state_reward_avg = self.meta.setdefault("state_reward_avg", {})
        key = str(int(next_state))
        old_avg = float(state_reward_avg.get(key, reward))
        state_reward_avg[key] = float(0.9 * old_avg + 0.1 * reward)

    def _choose_goal_biased_action(self, state: int, epsilon: float, prev_state: Optional[int]) -> int:
        """Goal optimization policy: Q boost + reward-gradient + anti-backtracking."""
        bad = set(self.meta.get("bad_transitions", []))

        if random.random() < epsilon:
            candidates = []
            for idx, act in enumerate(self.actions):
                neighbor = self.learner._neighbor_state(state, idx)
                if neighbor is None:
                    continue
                if self._transition_key(state, act) in bad:
                    continue
                candidates.append(idx)
            if candidates:
                return int(random.choice(candidates))
            return random.randrange(self.num_actions)

        row = self.q[state].astype(np.float64, copy=True)
        row *= float(self.cfg.goal_q_bias_multiplier)

        state_reward_avg = self.meta.get("state_reward_avg", {})
        for idx, act in enumerate(self.actions):
            neighbor = self.learner._neighbor_state(state, idx)
            key = self._transition_key(state, act)

            if neighbor is None:
                row[idx] -= 10000.0
                continue
            if key in bad:
                row[idx] -= 10000.0
                continue

            avg_reward = float(state_reward_avg.get(str(int(neighbor)), 0.0))
            row[idx] += float(self.cfg.goal_reward_gradient_weight) * avg_reward

            if prev_state is not None and neighbor == prev_state:
                row[idx] -= float(self.cfg.goal_backtrack_penalty)

        best_value = np.max(row)
        best_indices = np.flatnonzero(np.isclose(row, best_value))
        return int(random.choice(best_indices.tolist()))

    def _avoid_bad_action(self, state: int, action_idx: int) -> int:
        bad = set(self.meta.get("bad_transitions", []))
        if not bad:
            return action_idx
        candidates = [idx for idx, act in enumerate(self.actions) if self._transition_key(state, act) not in bad]
        if not candidates:
            return action_idx
        action = self.actions[action_idx]
        if self._transition_key(state, action) not in bad:
            return action_idx
        q_values = self.q[state, candidates]
        return int(candidates[int(np.argmax(q_values))])

    def _frontier_action(self, state: int) -> Optional[int]:
        bad = set(self.meta.get("bad_transitions", []))
        tried = set(self.meta.get("tried_transitions", []))
        safe_edges = self.meta.get("safe_edges", {})

        current_candidates = []
        for idx, act in enumerate(self.actions):
            key = self._transition_key(state, act)
            neighbor = self.learner._neighbor_state(state, idx)
            if neighbor is None or key in bad:
                continue
            if key not in tried:
                current_candidates.append(idx)
        if current_candidates:
            min_visits = min(self.action_visits[state, idx] for idx in current_candidates)
            best = [idx for idx in current_candidates if self.action_visits[state, idx] == min_visits]
            return int(random.choice(best))

        graph: Dict[int, List[Tuple[int, int]]] = {}
        for key, next_state in safe_edges.items():
            try:
                s_str, action = key.split(":", 1)
                s_int = int(s_str)
                a_idx = ACTION_TO_IDX.get(action)
                n_int = int(next_state)
            except Exception:
                continue
            if a_idx is None:
                continue
            graph.setdefault(s_int, []).append((n_int, a_idx))

        queue: List[Tuple[int, Optional[int]]] = [(state, None)]
        seen = {state}
        q_pos = 0
        while q_pos < len(queue):
            cur, first_action = queue[q_pos]
            q_pos += 1
            for idx, act in enumerate(self.actions):
                key = self._transition_key(cur, act)
                neighbor = self.learner._neighbor_state(cur, idx)
                if neighbor is None or key in bad:
                    continue
                if key not in tried:
                    return int(first_action if first_action is not None else idx)
            for nxt, a_idx in graph.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, a_idx if first_action is None else first_action))
        return None

    def _choose_exploration_action(self, state: int, epsilon: float, prev_state: Optional[int] = None) -> int:
        if self.cfg.goal_optimization_mode:
            return self._choose_goal_biased_action(state, epsilon, prev_state)

        if (not self.cfg.eval_mode and random.random() < self.cfg.frontier_search_prob):
            frontier_idx = self._frontier_action(state)
            if frontier_idx is not None:
                return self._avoid_bad_action(state, frontier_idx)
        action_idx = self.learner.choose_action(
            self.q, state, epsilon, self.action_visits, None if self.cfg.eval_mode else self.state_visits,
        )
        return self._avoid_bad_action(state, action_idx)

    def _remember_goal_entry(self, state: int, action: str, reward: float, step: int) -> None:
        """Remember the pre-terminal state and action that reached the goal.

        Many GridWorld APIs return world=-1 / next_state=-1 for terminal states,
        so the safest representation of the goal is: be at `state`, take `action`.
        """
        goal_entry = {
            "state": int(state),
            "action": action,
            "reward": float(reward),
            "step_found": int(step),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta["goal_entry"] = goal_entry
        print(f"Saved goal entry: from state {state}, take action {action}.")

    def find_shortest_known_goal_path(self, start_state: int) -> Optional[List[str]]:
        """Find shortest known path to the saved goal entry using safe_edges.

        Returns a list of actions. The last action is the terminal goal action.
        If safe_edges are incomplete, returns None and the caller should fall back to Q-learning.
        """
        goal_entry = self.meta.get("goal_entry")
        if not goal_entry:
            return None

        try:
            goal_state = int(goal_entry["state"])
            goal_action = str(goal_entry["action"])
        except Exception:
            return None

        safe_edges = self.meta.get("safe_edges", {})
        bad = set(self.meta.get("bad_transitions", []))

        # Build adjacency list from known safe transitions.
        graph: Dict[int, List[Tuple[int, str]]] = {}
        for key, nxt in safe_edges.items():
            try:
                s_str, action = key.split(":", 1)
                s_int = int(s_str)
                n_int = int(nxt)
            except Exception:
                continue
            if key in bad:
                continue
            if not self.grid.contains_state(s_int) or not self.grid.contains_state(n_int):
                continue
            graph.setdefault(s_int, []).append((n_int, action))

        # If we are already at the pre-goal state, just take the known terminal action.
        if int(start_state) == goal_state:
            return [goal_action]

        queue = deque([(int(start_state), [])])
        visited = {int(start_state)}

        while queue:
            state, path = queue.popleft()
            for next_state, action in graph.get(state, []):
                if next_state in visited:
                    continue
                new_path = path + [action]
                if next_state == goal_state:
                    return new_path + [goal_action]
                visited.add(next_state)
                queue.append((next_state, new_path))

        return None

    def follow_planned_goal_path(self, episode_index: int, initial_state: int, path: List[str]) -> Dict:
        """Follow a BFS-planned path, while still updating Q-values and graph memory.

        This is not blind replay: if an edge fails or becomes hazardous, it is recorded as bad.
        """
        state = initial_state
        total_reward = 0.0
        total_policy_reward = 0.0
        total_td_error = 0.0
        visited_states = [state]
        action_counts = {action: 0 for action in self.actions}
        terminal_reward = None
        terminal_kind = None
        done = False
        ended_by_step_limit = False
        step = 0

        for step, action in enumerate(path[: self.cfg.max_steps_per_episode], start=1):
            if action not in ACTION_TO_IDX:
                continue
            action_idx = ACTION_TO_IDX[action]
            action_counts[action] += 1

            self._respect_rate_limit("last_move_ts", self.cfg.move_delay_sec)
            move_data = self.client.move(self.cfg.world_id, action)
            self._mark_call("last_move_ts")

            outcome = self.parser.parse_move_result(move_data, current_state=state)
            reward = outcome.reward
            next_state = outcome.next_state
            done = outcome.done

            if next_state is None and not done:
                next_state = self._safe_get_location()
                if next_state is None:
                    next_state = state

            self._update_state_reward_avg(next_state, reward)

            if done and reward < 0:
                self._remember_bad_transition(state, action, reward)
                policy_reward = reward * float(self.cfg.hazard_penalty_multiplier)
            elif done and reward > 0:
                self._remember_goal_entry(state, action, reward, step)
                policy_reward = reward + (float(self.cfg.goal_short_path_bonus) / max(1, step))
            else:
                self._remember_safe_edge(state, action, next_state)
                policy_reward = reward

            if not self.cfg.eval_mode:
                td_error = self.learner.update_q(self.q, state, action_idx, policy_reward, next_state, done)
                total_td_error += abs(td_error)

            total_reward += reward
            total_policy_reward += policy_reward
            if next_state is not None:
                visited_states.append(next_state)

            if self.cfg.verbose:
                print(
                    f"[BFS-PATH] Episode {episode_index} | Step {step:03d} | State {state:4d} | "
                    f"Action {action:>5s} | Reward {reward:8.3f} | "
                    f"PolicyReward {policy_reward:8.3f} | Next {str(next_state):>4s} | Done={done}"
                )

            if done:
                terminal_reward = float(reward)
                terminal_kind = "hazard" if reward < 0 else "goal"
                if terminal_kind == "goal":
                    self._remember_goal_entry(state, action, reward, step)
                self._bump_total("terminal_episodes")
                break

            state = int(next_state)

        if not done and step >= self.cfg.max_steps_per_episode:
            done = True
            ended_by_step_limit = True

        summary = {
            "episode": episode_index,
            "total_reward": float(total_reward),
            "total_policy_reward": float(total_policy_reward),
            "steps": step,
            "ended": done,
            "epsilon_after": float(self.meta.get("epsilon", self.cfg.epsilon)),
            "unique_states": len(set(visited_states)),
            "blocked_moves": 0,
            "invalid_moves": 0,
            "assumed_terminal_events": 0,
            "mean_abs_td_error": float(total_td_error / max(1, step)),
            "terminal_reward": terminal_reward,
            "terminal_kind": terminal_kind,
            "ended_by_step_limit": ended_by_step_limit,
            "final_state": None if terminal_kind is not None else (int(state) if self.grid.contains_state(state) else None),
            "action_counts": action_counts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "bfs_shortest_known_goal_path",
            "planned_path_len": len(path),
        }

        history = self.meta.setdefault("history", [])
        history.append(summary)
        if len(history) > self.cfg.history_limit:
            del history[:-self.cfg.history_limit]
        self._update_running_metrics(summary)
        return summary

    def run_episode(self, episode_index: int, initial_state: int) -> Dict:
        epsilon = 0.0 if self.cfg.eval_mode else float(self.meta.get("epsilon", self.cfg.epsilon))
        state = initial_state
        total_reward = 0.0
        total_policy_reward = 0.0
        total_td_error = 0.0
        done = False
        blocked_moves = 0
        invalid_moves = 0
        assumed_terminal_events = 0
        ended_by_step_limit = False
        visited_states = [state]
        episode_state_counts = {}
        action_counts = {action: 0 for action in self.actions}
        step = 0
        terminal_reward = None
        terminal_kind = None

        for step in range(1, self.cfg.max_steps_per_episode + 1):
            if not self.grid.contains_state(state):
                raise GridWorldAPIError(f"Invalid state index: {state}")

            episode_state_counts[state] = episode_state_counts.get(state, 0) + 1

            if not self.cfg.eval_mode:
                self.state_visits[state] += 1.0
            self._bump_total(f"state_{state}_visits")
            prev_state = visited_states[-2] if len(visited_states) > 1 else None
            action_idx = self._choose_exploration_action(state, epsilon, prev_state)
            action = self.actions[action_idx]
            action_counts[action] += 1
            self._bump_total(f"action_{action}_count")
            if not self.cfg.eval_mode:
                self.action_visits[state, action_idx] += 1.0

            # self._respect_rate_limit("last_move_ts", self.cfg.move_delay_sec)
            move_data = self.client.move(self.cfg.world_id, action)
            self._mark_call("last_move_ts")

            outcome = self.parser.parse_move_result(move_data, current_state=state)
            reward = outcome.reward
            next_state = outcome.next_state
            done = outcome.done
            blocked_moves += int(outcome.blocked)
            invalid_moves += int(outcome.invalid_move)
            assumed_terminal_events += int(outcome.assumed_terminal)

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

            self._update_state_reward_avg(next_state, reward)
            novelty_reward = self._state_novelty_reward(next_state, done)
            loop_penalty = 0.0
            if next_state is not None and self.grid.contains_state(next_state):
                # Optimization shaping:
                # - self-loop wastes a move, so punish it strongly.
                # - normal revisits can be necessary, so punish them lightly.
                if next_state == state:
                    loop_penalty = -1.0
                else:
                    revisits = max(0, episode_state_counts.get(next_state, 0) - 1)
                    loop_penalty = -float(self.cfg.loop_penalty_weight) * revisits

            if done and reward < 0:
                self._remember_bad_transition(state, action, reward)
            elif next_state is not None and self.grid.contains_state(next_state):
                self._remember_safe_edge(state, action, next_state)

            if self.cfg.goal_optimization_mode:
                if done and reward > 0:
                    policy_reward = reward + (float(self.cfg.goal_short_path_bonus) / max(1, step))
                elif done and reward < 0:
                    policy_reward = reward * float(self.cfg.hazard_penalty_multiplier)
                else:
                    policy_reward = reward + loop_penalty
            else:
                if done and reward < 0:
                    policy_reward = reward * float(self.cfg.hazard_penalty_multiplier)
                elif done and reward > 0:
                    policy_reward = reward + (float(self.cfg.goal_short_path_bonus) / max(1, step))
                else:
                    policy_reward = reward + novelty_reward + loop_penalty

            td_error = 0.0
            if not self.cfg.eval_mode:
                td_error = self.learner.update_q(self.q, state, action_idx, policy_reward, next_state, done)
                total_td_error += abs(td_error)
            total_reward += reward
            total_policy_reward += policy_reward

            if next_state is not None:
                visited_states.append(next_state)

            if self.cfg.verbose:
                print(
                    f"Episode {episode_index} | Step {step:03d} | State {state:4d} | "
                    f"Action {action:>5s} | Reward {reward:8.3f} | "
                    f"PolicyReward {policy_reward:8.3f} | "
                    f"Next {str(next_state):>4s} | Done={done} | "
                    f"Blocked={outcome.blocked} | Invalid={outcome.invalid_move} | "
                    f"AssumedTerminal={outcome.assumed_terminal}"
                )

            if done:
                terminal_reward = float(reward)
                terminal_kind = "hazard" if reward < 0 else "goal"
                self._bump_total("terminal_episodes")
                if outcome.assumed_terminal:
                    self._bump_total("assumed_terminal_events")
                break

            state = next_state

        if not done and step >= self.cfg.max_steps_per_episode:
            done = True
            ended_by_step_limit = True

        if not self.cfg.eval_mode:
            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)
            self.meta["epsilon"] = epsilon
            self.meta["episodes_completed"] = int(self.meta.get("episodes_completed", 0)) + 1

        best = self.meta.get("best_episode_reward")
        if best is None or total_reward > best:
            self.meta["best_episode_reward"] = total_reward

        episode_summary = {
            "episode": episode_index,
            "total_reward": float(total_reward),
            "total_policy_reward": float(total_policy_reward),
            "steps": step,
            "ended": done,
            "epsilon_after": float(epsilon),
            "unique_states": len(set(visited_states)),
            "blocked_moves": blocked_moves,
            "invalid_moves": invalid_moves,
            "assumed_terminal_events": assumed_terminal_events,
            "mean_abs_td_error": float(total_td_error / max(1, step)),
            "terminal_reward": terminal_reward,
            "terminal_kind": terminal_kind,
            "ended_by_step_limit": ended_by_step_limit,
            "final_state": None if terminal_kind is not None else (int(state) if self.grid.contains_state(state) else None),
            "action_counts": action_counts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        history = self.meta.setdefault("history", [])
        history.append(episode_summary)
        if len(history) > self.cfg.history_limit:
            del history[:-self.cfg.history_limit]

        self._update_running_metrics(episode_summary)
        return episode_summary


    def _state_to_row_col(self, state: int) -> Tuple[int, int]:
        return int(state) // self.grid.cols, int(state) % self.grid.cols

    def _coordinate_action_candidates(
        self,
        state: int,
        target_state: int,
        prev_state: Optional[int] = None,
    ) -> List[int]:
        """Return actions ordered by greedy movement toward target_state.

        Grid encoding is row * cols + col, but the API action names are observed as:
        - N = column + 1
        - S = column - 1
        - E = row + 1
        - W = row - 1

        Hazards are handled by bad_transitions; this only provides a goal-direction fallback.
        """
        row, col = self._state_to_row_col(state)
        target_row, target_col = self._state_to_row_col(target_state)
        d_row = target_row - row
        d_col = target_col - col

        primary: List[str] = []
        secondary: List[str] = []

        vertical = "E" if d_row > 0 else "W" if d_row < 0 else None
        horizontal = "N" if d_col > 0 else "S" if d_col < 0 else None

        if abs(d_row) >= abs(d_col):
            if vertical is not None:
                primary.append(vertical)
            if horizontal is not None:
                primary.append(horizontal)
        else:
            if horizontal is not None:
                primary.append(horizontal)
            if vertical is not None:
                primary.append(vertical)

        # Add remaining actions as fallback, ranked by current Q-value.
        used = set(primary)
        q_ranked = sorted(
            [a for a in self.actions if a not in used],
            key=lambda a: float(self.q[state, ACTION_TO_IDX[a]]),
            reverse=True,
        )
        secondary.extend(q_ranked)

        ordered_actions = primary + secondary
        bad = set(self.meta.get("bad_transitions", []))
        candidates: List[int] = []

        for action in ordered_actions:
            idx = ACTION_TO_IDX[action]
            neighbor = self.learner._neighbor_state(state, idx)
            if neighbor is None:
                continue
            if self._transition_key(state, action) in bad:
                continue
            # Avoid immediately going backward if another candidate exists.
            if prev_state is not None and neighbor == prev_state:
                continue
            candidates.append(idx)

        if candidates:
            return candidates

        # If all non-backtracking candidates were filtered, allow backtracking but still avoid known hazards.
        for action in ordered_actions:
            idx = ACTION_TO_IDX[action]
            neighbor = self.learner._neighbor_state(state, idx)
            if neighbor is None:
                continue
            if self._transition_key(state, action) in bad:
                continue
            candidates.append(idx)

        if candidates:
            return candidates

        # Last resort: any valid in-grid action.
        for action in self.actions:
            idx = ACTION_TO_IDX[action]
            if self.learner._neighbor_state(state, idx) is not None:
                candidates.append(idx)

        return candidates or [random.randrange(self.num_actions)]

    def coordinate_exploit_to_goal(self, episode_index: int, initial_state: int) -> Dict:
        """Exploit toward the known goal entry using coordinates.

        Priority:
        1. If already at the saved pre-goal state, take the saved terminal goal action.
        2. Otherwise, greedily reduce Manhattan distance to the pre-goal state.
        3. Avoid known bad/hazard transitions.
        4. If the chosen direction fails, the transition is remembered and future runs avoid it.

        This is faster than generic Q-learning when there are no walls, but hazards may still
        sit on the direct route, so BFS safe path remains preferred whenever available.
        """
        goal_entry = self.meta.get("goal_entry")
        if not goal_entry:
            raise GridWorldAPIError("No known goal entry for coordinate exploit.")

        goal_state = int(goal_entry["state"])
        goal_action = str(goal_entry["action"])

        state = int(initial_state)
        total_reward = 0.0
        total_policy_reward = 0.0
        total_td_error = 0.0
        visited_states = [state]
        action_counts = {action: 0 for action in self.actions}
        terminal_reward = None
        terminal_kind = None
        done = False
        ended_by_step_limit = False
        step = 0

        print(
            f"[COORD-EXPLOIT] Target pre-goal state={goal_state}, terminal action={goal_action}."
        )

        for step in range(1, self.cfg.max_steps_per_episode + 1):
            prev_state = visited_states[-2] if len(visited_states) >= 2 else None

            if state == goal_state and goal_action in ACTION_TO_IDX:
                action_idx = ACTION_TO_IDX[goal_action]
            else:
                candidates = self._coordinate_action_candidates(state, goal_state, prev_state)
                action_idx = candidates[0]

            action = self.actions[action_idx]
            action_counts[action] += 1

            self._respect_rate_limit("last_move_ts", self.cfg.move_delay_sec)
            move_data = self.client.move(self.cfg.world_id, action)
            self._mark_call("last_move_ts")

            outcome = self.parser.parse_move_result(move_data, current_state=state)
            reward = outcome.reward
            next_state = outcome.next_state
            done = outcome.done

            if next_state is None and not done:
                next_state = self._safe_get_location()
                if next_state is None:
                    next_state = state

            self._update_state_reward_avg(next_state, reward)

            if done and reward < 0:
                self._remember_bad_transition(state, action, reward)
                policy_reward = reward * float(self.cfg.hazard_penalty_multiplier)
            elif done and reward > 0:
                self._remember_goal_entry(state, action, reward, step)
                policy_reward = reward + (float(self.cfg.goal_short_path_bonus) / max(1, step))
            else:
                self._remember_safe_edge(state, action, next_state)

                # For coordinate exploit, use a small Manhattan-distance shaping term.
                # This is only used after goal is known.
                old_dist = abs(self._state_to_row_col(state)[0] - self._state_to_row_col(goal_state)[0]) + abs(
                    self._state_to_row_col(state)[1] - self._state_to_row_col(goal_state)[1]
                )
                new_dist = old_dist
                if next_state is not None and self.grid.contains_state(next_state):
                    new_dist = abs(self._state_to_row_col(int(next_state))[0] - self._state_to_row_col(goal_state)[0]) + abs(
                        self._state_to_row_col(int(next_state))[1] - self._state_to_row_col(goal_state)[1]
                    )
                progress_bonus = 0.25 if new_dist < old_dist else -0.25 if new_dist > old_dist else 0.0
                self_loop_penalty = -1.0 if next_state == state else 0.0
                policy_reward = reward + progress_bonus + self_loop_penalty

            if not self.cfg.eval_mode:
                td_error = self.learner.update_q(self.q, state, action_idx, policy_reward, next_state, done)
                total_td_error += abs(td_error)

            total_reward += reward
            total_policy_reward += policy_reward

            if next_state is not None:
                visited_states.append(int(next_state))

            if self.cfg.verbose:
                print(
                    f"[COORD-EXPLOIT] Episode {episode_index} | Step {step:03d} | State {state:4d} | "
                    f"Action {action:>5s} | Reward {reward:8.3f} | "
                    f"PolicyReward {policy_reward:8.3f} | Next {str(next_state):>4s} | Done={done}"
                )

            if done:
                terminal_reward = float(reward)
                terminal_kind = "hazard" if reward < 0 else "goal"
                self._bump_total("terminal_episodes")
                break

            state = int(next_state)

        if not done and step >= self.cfg.max_steps_per_episode:
            done = True
            ended_by_step_limit = True

        summary = {
            "episode": episode_index,
            "total_reward": float(total_reward),
            "total_policy_reward": float(total_policy_reward),
            "steps": step,
            "ended": done,
            "epsilon_after": float(self.meta.get("epsilon", self.cfg.epsilon)),
            "unique_states": len(set(visited_states)),
            "blocked_moves": 0,
            "invalid_moves": 0,
            "assumed_terminal_events": 0,
            "mean_abs_td_error": float(total_td_error / max(1, step)),
            "terminal_reward": terminal_reward,
            "terminal_kind": terminal_kind,
            "ended_by_step_limit": ended_by_step_limit,
            "final_state": None if terminal_kind is not None else (int(state) if self.grid.contains_state(state) else None),
            "action_counts": action_counts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "coordinate_exploit",
            "target_goal_state": goal_state,
            "target_goal_action": goal_action,
        }

        history = self.meta.setdefault("history", [])
        history.append(summary)
        if len(history) > self.cfg.history_limit:
            del history[:-self.cfg.history_limit]

        self._update_running_metrics(summary)
        return summary


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


class CampaignRunner:
    def __init__(self, config: QConfig):
        self.cfg = config
        self.client = GridWorldClient(config.team_id, config.api_key, config.user_id)
        self.parser = GridWorldResponseParser(
            GridSpec(rows=config.rows, cols=config.cols),
            null_state_terminal_reward_abs=config.null_state_terminal_reward_abs,
        )
        self.store = CampaignStore(config.storage_dir)
        self.progress = self.store.load(config)

        self.store.save(self.progress)
    def _save_progress(self) -> None:

        self.store.save(self.progress)
    def _world_progress(self, world_id: int) -> Dict:
        return self.progress["worlds"][str(world_id)]

    def _world_is_complete(self, world_id: int) -> bool:
        world_progress = self._world_progress(world_id)
        return bool(world_progress["goal_found"]) and (
            int(world_progress["goal_hits_completed"]) >= self.cfg.traversals_per_world
        )

    def _next_incomplete_world(self) -> Optional[int]:
        for world_id in range(self.cfg.campaign_world_start, self.cfg.campaign_world_end + 1):
            if not self._world_is_complete(world_id):
                return world_id
        return None

    def _respect_enter_rate_limit(self) -> None:
        last_ts = float(self.progress.get("last_enter_ts", 0.0))
        elapsed = time.time() - last_ts
        wait = max(0.0, self.cfg.enter_delay_sec - elapsed)
        if wait > 0:
            print(f"Waiting {wait:.1f}s before next enter to respect campaign enter delay...")
            time.sleep(wait)

    def _mark_enter(self) -> None:
        self.progress["last_enter_ts"] = time.time()
        self._save_progress()

    def _get_location_info(self) -> LocationInfo:
        return self.parser.parse_location_info(self.client.get_location())

    def _build_agent(self, world_id: int) -> QLearningAgent:
        world_cfg = QConfig(**{**self.cfg.__dict__, "world_id": world_id})
        return QLearningAgent(world_cfg)

    def _force_switch_world(self, current_world_id: int) -> None:
        if not self.cfg.allow_world_switch:
            raise GridWorldAPIError(f"Already in world {current_world_id}; finish it")
        if not self.cfg.reset_otp:
            raise GridWorldAPIError(
                "World switching requires a reset OTP. Provide --reset-otp or GRIDWORLD_RESET_OTP."
            )

        print(
            f"Currently in world {current_world_id}; resetting team state to world -1 "
            "before entering the requested world."
        )
        response = self.client.reset_team(self.cfg.reset_otp)
        print(f"RESET RESPONSE: {response}")
        location = self._get_location_info()
        if location.world_id != -1:
            raise GridWorldAPIError(
                f"Reset completed, but location still reports world {location.world_id} instead of -1."
            )

    def _start_or_resume_world(self, world_id: int) -> Tuple[QLearningAgent, int]:
        agent = self._build_agent(world_id)
        location = self._get_location_info()

        if location.world_id == world_id and agent.grid.contains_state(location.state):
            print(f"Resuming world {world_id} from state {location.state}.")
            self.progress["active_world"] = world_id
            self._save_progress()
            return agent, int(location.state)

        if location.world_id == -1:
            self._respect_enter_rate_limit()
            start_state = agent.enter_world_once()
            self._mark_enter()
            print(f"Entered world {world_id} at state {start_state}.")
            self.progress["active_world"] = world_id
            self._save_progress()
            return agent, start_state

        self._respect_enter_rate_limit()
        start_state = agent.enter_world_once()
        self._mark_enter()
        print(f"Entered world {world_id} at state {start_state}.")
        self.progress["active_world"] = world_id
        self._save_progress()
        return agent, start_state

    def run(self) -> None:
        while True:
            target_world = self._next_incomplete_world()
            if target_world is None:
                print(
                    "Campaign complete: selected world range has reached the required goal hits."
                )
                break

            try:
                agent, start_state = self._start_or_resume_world(target_world)
                world_progress = self._world_progress(agent.cfg.world_id)
                goal_was_already_found = bool(world_progress["goal_found"])

                if goal_was_already_found:
                    print(f"World {agent.cfg.world_id} goal already found. Starting goal-optimization run.")

                    # Keep learning enabled. Full eval_mode would freeze Q-values,
                    # so the agent would not improve the path after the first goal.
                    agent.cfg.eval_mode = False
                    agent.cfg.goal_optimization_mode = True

                    # Almost greedy: mostly follows the best known path,
                    # but keeps tiny exploration to discover shorter alternatives.
                    agent.meta["epsilon"] = 0.01

                    # Turn off exploration/novelty bonuses during optimization.
                    # Important: update both cfg and learner, because QLearner stores
                    # these values internally after construction.
                    agent.cfg.exploration_bonus = 0.0
                    agent.cfg.state_novelty_bonus = 0.0
                    agent.cfg.unvisited_state_bonus = 0.0
                    agent.cfg.frontier_bonus = 0.0
                    agent.cfg.frontier_search_prob = 0.0
                    agent.learner.exploration_bonus = 0.0
                    agent.learner.unvisited_state_bonus = 0.0
                    agent.learner.frontier_bonus = 0.0

                    # More aggressive optimization after the goal is known.
                    agent.learner.alpha = 0.4
                    agent.learner.gamma = 0.98
                    agent.cfg.goal_q_bias_multiplier = max(agent.cfg.goal_q_bias_multiplier, 2.0)
                    agent.cfg.goal_reward_gradient_weight = max(agent.cfg.goal_reward_gradient_weight, 1.0)
                    agent.cfg.goal_backtrack_penalty = max(agent.cfg.goal_backtrack_penalty, 0.3)

                episode_index = int(world_progress["terminals_found"]) + 1

                # If we already found the goal before, use the known safe graph to compute
                # the shortest currently known path to the saved goal entry. If the graph is
                # incomplete because older runs did not save safe_edges, fall back to smart Q-learning
                # and keep recording safe_edges until BFS becomes possible.
                summary = None
                if goal_was_already_found:
                    planned_path = agent.find_shortest_known_goal_path(start_state)
                    if planned_path:
                        print(
                            f"Using BFS shortest known goal path for world {agent.cfg.world_id}: "
                            f"{len(planned_path)} action(s)."
                        )
                        summary = agent.follow_planned_goal_path(episode_index, start_state, planned_path)
                    elif agent.meta.get("goal_entry"):
                        print(
                            "No complete safe-edge path to goal yet. Using coordinate-biased exploit "
                            "because this world has no walls. This will also record safe_edges for future BFS."
                        )
                        summary = agent.coordinate_exploit_to_goal(episode_index, start_state)
                    else:
                        print(
                            "No goal entry and no complete safe-edge path yet. Falling back to goal-biased Q-learning "
                            "and recording safe_edges for future BFS."
                        )

                if summary is None:
                    summary = agent.run_episode(episode_index, start_state)
                agent.save()

                self.progress["active_world"] = agent.cfg.world_id
                world_progress["last_known_state"] = summary.get("final_state")
                self.progress["last_summary"] = {
                    "world_id": agent.cfg.world_id,
                    "terminal_attempt": episode_index,
                    **summary,
                }

                if summary["ended"]:
                    world_progress["terminals_found"] = int(world_progress["terminals_found"]) + 1
                    world_progress["last_terminal_kind"] = summary.get("terminal_kind")
                    world_progress["last_terminal_reward"] = summary.get("terminal_reward")
                    world_progress["last_known_state"] = summary.get("final_state")

                    if summary.get("terminal_kind") == "hazard":
                        world_progress["hazards_found"] = int(world_progress["hazards_found"]) + 1
                        print(
                            f"World {agent.cfg.world_id} hit a hazard terminal "
                            f"(reward={summary.get('terminal_reward')}). Continuing search."
                        )
                    elif summary.get("terminal_kind") == "goal":
                        if not goal_was_already_found:
                            world_progress["goal_found"] = True
                            world_progress["goal_reward"] = summary.get("terminal_reward")
                            print(
                                f"World {agent.cfg.world_id} discovered a goal terminal "
                                f"(reward={summary.get('terminal_reward')}). Starting 5 exploit goal runs."
                            )
                        else:
                            world_progress["goal_hits_completed"] = (
                                int(world_progress["goal_hits_completed"]) + 1
                            )
                            completed = int(world_progress["goal_hits_completed"])
                            print(
                                f"World {agent.cfg.world_id} reached the goal again "
                                f"(reward={summary.get('terminal_reward')}). "
                                f"Goal hits: {completed}/{self.cfg.traversals_per_world}."
                            )
                else:
                    print(
                        f"World {agent.cfg.world_id} episode completed at the step limit "
                        f"after {summary['steps']} step(s)."
                    )

                if summary["ended"]:
                    self.progress["active_world"] = None

                self._save_progress()
            except KeyboardInterrupt:
                print("Interrupted by user.")
                self._save_progress()
                break
            except Exception as exc:
                print(f"Campaign failed: {exc}")
                self._save_progress()
                raise


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
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.95)
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
    parser.add_argument("--null-state-terminal-reward", type=float, default=1000.0)
    parser.add_argument("--exploration-bonus", type=float, default=0.2)
    parser.add_argument("--state-novelty-bonus", type=float, default=0.0)
    parser.add_argument("--unvisited-state-bonus", type=float, default=1.0)
    parser.add_argument("--frontier-bonus", type=float, default=0.2)
    parser.add_argument("--frontier-search-prob", type=float, default=0.85)
    parser.add_argument("--loop-penalty-weight", type=float, default=0.1)
    parser.add_argument("--hazard-penalty-multiplier", type=float, default=3.0)
    parser.add_argument("--goal-short-path-bonus", type=float, default=300.0)
    parser.add_argument("--goal-q-bias-multiplier", type=float, default=2.0)
    parser.add_argument("--goal-reward-gradient-weight", type=float, default=1.0)
    parser.add_argument("--goal-backtrack-penalty", type=float, default=0.3)
    parser.add_argument(
        "--start-world",
        type=int,
        default=None,
        help="Start the campaign from this world number. Equivalent to setting campaign-world-start.",
    )
    parser.add_argument("--campaign-world-start", type=int, default=2)
    parser.add_argument("--campaign-world-end", type=int, default=2)
    parser.add_argument("--traversals-per-world", type=int, default=3)
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Disable exploration and learning updates; run the current policy greedily for evaluation.",
    )
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

    campaign_world_start = args.start_world if args.start_world is not None else args.campaign_world_start
    if campaign_world_start > args.campaign_world_end:
        parser.error("start-world/campaign-world-start cannot be greater than campaign-world-end.")

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
        null_state_terminal_reward_abs=args.null_state_terminal_reward,
        exploration_bonus=args.exploration_bonus,
        state_novelty_bonus=args.state_novelty_bonus,
        unvisited_state_bonus=args.unvisited_state_bonus,
        frontier_bonus=args.frontier_bonus,
        frontier_search_prob=args.frontier_search_prob,
        loop_penalty_weight=args.loop_penalty_weight,
        hazard_penalty_multiplier=args.hazard_penalty_multiplier,
        goal_short_path_bonus=args.goal_short_path_bonus,
        goal_q_bias_multiplier=args.goal_q_bias_multiplier,
        goal_reward_gradient_weight=args.goal_reward_gradient_weight,
        goal_backtrack_penalty=args.goal_backtrack_penalty,
        eval_mode=args.eval_mode,
        campaign_world_start=campaign_world_start,
        campaign_world_end=args.campaign_world_end,
        traversals_per_world=args.traversals_per_world,
    )


def main() -> None:
    cfg = parse_args()
    runner = CampaignRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
