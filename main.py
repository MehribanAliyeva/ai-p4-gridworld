import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

GW_URL = "https://www.notexponential.com/aip2pgaming/api/rl/gw.php"
SCORE_URL = "https://www.notexponential.com/aip2pgaming/api/rl/score.php"
DEFAULT_ACTIONS = ["N", "S", "E", "W"]
NUM_STATES = 1600


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
    enter_delay_sec: float = 30
    autosave_every: int = 1
    storage_dir: str = "q_data"
    actions: Tuple[str, ...] = tuple(DEFAULT_ACTIONS)
    verbose: bool = True


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
                except json.JSONDecodeError as e:
                    print("RAW CURL STDOUT:")
                    print(stdout)
                    print("RAW CURL STDERR:")
                    print(stderr)
                    raise GridWorldAPIError(f"Invalid JSON response: {stdout}") from e
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
        data = run_curl(cmd)
        return self._check_response(data)

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
        data = run_curl(cmd)
        return self._check_response(data)

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
        data = run_curl(cmd)
        return self._check_response(data)

    def get_runs(self, count: int = 10) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{SCORE_URL}?type=runs&teamId={self.team_id}&count={count}",
            *self._headers(),
        ]
        data = run_curl(cmd)
        return self._check_response(data)

    def get_score(self) -> Dict:
        cmd = [
            "curl",
            "-s",
            "-X",
            "GET",
            f"{SCORE_URL}?type=score&teamId={self.team_id}",
            *self._headers(),
        ]
        data = run_curl(cmd)
        return self._check_response(data)


class QLearningAgent:
    def __init__(self, config: QConfig):
        self.cfg = config
        self.actions = list(config.actions)
        self.num_actions = len(self.actions)
        self.storage = Path(config.storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.q_path = self.storage / f"world_{config.world_id}_q.npy"
        self.meta_path = self.storage / f"world_{config.world_id}_meta.json"
        self.q = self._load_q_table()
        self.meta = self._load_meta()
        self.client = GridWorldClient(config.team_id, config.api_key, config.user_id)

    def _load_q_table(self) -> np.ndarray:
        if self.q_path.exists():
            q = np.load(self.q_path)
            expected_shape = (NUM_STATES, self.num_actions)
            if q.shape != expected_shape:
                raise ValueError(
                    f"Saved Q table shape {q.shape} does not match expected {expected_shape}"
                )
            return q
        return np.zeros((NUM_STATES, self.num_actions), dtype=np.float32)

    def _load_meta(self) -> Dict:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "episodes_completed": 0,
            "epsilon": self.cfg.epsilon,
            "best_episode_reward": None,
            "last_enter_ts": 0.0,
            "last_move_ts": 0.0,
            "history": [],
        }

    def save(self) -> None:
        np.save(self.q_path, self.q)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def _respect_rate_limit(self, key: str, min_delay: float) -> None:
        last_ts = float(self.meta.get(key, 0.0))
        elapsed = time.time() - last_ts
        wait = max(0.0, min_delay - elapsed)
        if wait > 0:
            print(f"Waiting {wait:.1f}s to respect API rate limit...")
            time.sleep(wait)

    def _mark_call(self, key: str) -> None:
        self.meta[key] = time.time()

    @staticmethod
    def _extract_first_int(obj: Dict, keys: List[str]) -> Optional[int]:
        for key in keys:
            if key in obj:
                try:
                    return int(obj[key])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _extract_first_float(obj: Dict, keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
        return None

    def _parse_location_state(self, data: Dict) -> Optional[int]:
        possible_keys = [
            "state",
            "location",
            "currentState",
            "newState",
            "stateId",
        ]

        def parse_state_value(value) -> Optional[int]:
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
                            if 0 <= row < 40 and 0 <= col < 40:
                                return row * 40 + col
                        except ValueError:
                            pass

            return None

        def try_extract(obj: Dict) -> Optional[int]:
            if not isinstance(obj, dict):
                return None

            for key in possible_keys:
                if key in obj:
                    parsed = parse_state_value(obj[key])
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

    def _parse_move_result(self, data: Dict) -> Tuple[float, Optional[int], bool, Dict]:
        def parse_state_value(value) -> Optional[int]:
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
                            if 0 <= row < 40 and 0 <= col < 40:
                                return row * 40 + col
                        except ValueError:
                            pass

            return None

        reward = self._extract_first_float(data, ["reward", "score", "points"])
        world_id = self._extract_first_int(data, ["worldId", "world"])
        next_state = None
        done = False

        for key in ["newState", "state", "location", "currentState"]:
            if key in data:
                next_state = parse_state_value(data[key])
                if next_state is not None:
                    break

        for parent in ("data", "result", "details"):
            nested = data.get(parent)
            if isinstance(nested, dict):
                if reward is None:
                    reward = self._extract_first_float(nested, ["reward", "score", "points"])
                if world_id is None:
                    world_id = self._extract_first_int(nested, ["worldId", "world"])
                if next_state is None:
                    for key in ["newState", "state", "location", "currentState"]:
                        if key in nested:
                            next_state = parse_state_value(nested[key])
                            if next_state is not None:
                                break

        if reward is None:
            raise GridWorldAPIError(f"Could not parse reward from move response: {data}")

        if world_id == -1 or next_state == -1:
            done = True

        msg = str(data.get("message", "")).lower()
        if "exit" in msg or "terminal" in msg or "finished" in msg:
            done = True

        return reward, next_state, done, data

    def choose_action(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        return int(np.argmax(self.q[state]))

    def update_q(
        self,
        state: int,
        action_idx: int,
        reward: float,
        next_state: Optional[int],
        done: bool,
    ) -> None:
        current_q = self.q[state, action_idx]
        target = reward
        if not done and next_state is not None and 0 <= next_state < NUM_STATES:
            target += self.cfg.gamma * np.max(self.q[next_state])
        self.q[state, action_idx] = current_q + self.cfg.alpha * (target - current_q)

    def _safe_get_location(self) -> Optional[int]:
        try:
            loc = self.client.get_location()
            return self._parse_location_state(loc)
        except Exception as e:
            print(f"Warning: could not read current location: {e}")
            return None

    def start_episode(self) -> int:
        self._respect_rate_limit("last_enter_ts", self.cfg.enter_delay_sec)
        try:
            enter_response = self.client.enter_world(self.cfg.world_id)
            self._mark_call("last_enter_ts")

            print("ENTER RESPONSE:")
            print(json.dumps(enter_response, indent=2))

            state = self._parse_location_state(enter_response)
            if state is None:
                loc = self.client.get_location()
                print("LOCATION RESPONSE:")
                print(json.dumps(loc, indent=2))
                state = self._parse_location_state(loc)

            if state is None:
                raise GridWorldAPIError(
                    "Could not determine initial state after entering the world. "
                    "See ENTER RESPONSE / LOCATION RESPONSE above."
                )

            return state

        except GridWorldAPIError as e:
            if "already in a world" in str(e).lower() or "inconsistent state" in str(e).lower():
                loc = self.client.get_location()
                print("RECOVERY LOCATION RESPONSE:")
                print(json.dumps(loc, indent=2))
                state = self._parse_location_state(loc)
                if state is not None and 0 <= state < NUM_STATES:
                    return state
            raise

    def run_episode(self, episode_index: int) -> Dict:
        epsilon = float(self.meta.get("epsilon", self.cfg.epsilon))
        state = self.start_episode()
        total_reward = 0.0
        done = False
        visited_states = [state]

        for step in range(1, self.cfg.max_steps_per_episode + 1):
            if not (0 <= state < NUM_STATES):
                raise GridWorldAPIError(f"Invalid state index: {state}")

            action_idx = self.choose_action(state, epsilon)
            action = self.actions[action_idx]

            self._respect_rate_limit("last_move_ts", self.cfg.move_delay_sec)
            move_data = self.client.move(self.cfg.world_id, action)
            self._mark_call("last_move_ts")

            reward, next_state, move_done, raw = self._parse_move_result(move_data)
            self.update_q(state, action_idx, reward, next_state, move_done)

            total_reward += reward
            done = move_done
            if next_state is not None:
                visited_states.append(next_state)

            if self.cfg.verbose:
                print(
                    f"Episode {episode_index} | Step {step:03d} | State {state:4d} | "
                    f"Action {action:>5s} | Reward {reward:8.3f} | Next {str(next_state):>4s} | Done={done}"
                )

            if done:
                break

            if next_state is None:
                next_state = self._safe_get_location()
                if next_state is None:
                    raise GridWorldAPIError(
                        "Move response did not include next state, and fallback location lookup also failed."
                    )

            state = next_state

        epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)
        self.meta["epsilon"] = epsilon
        self.meta["episodes_completed"] = int(self.meta.get("episodes_completed", 0)) + 1

        best = self.meta.get("best_episode_reward")
        if best is None or total_reward > best:
            self.meta["best_episode_reward"] = total_reward

        episode_summary = {
            "episode": episode_index,
            "total_reward": total_reward,
            "steps": step,
            "ended": done,
            "epsilon_after": epsilon,
            "unique_states": len(set(visited_states)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta.setdefault("history", []).append(episode_summary)
        return episode_summary

    def train(self) -> None:
        for ep in range(1, self.cfg.max_episodes + 1):
            try:
                summary = self.run_episode(ep)
                print("-" * 90)
                print("Episode summary:", json.dumps(summary, indent=2))
                print("-" * 90)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:
                print(f"Episode {ep} failed: {e}")
                self.save()
                raise

            if ep % self.cfg.autosave_every == 0:
                self.save()

        self.save()
        print(f"Saved Q-table to {self.q_path}")
        print(f"Saved metadata to {self.meta_path}")


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
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--move-delay", type=float, default=15.5)
    parser.add_argument("--enter-delay", type=float, default=601.0)
    parser.add_argument("--storage-dir", type=str, default="q_data")
    parser.add_argument(
        "--actions",
        nargs=4,
        default=DEFAULT_ACTIONS,
        help="Exactly 4 action strings, e.g. N S E W or UP DOWN LEFT RIGHT",
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
    )


def main() -> None:
    cfg = parse_args()
    agent = QLearningAgent(cfg)
    agent.train()


if __name__ == "__main__":
    main()