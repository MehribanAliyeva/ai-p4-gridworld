import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import GridSpec, GridWorldResponseParser, MoveOutcome, QConfig, QLearner, QLearningAgent


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = GridWorldResponseParser(GridSpec(rows=5, cols=5))

    def test_parse_location_state_from_coordinates(self) -> None:
        state = self.parser.parse_location_state({"data": {"location": "2:3"}})
        self.assertEqual(state, 13)

    def test_parse_move_result_marks_blocked_move_as_self_loop(self) -> None:
        outcome = self.parser.parse_move_result(
            {"reward": -1, "message": "Blocked by wall"},
            current_state=7,
        )
        self.assertEqual(
            outcome,
            MoveOutcome(
                reward=-1.0,
                next_state=7,
                done=False,
                blocked=True,
                invalid_move=False,
                assumed_terminal=False,
                raw={"reward": -1, "message": "Blocked by wall"},
            ),
        )

    def test_parse_move_result_detects_terminal_message(self) -> None:
        outcome = self.parser.parse_move_result(
            {"reward": 50, "message": "Reached terminal exit", "newState": -1},
            current_state=4,
        )
        self.assertTrue(outcome.done)
        self.assertEqual(outcome.next_state, -1)
        self.assertFalse(outcome.assumed_terminal)

    def test_large_negative_reward_only_assumes_terminal_when_next_state_is_null(self) -> None:
        terminal_outcome = self.parser.parse_move_result(
            {"reward": -1000, "newState": None, "worldId": 0},
            current_state=4,
        )
        non_terminal_outcome = self.parser.parse_move_result(
            {"reward": -1000, "newState": 3, "worldId": 0},
            current_state=4,
        )
        self.assertTrue(terminal_outcome.done)
        self.assertTrue(terminal_outcome.assumed_terminal)
        self.assertFalse(non_terminal_outcome.assumed_terminal)

    def test_large_positive_reward_only_assumes_terminal_when_next_state_is_null(self) -> None:
        terminal_outcome = self.parser.parse_move_result(
            {"reward": 1000, "newState": None, "worldId": 0},
            current_state=4,
        )
        non_terminal_outcome = self.parser.parse_move_result(
            {"reward": 1000, "newState": 3, "worldId": 0},
            current_state=4,
        )
        self.assertTrue(terminal_outcome.done)
        self.assertTrue(terminal_outcome.assumed_terminal)
        self.assertFalse(non_terminal_outcome.assumed_terminal)


class LearnerTests(unittest.TestCase):
    def test_choose_action_breaks_ties_randomly_among_best_actions(self) -> None:
        learner = QLearner(alpha=0.1, gamma=0.9, num_states=1, num_actions=4)
        q_table = np.array([[1.0, 3.0, 3.0, 0.5]], dtype=np.float32)
        random.seed(0)

        chosen = {learner.choose_action(q_table, 0, epsilon=0.0) for _ in range(50)}
        self.assertEqual(chosen, {1, 2})

    def test_exploration_bonus_prefers_less_visited_action(self) -> None:
        learner = QLearner(alpha=0.1, gamma=0.9, num_states=1, num_actions=4, exploration_bonus=2.0)
        q_table = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        visits = np.array([[20.0, 0.0, 20.0, 20.0]], dtype=np.float32)
        random.seed(0)

        chosen = learner.choose_action(q_table, 0, epsilon=0.0, action_visits=visits)
        self.assertEqual(chosen, 1)


class AgentTests(unittest.TestCase):
    def test_agent_loads_grid_shape_from_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            q_path = tmp_path / "world_0_q.npy"
            meta_path = tmp_path / "world_0_meta.json"
            np.save(q_path, np.zeros((12, 4), dtype=np.float32))
            meta_path.write_text(
                '{"grid": {"rows": 3, "cols": 4}, "history": [], "metrics": {}, "totals": {}}',
                encoding="utf-8",
            )

            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir)
            agent = QLearningAgent(cfg)

            self.assertEqual(agent.grid.rows, 3)
            self.assertEqual(agent.grid.cols, 4)
            self.assertEqual(agent.q.shape, (12, 4))

    def test_missing_next_state_can_fall_back_to_self_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir, max_episodes=1)
            agent = QLearningAgent(cfg)
            agent._safe_get_location = lambda: None

            next_state = None
            state = 5
            outcome = MoveOutcome(
                reward=-1.0,
                next_state=next_state,
                done=False,
                blocked=False,
                invalid_move=False,
                assumed_terminal=False,
                raw={"reward": -1},
            )

            if next_state is None and not outcome.done:
                next_state = agent._safe_get_location()
                if next_state is None and agent.cfg.infer_self_loop_on_missing_state:
                    next_state = state
                    agent._bump_total("inferred_self_loops")
                    agent._record_warning("test warning")

            self.assertEqual(next_state, state)
            self.assertEqual(agent.meta["totals"]["inferred_self_loops"], 1)
            self.assertEqual(len(agent.meta["warnings"]), 1)

    def test_location_retry_succeeds_after_transient_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(
                team_id=1,
                api_key="k",
                user_id="u",
                storage_dir=tmp_dir,
                location_retry_attempts=3,
                location_retry_delay_sec=0.0,
            )
            agent = QLearningAgent(cfg)
            responses = iter([{"state": None}, {"state": "bad"}, {"state": 7}])
            agent.client.get_location = lambda: next(responses)

            self.assertEqual(agent._safe_get_location(), 7)

    def test_enter_delay_default_matches_assignment_limit(self) -> None:
        cfg = QConfig(team_id=1, api_key="k", user_id="u")
        self.assertEqual(cfg.enter_delay_sec, 600.0)


if __name__ == "__main__":
    unittest.main()
