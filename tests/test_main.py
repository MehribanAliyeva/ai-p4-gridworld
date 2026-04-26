import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import CampaignRunner, CampaignStore, GridSpec, GridWorldResponseParser, MoveOutcome, QConfig, QLearner, QLearningAgent


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = GridWorldResponseParser(GridSpec(rows=5, cols=5))

    def test_parse_location_state_from_coordinates(self) -> None:
        state = self.parser.parse_location_state({"data": {"location": "2:3"}})
        self.assertEqual(state, 13)

    def test_parse_location_info_reads_world_and_state(self) -> None:
        info = self.parser.parse_location_info({"world": 3, "state": "2:3"})
        self.assertEqual(info.world_id, 3)
        self.assertEqual(info.state, 13)

    def test_parse_move_result_does_not_infer_self_loop_from_message_text(self) -> None:
        outcome = self.parser.parse_move_result(
            {"reward": -1, "message": "Blocked by wall"},
            current_state=7,
        )
        self.assertEqual(
            outcome,
            MoveOutcome(
                reward=-1.0,
                next_state=None,
                done=False,
                blocked=False,
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

    def test_parse_move_result_does_not_infer_blocked_or_invalid_from_missing_message(self) -> None:
        outcome = self.parser.parse_move_result(
            {"reward": -1, "newState": None, "worldId": 0},
            current_state=4,
        )
        self.assertFalse(outcome.blocked)
        self.assertFalse(outcome.invalid_move)
        self.assertFalse(outcome.done)

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

    def test_unvisited_state_bonus_prefers_unseen_neighbor(self) -> None:
        learner = QLearner(
            alpha=0.1,
            gamma=0.9,
            num_states=1600,
            num_actions=4,
            exploration_bonus=0.0,
            rows=40,
            cols=40,
            unvisited_state_bonus=5.0,
            frontier_bonus=0.0,
        )
        q_table = np.zeros((1600, 4), dtype=np.float32)
        action_visits = np.zeros((1600, 4), dtype=np.float32)
        state_visits = np.ones((1600,), dtype=np.float32)
        state_visits[1] = 0.0
        state_visits[40] = 10.0
        random.seed(0)

        chosen = learner.choose_action(q_table, 0, epsilon=0.0, action_visits=action_visits, state_visits=state_visits)
        self.assertEqual(chosen, 2)

    def test_eval_mode_equivalent_policy_is_greedy_when_epsilon_zero(self) -> None:
        learner = QLearner(alpha=0.1, gamma=0.9, num_states=1, num_actions=4, exploration_bonus=0.0)
        q_table = np.array([[1.0, 5.0, 2.0, 1.0]], dtype=np.float32)
        random.seed(0)

        chosen = learner.choose_action(q_table, 0, epsilon=0.0)
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

    def test_state_novelty_reward_is_zero_in_eval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir, eval_mode=True)
            agent = QLearningAgent(cfg)
            agent.state_visits[7] = 0

            self.assertEqual(agent._state_novelty_reward(7, done=False), 0.0)

    def test_state_novelty_reward_decreases_with_visits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(
                team_id=1,
                api_key="k",
                user_id="u",
                storage_dir=tmp_dir,
                state_novelty_bonus=1.0,
            )
            agent = QLearningAgent(cfg)
            agent.state_visits[7] = 0
            first = agent._state_novelty_reward(7, done=False)
            agent.state_visits[7] = 8
            later = agent._state_novelty_reward(7, done=False)

            self.assertGreater(first, later)

    def test_run_episode_labels_terminal_as_hazard_or_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(
                team_id=1,
                api_key="k",
                user_id="u",
                storage_dir=tmp_dir,
                max_steps_per_episode=1,
                epsilon=0.0,
                epsilon_min=0.0,
                exploration_bonus=0.0,
                state_novelty_bonus=0.0,
                verbose=False,
                move_delay_sec=0.0,
            )
            agent = QLearningAgent(cfg)
            agent.client.move = lambda world_id, move: {"reward": -1000, "newState": None, "worldId": world_id}

            summary = agent.run_episode(1, 0)

            self.assertTrue(summary["ended"])
            self.assertEqual(summary["terminal_kind"], "hazard")
            self.assertEqual(summary["terminal_reward"], -1000.0)
            self.assertIsNone(summary["final_state"])

    def test_run_episode_ends_at_step_limit_without_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(
                team_id=1,
                api_key="k",
                user_id="u",
                storage_dir=tmp_dir,
                max_steps_per_episode=1,
                epsilon=0.0,
                epsilon_min=0.0,
                exploration_bonus=0.0,
                state_novelty_bonus=0.0,
                verbose=False,
                move_delay_sec=0.0,
            )
            agent = QLearningAgent(cfg)
            agent.client.move = lambda world_id, move: {"reward": -0.1, "newState": 1, "worldId": world_id}

            summary = agent.run_episode(1, 0)

            self.assertTrue(summary["ended"])
            self.assertTrue(summary["ended_by_step_limit"])
            self.assertIsNone(summary["terminal_kind"])
            self.assertEqual(summary["final_state"], 1)

    def test_campaign_store_initializes_world_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir)
            store = CampaignStore(tmp_dir)
            progress = store.load(cfg)

            self.assertFalse(progress["worlds"]["1"]["goal_found"])
            self.assertEqual(progress["worlds"]["1"]["goal_hits_completed"], 0)
            self.assertEqual(progress["worlds"]["10"]["hazards_found"], 0)

    def test_campaign_runner_waits_for_goal_before_world_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir, traversals_per_world=5)
            runner = CampaignRunner(cfg)
            world = runner._world_progress(1)

            world["goal_hits_completed"] = 5
            self.assertFalse(runner._world_is_complete(1))

            world["goal_found"] = True
            self.assertTrue(runner._world_is_complete(1))

    def test_campaign_store_migrates_old_traversal_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_path = Path(tmp_dir) / "campaign_progress.json"
            progress_path.write_text(
                '{"worlds": {"1": {"traversals_completed": 3}}}',
                encoding="utf-8",
            )

            cfg = QConfig(team_id=1, api_key="k", user_id="u", storage_dir=tmp_dir)
            store = CampaignStore(tmp_dir)
            progress = store.load(cfg)

            self.assertEqual(progress["worlds"]["1"]["goal_hits_completed"], 3)
            self.assertNotIn("traversals_completed", progress["worlds"]["1"])


if __name__ == "__main__":
    unittest.main()
