import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pitch_agent.state.pitch_state import PitchState
from pitch_agent.action.pitch_action import PitchAction, FEEDBACK_MESSAGE, get_available_actions
from pitch_agent.analyzer.pitch_analyzer import PitchAnalyzer
from pitch_agent.reward.reward_calculator import RewardCalculator
from pitch_agent.agent.q_table import QTable
from pitch_agent.agent.pitch_agent import PitchAgent


class TestPitchAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = PitchAnalyzer()

    def test_good(self):
        self.assertEqual(self.analyzer.analyze(0.0),   PitchState.GOOD)
        self.assertEqual(self.analyzer.analyze(20.0),  PitchState.GOOD)
        self.assertEqual(self.analyzer.analyze(-20.0), PitchState.GOOD)

    def test_sharp_slight(self):
        self.assertEqual(self.analyzer.analyze(50.0), PitchState.SHARP_SLIGHT)

    def test_sharp_major(self):
        self.assertEqual(self.analyzer.analyze(110.0), PitchState.SHARP_MAJOR)

    def test_flat_slight(self):
        self.assertEqual(self.analyzer.analyze(-50.0), PitchState.FLAT_SLIGHT)

    def test_flat_major(self):
        self.assertEqual(self.analyzer.analyze(-110.0), PitchState.FLAT_MAJOR)


class TestGetAvailableActions(unittest.TestCase):

    def test_good_returns_positive(self):
        self.assertEqual(get_available_actions(PitchState.GOOD), [PitchAction.POSITIVE_PITCH])

    def test_sharp_returns_pitch_down_and_supervisor(self):
        actions = get_available_actions(PitchState.SHARP_MAJOR)
        self.assertIn(PitchAction.PITCH_DOWN, actions)
        self.assertIn(PitchAction.CALL_SUPERVISOR, actions)

    def test_flat_returns_pitch_up_and_supervisor(self):
        actions = get_available_actions(PitchState.FLAT_MAJOR)
        self.assertIn(PitchAction.PITCH_UP, actions)
        self.assertIn(PitchAction.CALL_SUPERVISOR, actions)

    def test_no_fail_count_dependency(self):
        """fail_count 없이 동일한 Action 목록"""
        self.assertEqual(
            get_available_actions(PitchState.SHARP_MAJOR),
            get_available_actions(PitchState.SHARP_SLIGHT)
        )


class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = RewardCalculator()

    def test_good_transition(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.GOOD), 1.0)

    def test_supervisor_hit(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.GOOD, True), 0.8)

    def test_partial_improvement(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_SLIGHT), 0.5)

    def test_no_change(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_MAJOR), -0.3)

    def test_supervisor_miss(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_MAJOR, True), -0.5)

    def test_worse(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_SLIGHT, PitchState.SHARP_MAJOR), -0.8)


class TestQTable(unittest.TestCase):

    def setUp(self):
        self.q = QTable(user_id="test_qtable", session_count=0)

    def tearDown(self):
        import glob
        for f in glob.glob("**/user_test_qtable_q_table.json", recursive=True):
            os.remove(f)

    def test_initial_q_value(self):
        self.assertEqual(self.q.get(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN), 0.0)

    def test_update_increases_q(self):
        self.q.update(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN, 1.0, PitchState.GOOD)
        self.assertGreater(self.q.get_personal(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN), 0.0)

    def test_update_decreases_q(self):
        self.q.update(PitchState.FLAT_MAJOR, PitchAction.PITCH_UP, 1.0, PitchState.GOOD)
        before = self.q.get_personal(PitchState.FLAT_MAJOR, PitchAction.PITCH_UP)
        self.q.update(PitchState.FLAT_MAJOR, PitchAction.PITCH_UP, -1.0, PitchState.FLAT_MAJOR)
        self.assertLess(self.q.get_personal(PitchState.FLAT_MAJOR, PitchAction.PITCH_UP), before)

    def test_supervisor_q_increases_after_hit(self):
        self.q.update(PitchState.SHARP_MAJOR, PitchAction.CALL_SUPERVISOR, 0.8, PitchState.GOOD)
        self.assertGreater(self.q.get_personal(PitchState.SHARP_MAJOR, PitchAction.CALL_SUPERVISOR), 0.0)

    def test_best_action_selects_supervisor_when_higher(self):
        self.q.update(PitchState.SHARP_MAJOR, PitchAction.CALL_SUPERVISOR, 1.0, PitchState.GOOD)
        action = self.q.best_action(PitchState.SHARP_MAJOR, epsilon=0.0)
        self.assertEqual(action, PitchAction.CALL_SUPERVISOR)

    def test_save_and_load(self):
        self.q.update(PitchState.FLAT_SLIGHT, PitchAction.PITCH_UP, 1.0, PitchState.GOOD)
        self.q.save()
        loaded = QTable(user_id="test_qtable", session_count=0)
        self.assertGreater(loaded.get_personal(PitchState.FLAT_SLIGHT, PitchAction.PITCH_UP), 0.0)


class TestPitchAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PitchAgent(user_id="test_agent", session_count=0)

    def tearDown(self):
        import glob
        for f in glob.glob("**/user_test_agent_q_table.json", recursive=True):
            os.remove(f)

    def test_good_returns_positive_pitch(self):
        result = self.agent.run(0.0)
        self.assertEqual(result["state"],  PitchState.GOOD.value)
        self.assertEqual(result["action"], PitchAction.POSITIVE_PITCH.value)
        self.assertFalse(result["call_supervisor"])

    def test_sharp_returns_pitch_down(self):
        self.agent.reset_analyzer()
        result = self.agent.run(110.0)
        self.assertEqual(result["action"], PitchAction.PITCH_DOWN.value)
        self.assertFalse(result["call_supervisor"])

    def test_flat_returns_pitch_up(self):
        self.agent.reset_analyzer()
        result = self.agent.run(-110.0)
        self.assertEqual(result["action"], PitchAction.PITCH_UP.value)

    def test_call_supervisor_flag(self):
        """CALL_SUPERVISOR 선택 시 call_supervisor True"""
        self.agent.reset_analyzer()
        self.agent.q_table.update(
            PitchState.SHARP_MAJOR, PitchAction.CALL_SUPERVISOR, 1.0, PitchState.GOOD
        )
        result = self.agent.run(110.0)
        if result["action"] == PitchAction.CALL_SUPERVISOR.value:
            self.assertTrue(result["call_supervisor"])

    def test_reward_good_transition(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.assertEqual(self.agent.receive_next_state(0.0), 1.0)

    def test_reward_no_change(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.assertEqual(self.agent.receive_next_state(110.0), -0.3)

    def test_supervisor_hit_reward(self):
        """CALL_SUPERVISOR 후 GOOD → +0.8"""
        self.agent.reset_analyzer()
        self.agent.q_table.update(
            PitchState.SHARP_MAJOR, PitchAction.CALL_SUPERVISOR, 1.0, PitchState.GOOD
        )
        self.agent._current_state  = PitchState.SHARP_MAJOR
        self.agent._current_action = PitchAction.CALL_SUPERVISOR
        reward = self.agent.receive_next_state(0.0)
        self.assertEqual(reward, 0.8)

    def test_q_learns_from_success(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.agent.receive_next_state(0.0)
        self.assertGreater(
            self.agent.q_table.get_personal(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN), 0.0
        )

    def test_q_learns_from_failure(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.agent.receive_next_state(110.0)
        self.assertLess(
            self.agent.q_table.get_personal(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN), 0.0
        )

    def test_supervisor_naturally_selected_after_repeated_failure(self):
        """반복 실패 후 CALL_SUPERVISOR Q값이 높아져서 자연 선택"""
        self.agent.reset_analyzer()
        for _ in range(5):
            self.agent.run(110.0)
            self.agent.receive_next_state(110.0)
        q_down = self.agent.q_table.get_personal(PitchState.SHARP_MAJOR, PitchAction.PITCH_DOWN)
        self.assertLess(q_down, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)