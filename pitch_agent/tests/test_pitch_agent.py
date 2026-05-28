import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pitch_agent.state.pitch_state import PitchState
from pitch_agent.action.pitch_action import PitchAction, FEEDBACK_MESSAGE, get_available_actions
from pitch_agent.analyzer.pitch_analyzer import PitchAnalyzer
from pitch_agent.reward.reward_calculator import RewardCalculator
from pitch_agent.agent.q_table import QTable
from pitch_agent.agent.pitch_agent import PitchAgent
from pitch_agent.config import FAIL_THRESHOLD


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

    def test_no_drift_state(self):
        """DRIFT 제거 확인"""
        self.assertFalse(hasattr(PitchState, 'DRIFT'))


class TestGetAvailableActions(unittest.TestCase):

    def test_good_returns_positive(self):
        self.assertIn(PitchAction.POSITIVE_PITCH, get_available_actions(PitchState.GOOD, 0))

    def test_sharp_returns_pitch_down(self):
        actions = get_available_actions(PitchState.SHARP_MAJOR, 0)
        self.assertIn(PitchAction.PITCH_DOWN, actions)
        self.assertNotIn(PitchAction.SWITCH_PITCH_TO_POSTURE, actions)

    def test_flat_returns_pitch_up(self):
        self.assertIn(PitchAction.PITCH_UP, get_available_actions(PitchState.FLAT_MAJOR, 0))

    def test_tripled_out_adds_switch(self):
        actions = get_available_actions(PitchState.SHARP_MAJOR, FAIL_THRESHOLD)
        self.assertIn(PitchAction.SWITCH_PITCH_TO_POSTURE, actions)
        self.assertIn(PitchAction.PITCH_DOWN, actions)

    def test_no_pitch_fix_drift_action(self):
        """PITCH_FIX_DRIFT 제거 확인"""
        self.assertFalse(hasattr(PitchAction, 'PITCH_FIX_DRIFT'))


class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = RewardCalculator()

    def test_good_transition(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.GOOD, 0), 1.0)

    def test_partial_improvement(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_SLIGHT, 0), 0.5)

    def test_no_change(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_MAJOR, 0), -0.3)

    def test_repeat_fail(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_MAJOR, 2), -0.6)

    def test_full_fail(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_MAJOR, PitchState.SHARP_MAJOR, 3), -1.0)

    def test_worse(self):
        self.assertEqual(self.calc.calculate(PitchState.SHARP_SLIGHT, PitchState.SHARP_MAJOR, 0), -0.8)


class TestQTable(unittest.TestCase):

    def setUp(self):
        self.q = QTable(user_id="test_qtable", session_count=0)

    def tearDown(self):
        path = "data/q_tables/user_test_qtable_q_table.json"
        if os.path.exists(path): os.remove(path)

    def test_initial_q_value(self):
        self.assertEqual(self.q.get(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN), 0.0)

    def test_update_increases_q(self):
        self.q.update(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN, 1.0, PitchState.GOOD, 0)
        self.assertGreater(self.q.get_personal(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN), 0.0)

    def test_update_decreases_q(self):
        self.q.update(PitchState.FLAT_MAJOR, 0, PitchAction.PITCH_UP, 1.0, PitchState.GOOD, 0)
        before = self.q.get_personal(PitchState.FLAT_MAJOR, 0, PitchAction.PITCH_UP)
        self.q.update(PitchState.FLAT_MAJOR, 0, PitchAction.PITCH_UP, -1.0, PitchState.FLAT_MAJOR, 1)
        self.assertLess(self.q.get_personal(PitchState.FLAT_MAJOR, 0, PitchAction.PITCH_UP), before)

    def test_fail_count_different_keys(self):
        self.q.update(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN, 1.0,  PitchState.GOOD, 0)
        self.q.update(PitchState.SHARP_MAJOR, 1, PitchAction.PITCH_DOWN, -0.3, PitchState.SHARP_MAJOR, 2)
        self.assertNotEqual(
            self.q.get_personal(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN),
            self.q.get_personal(PitchState.SHARP_MAJOR, 1, PitchAction.PITCH_DOWN)
        )

    def test_best_action_tripled_out(self):
        self.q.update(PitchState.SHARP_MAJOR, FAIL_THRESHOLD,
                      PitchAction.SWITCH_PITCH_TO_POSTURE, 0.8, PitchState.GOOD, 0)
        action = self.q.best_action(PitchState.SHARP_MAJOR, FAIL_THRESHOLD, epsilon=0.0)
        self.assertEqual(action, PitchAction.SWITCH_PITCH_TO_POSTURE)

    def test_save_and_load(self):
        self.q.update(PitchState.SHARP_SLIGHT, 0, PitchAction.PITCH_DOWN, 1.0, PitchState.GOOD, 0)
        self.q.save()
        loaded = QTable(user_id="test_qtable", session_count=0)
        self.assertGreater(loaded.get_personal(PitchState.SHARP_SLIGHT, 0, PitchAction.PITCH_DOWN), 0.0)


class TestPitchAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PitchAgent(user_id="test_agent", session_count=0)

    def tearDown(self):
        path = "data/q_tables/user_test_agent_q_table.json"
        if os.path.exists(path): os.remove(path)

    def test_good_returns_positive_pitch(self):
        result = self.agent.run(0.0)
        self.assertEqual(result["state"],  PitchState.GOOD.value)
        self.assertEqual(result["action"], PitchAction.POSITIVE_PITCH.value)

    def test_sharp_major_returns_pitch_down(self):
        self.agent.reset_analyzer()
        self.assertEqual(self.agent.run(110.0)["action"], PitchAction.PITCH_DOWN.value)

    def test_flat_major_returns_pitch_up(self):
        self.agent.reset_analyzer()
        self.assertEqual(self.agent.run(-110.0)["action"], PitchAction.PITCH_UP.value)

    def test_fail_count_bad_state_continuous(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        result = self.agent.run(110.0)
        self.assertEqual(result["fail_count"], 1)

    def test_fail_count_resets_on_good(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.agent.run(110.0)
        self.assertEqual(self.agent.run(0.0)["fail_count"], 0)

    def test_tripled_out_flag(self):
        self.agent.reset_analyzer()
        for _ in range(FAIL_THRESHOLD + 1):
            result = self.agent.run(110.0)
        self.assertTrue(result["tripled_out"])

    def test_reward_good_transition(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.assertEqual(self.agent.receive_next_state(0.0), 1.0)

    def test_reward_no_change(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.assertEqual(self.agent.receive_next_state(110.0), -0.3)

    def test_q_learns_from_success(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.agent.receive_next_state(0.0)
        self.assertGreater(
            self.agent.q_table.get_personal(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN), 0.0
        )

    def test_q_learns_from_failure(self):
        self.agent.reset_analyzer()
        self.agent.run(110.0)
        self.agent.receive_next_state(110.0)
        self.assertLess(
            self.agent.q_table.get_personal(PitchState.SHARP_MAJOR, 0, PitchAction.PITCH_DOWN), 0.0
        )

    def test_full_session_saves(self):
        self.agent.reset_analyzer()
        session = [110.0, 50.0, 0.0, -110.0, 0.0]
        for i, dev in enumerate(session):
            self.agent.run(dev)
            if i + 1 < len(session):
                self.agent.receive_next_state(session[i + 1])
        self.agent.save()
        self.assertTrue(os.path.exists("data/q_tables/user_test_agent_q_table.json"))

    def test_switch_selected_when_tripled_out(self):
        self.agent.reset_analyzer()
        self.agent.q_table.update(
            PitchState.FLAT_MAJOR, FAIL_THRESHOLD,
            PitchAction.SWITCH_PITCH_TO_POSTURE, 0.8, PitchState.GOOD, 0
        )
        for _ in range(FAIL_THRESHOLD + 1):
            result = self.agent.run(-110.0)
        self.assertIn(result["action"], [
            PitchAction.PITCH_UP.value,
            PitchAction.SWITCH_PITCH_TO_POSTURE.value
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
