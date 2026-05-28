from ..analyzer.pitch_analyzer import PitchAnalyzer
from ..agent.q_table import QTable
from ..reward.reward_calculator import RewardCalculator
from ..state.pitch_state import PitchState
from ..action.pitch_action import PitchAction, FEEDBACK_MESSAGE
from ..config import FAIL_THRESHOLD


class PitchAgent:

    def __init__(self, user_id: str, session_count: int = 0):
        self.user_id           = user_id
        self.analyzer          = PitchAnalyzer()
        self.q_table           = QTable(user_id, session_count)
        self.reward_calculator = RewardCalculator()

        self._current_state  = None
        self._current_action = None
        self._fail_count     = 0

    def run(self, cents_deviation: float) -> dict:
        state  = self.analyzer.analyze(cents_deviation)
        self._update_fail_count(state)

        action   = self.q_table.best_action(state, self._fail_count)
        feedback = FEEDBACK_MESSAGE.get(action) if action else None

        self._current_state  = state
        self._current_action = action

        return self._build_result(state, action, feedback)

    def receive_next_state(self, next_cents_deviation: float) -> float:
        if self._current_state is None or self._current_action is None:
            return 0.0

        next_state      = self.analyzer.analyze(next_cents_deviation)
        next_fail_count = 0 if next_state == PitchState.GOOD else self._fail_count + 1

        reward = self.reward_calculator.calculate(
            self._current_state, next_state, self._fail_count
        )
        self.q_table.update(
            self._current_state, self._fail_count,
            self._current_action,
            reward,
            next_state, next_fail_count
        )
        return reward

    def save(self):
        self.q_table.save()

    def reset_analyzer(self):
        self.analyzer.reset()
        self._current_state  = None
        self._current_action = None
        self._fail_count     = 0

    @property
    def fail_count(self) -> int:
        return self._fail_count

    @property
    def current_state(self) -> PitchState:
        return self._current_state

    def _update_fail_count(self, state: PitchState):
        if state == PitchState.GOOD:
            self._fail_count = 0
        elif self._current_state is None:
            self._fail_count = 0
        else:
            self._fail_count += 1

    def _build_result(self, state, action, feedback) -> dict:
        return {
            "agent":       "pitch",
            "state":       state.value,
            "fail_count":  self._fail_count,
            "action":      action.value if action else None,
            "feedback":    feedback,
            "tripled_out": self._fail_count >= FAIL_THRESHOLD,
        }
