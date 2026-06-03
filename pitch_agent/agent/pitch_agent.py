from ..analyzer.pitch_analyzer import PitchAnalyzer
from ..agent.q_table import QTable
from ..reward.reward_calculator import RewardCalculator
from ..state.pitch_state import PitchState
from ..action.pitch_action import PitchAction, FEEDBACK_MESSAGE


class PitchAgent:
    """
    음정 에이전트
    - State = PitchState (fail_count 제거)
    - CALL_SUPERVISOR: Q값 기반으로 자연스럽게 선택
    """

    def __init__(self, user_id: str, session_count: int = 0):
        self.user_id           = user_id
        self.analyzer          = PitchAnalyzer()
        self.q_table           = QTable(user_id, session_count)
        self.reward_calculator = RewardCalculator()

        self._current_state  = None
        self._current_action = None

    def run(self, cents_deviation: float) -> dict:
        state    = self.analyzer.analyze(cents_deviation)
        action   = self.q_table.best_action(state)
        feedback = FEEDBACK_MESSAGE.get(action) if action else None

        self._current_state  = state
        self._current_action = action

        return self._build_result(state, action, feedback)

    def receive_next_state(self, next_cents_deviation: float) -> float:
        if self._current_state is None or self._current_action is None:
            return 0.0

        next_state           = self.analyzer.analyze(next_cents_deviation)
        action_was_supervisor = (self._current_action == PitchAction.CALL_SUPERVISOR)

        reward = self.reward_calculator.calculate(
            self._current_state, next_state, action_was_supervisor
        )
        self.q_table.update(
            self._current_state,
            self._current_action,
            reward,
            next_state
        )
        return reward

    def save(self):
        self.q_table.save()

    def reset_analyzer(self):
        self.analyzer.reset()
        self._current_state  = None
        self._current_action = None

    @property
    def current_state(self) -> PitchState:
        return self._current_state

    def _build_result(self, state: PitchState, action, feedback) -> dict:
        return {
            "agent":            "pitch",
            "state":            state.value,
            "action":           action.value if action else None,
            "feedback":         feedback,
            "call_supervisor":  action == PitchAction.CALL_SUPERVISOR,
        }