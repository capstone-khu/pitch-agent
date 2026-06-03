from ..state.pitch_state import PitchState, STATE_SEVERITY, SHARP_GROUP, FLAT_GROUP
from ..config import REWARD


class RewardCalculator:

    def calculate(self, prev_state: PitchState, curr_state: PitchState,
                  action_was_supervisor: bool = False) -> float:
        """
        prev_state:            피드백 전 State
        curr_state:            피드백 후 다음 State
        action_was_supervisor: CALL_SUPERVISOR 액션이었는지 여부
        """
        if curr_state == PitchState.GOOD:
            if action_was_supervisor:
                return REWARD["SUPERVISOR_HIT"]   # +0.8
            return REWARD["GOOD"]                  # +1.0

        if self._is_worse(prev_state, curr_state):
            return REWARD["WORSE"]                 # -0.8

        if self._is_partial_improvement(prev_state, curr_state):
            return REWARD["PARTIAL"]               # +0.5

        # 변화 없음
        if action_was_supervisor:
            return REWARD["SUPERVISOR_MISS"]       # -0.5
        return REWARD["NO_CHANGE"]                 # -0.3

    def _is_worse(self, prev: PitchState, curr: PitchState) -> bool:
        return STATE_SEVERITY.get(curr, 0) > STATE_SEVERITY.get(prev, 0)

    def _is_partial_improvement(self, prev: PitchState, curr: PitchState) -> bool:
        prev_sev = STATE_SEVERITY.get(prev, 0)
        curr_sev = STATE_SEVERITY.get(curr, 0)
        same_sharp = prev in SHARP_GROUP and curr in SHARP_GROUP
        same_flat  = prev in FLAT_GROUP  and curr in FLAT_GROUP
        return (same_sharp or same_flat) and curr_sev < prev_sev