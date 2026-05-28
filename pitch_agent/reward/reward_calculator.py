from ..state.pitch_state import PitchState, STATE_SEVERITY, SHARP_GROUP, FLAT_GROUP
from ..config import REWARD


class RewardCalculator:
    """
    이전 State와 현재 State를 비교해서 Reward/Penalty 계산
    """

    def calculate(self, prev_state: PitchState, curr_state: PitchState, fail_count: int) -> float:
        """
        prev_state : 피드백을 주기 전 State
        curr_state : 피드백을 준 후 다음 연주의 State
        fail_count : 현재 동일 액션 연속 실패 횟수
        """

        # GOOD으로 전환 → 완전 개선
        if curr_state == PitchState.GOOD:
            return REWARD["GOOD"]

        # 악화된 경우
        if self._is_worse(prev_state, curr_state):
            return REWARD["WORSE"]

        # 부분 개선 (심각 → 경미)
        if self._is_partial_improvement(prev_state, curr_state):
            return REWARD["PARTIAL"]

        # 변화 없음 → 실패 횟수에 따라 패널티 증가
        if fail_count >= 3:
            return REWARD["FULL_FAIL"]
        elif fail_count == 2:
            return REWARD["REPEAT_FAIL"]
        else:
            return REWARD["NO_CHANGE"]

    def _is_worse(self, prev: PitchState, curr: PitchState) -> bool:
        """심각도가 높아졌는지 확인"""
        return STATE_SEVERITY.get(curr, 0) > STATE_SEVERITY.get(prev, 0)

    def _is_partial_improvement(self, prev: PitchState, curr: PitchState) -> bool:
        """
        같은 방향이면서 심각도가 낮아진 경우
        예) SHARP_MAJOR → SHARP_SLIGHT
        """
        prev_severity = STATE_SEVERITY.get(prev, 0)
        curr_severity = STATE_SEVERITY.get(curr, 0)

        same_sharp = prev in SHARP_GROUP and curr in SHARP_GROUP
        same_flat  = prev in FLAT_GROUP  and curr in FLAT_GROUP

        return (same_sharp or same_flat) and curr_severity < prev_severity
