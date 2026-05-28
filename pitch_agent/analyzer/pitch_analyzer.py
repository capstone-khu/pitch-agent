from ..state.pitch_state import PitchState

SLIGHT_THRESHOLD = 30   # 30cents 미만 → GOOD
MAJOR_THRESHOLD  = 100  # 100cents 이상 → MAJOR


class PitchAnalyzer:
    """
    cents 편차 → PitchState 변환
    DRIFT 제거: 단순 cents 기준 분류만 수행
    """

    def analyze(self, cents_deviation: float) -> PitchState:
        abs_dev = abs(cents_deviation)

        if abs_dev < SLIGHT_THRESHOLD:
            return PitchState.GOOD

        if cents_deviation > 0:
            return PitchState.SHARP_MAJOR if cents_deviation >= MAJOR_THRESHOLD else PitchState.SHARP_SLIGHT

        return PitchState.FLAT_MAJOR if abs_dev >= MAJOR_THRESHOLD else PitchState.FLAT_SLIGHT

    def reset(self):
        pass  # DRIFT 제거로 히스토리 불필요
