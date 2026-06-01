from enum import Enum
from ..state.pitch_state import PitchState


class PitchAction(Enum):
    PITCH_UP         = "PITCH_UP"          # SA-01 음정을 올리세요
    PITCH_DOWN       = "PITCH_DOWN"        # SA-02 음정을 내리세요
    POSITIVE_PITCH   = "POSITIVE_PITCH"    # SA-03 잘 하고 있습니다
    CALL_SUPERVISOR  = "CALL_SUPERVISOR"   # SA-04 슈퍼바이저 호출


FEEDBACK_MESSAGE = {
    PitchAction.PITCH_UP:        "음정을 올리세요",
    PitchAction.PITCH_DOWN:      "음정을 내리세요",
    PitchAction.POSITIVE_PITCH:  "잘 하고 있습니다. 계속 유지하세요",
    PitchAction.CALL_SUPERVISOR: "슈퍼바이저에게 도움을 요청합니다",
}


def get_available_actions(state: PitchState) -> list:
    """
    State별 가능한 Action 목록
    SHARP/FLAT 모두 CALL_SUPERVISOR 선택지 포함
    → Q값으로 자연스럽게 판단
    """
    if state == PitchState.GOOD:
        return [PitchAction.POSITIVE_PITCH]

    if state in (PitchState.SHARP_SLIGHT, PitchState.SHARP_MAJOR):
        return [PitchAction.PITCH_DOWN, PitchAction.CALL_SUPERVISOR]

    if state in (PitchState.FLAT_SLIGHT, PitchState.FLAT_MAJOR):
        return [PitchAction.PITCH_UP, PitchAction.CALL_SUPERVISOR]

    return []