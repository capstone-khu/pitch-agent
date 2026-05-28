from enum import Enum
from ..state.pitch_state import PitchState
from ..config import FAIL_THRESHOLD


class PitchAction(Enum):
    PITCH_UP                = "PITCH_UP"                 # SA-01
    PITCH_DOWN              = "PITCH_DOWN"               # SA-02
    POSITIVE_PITCH          = "POSITIVE_PITCH"           # SA-04
    SWITCH_PITCH_TO_POSTURE = "SWITCH_PITCH_TO_POSTURE"  # SA-05


FEEDBACK_MESSAGE = {
    PitchAction.PITCH_UP:                "음정을 올리세요",
    PitchAction.PITCH_DOWN:              "음정을 내리세요",
    PitchAction.POSITIVE_PITCH:          "잘 하고 있습니다. 계속 유지하세요",
    PitchAction.SWITCH_PITCH_TO_POSTURE: "음정 교정이 반복 실패하고 있습니다. 자세를 점검하세요",
}


def get_available_actions(state: PitchState, fail_count: int) -> list:
    tripled = fail_count >= FAIL_THRESHOLD

    if state == PitchState.GOOD:
        return [PitchAction.POSITIVE_PITCH]

    if state in (PitchState.SHARP_SLIGHT, PitchState.SHARP_MAJOR):
        actions = [PitchAction.PITCH_DOWN]
        if tripled:
            actions.append(PitchAction.SWITCH_PITCH_TO_POSTURE)
        return actions

    if state in (PitchState.FLAT_SLIGHT, PitchState.FLAT_MAJOR):
        actions = [PitchAction.PITCH_UP]
        if tripled:
            actions.append(PitchAction.SWITCH_PITCH_TO_POSTURE)
        return actions

    return []
