from enum import Enum


class PitchState(Enum):
    GOOD         = "GOOD"
    SHARP_SLIGHT = "SHARP_SLIGHT"  # 반음 미만 높음
    SHARP_MAJOR  = "SHARP_MAJOR"   # 반음 이상 높음
    FLAT_SLIGHT  = "FLAT_SLIGHT"   # 반음 미만 낮음
    FLAT_MAJOR   = "FLAT_MAJOR"    # 반음 이상 낮음


# 심각도 (Reward 부분 개선 판단용)
STATE_SEVERITY = {
    PitchState.GOOD:         0,
    PitchState.SHARP_SLIGHT: 1,
    PitchState.FLAT_SLIGHT:  1,
    PitchState.SHARP_MAJOR:  2,
    PitchState.FLAT_MAJOR:   2,
}

SHARP_GROUP = {PitchState.SHARP_SLIGHT, PitchState.SHARP_MAJOR}
FLAT_GROUP  = {PitchState.FLAT_SLIGHT,  PitchState.FLAT_MAJOR}
