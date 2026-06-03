import os

# 학습 하이퍼파라미터
ALPHA = 0.1
GAMMA = 0.9

# Q테이블 경로
_BASE           = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_DIR          = os.path.join(_BASE, "data", "q_tables")
COMMON_Q_TABLE_PATH  = os.path.join(_BASE, "data", "q_tables", "common_q_table.json")
USER_Q_TABLE_PATH    = os.path.join(_BASE, "data", "q_tables", "user_{user_id}_q_table.json")

# 세션 히스토리 경로
SESSION_HISTORY_DIR  = os.path.join(_BASE, "data", "session_history")
SESSION_HISTORY_PATH = os.path.join(_BASE, "data", "session_history", "user_{user_id}.json")

# 삼진아웃 임계값 (세션 단위)
SESSION_FAIL_THRESHOLD = 3

# 반복 레슨 후 성공 시 추가 리워드
REPEAT_LESSON_BONUS = 0.5

# Reward 값
REWARD = {
    "GOOD":            1.0,
    "PARTIAL":         0.5,
    "NO_CHANGE":      -0.3,
    "WORSE":          -0.8,
    "SUPERVISOR_HIT":  0.8,   # CALL_SUPERVISOR 후 개선
    "SUPERVISOR_MISS": -0.5,  # CALL_SUPERVISOR 후 개선 없음
}

# 개인화 스케줄
PERSONALIZATION_SCHEDULE = [
    {"min_sessions": 0,  "common": 1.0, "personal": 0.0},
    {"min_sessions": 10, "common": 0.7, "personal": 0.3},
    {"min_sessions": 30, "common": 0.3, "personal": 0.7},
    {"min_sessions": 50, "common": 0.0, "personal": 1.0},
]