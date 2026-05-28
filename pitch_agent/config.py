# 학습 하이퍼파라미터
ALPHA = 0.1   # 학습률
GAMMA = 0.9   # 할인율

# 삼진아웃 임계값
FAIL_THRESHOLD = 3

# 개인화 비율 (세션 수 기반)
PERSONALIZATION_SCHEDULE = [
    {"min_sessions": 0,  "common": 1.0, "personal": 0.0},
    {"min_sessions": 10, "common": 0.7, "personal": 0.3},
    {"min_sessions": 30, "common": 0.3, "personal": 0.7},
    {"min_sessions": 50, "common": 0.0, "personal": 1.0},
]

# Q테이블 저장 경로
Q_TABLE_DIR = "data/q_tables"
COMMON_Q_TABLE_PATH = f"{Q_TABLE_DIR}/common_q_table.json"
USER_Q_TABLE_PATH   = f"{Q_TABLE_DIR}/user_{{user_id}}_q_table.json"

# Reward 값 정의
REWARD = {
    "GOOD":          1.0,   # State GOOD 전환
    "PARTIAL":       0.5,   # 심각 → 경미로 개선
    "NO_CHANGE":    -0.3,   # 변화 없음
    "REPEAT_FAIL":  -0.6,   # 동일 액션 2회 실패
    "FULL_FAIL":    -1.0,   # 동일 액션 3회 실패
    "WORSE":        -0.8,   # 악화
}

# 세션 히스토리 경로
SESSION_HISTORY_DIR  = "data/session_history"
SESSION_HISTORY_PATH = "data/session_history/user_{user_id}.json"

# 삼진아웃 임계값 (세션 단위)
SESSION_FAIL_THRESHOLD = 3

# 반복 레슨 후 성공 시 추가 리워드
REPEAT_LESSON_BONUS = 0.5