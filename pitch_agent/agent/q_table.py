import json
import os
import random

from ..state.pitch_state import PitchState
from ..action.pitch_action import PitchAction, get_available_actions
from ..config import ALPHA, GAMMA, PERSONALIZATION_SCHEDULE, COMMON_Q_TABLE_PATH, USER_Q_TABLE_PATH, Q_TABLE_DIR


class QTable:
    """
    Q테이블 관리
    - State = (PitchState, fail_count) 튜플을 키로 사용
    - Q(S, A) <- Q(S, A) + α[R + γ·maxQ(S', A') - Q(S, A)]
    - 사용자별 Q테이블 + 공통 Q테이블 혼합
    """

    def __init__(self, user_id: str, session_count: int = 0):
        self.user_id       = user_id
        self.session_count = session_count
        self.alpha         = ALPHA
        self.gamma         = GAMMA

        self.common_table   = self._load(COMMON_Q_TABLE_PATH)
        self.personal_table = self._load(USER_Q_TABLE_PATH.format(user_id=user_id))

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def get(self, state: PitchState, fail_count: int, action: PitchAction) -> float:
        """혼합 비율 적용한 Q값 반환"""
        ratio        = self._get_personalization_ratio()
        common_val   = self._table_get(self.common_table,   state, fail_count, action)
        personal_val = self._table_get(self.personal_table, state, fail_count, action)

        # 개인 데이터가 존재하면 최소 30% 반영 보장
        if personal_val != 0.0:
            personal_ratio = max(ratio["personal"], 0.3)
            common_ratio   = 1.0 - personal_ratio
            return common_ratio * common_val + personal_ratio * personal_val

        return ratio["common"] * common_val

    def best_action(self, state: PitchState, fail_count: int, epsilon: float = 0.1) -> PitchAction | None:
        """
        epsilon-greedy 전략으로 최적 Action 반환
        State + 실패카운트 조합으로 가능한 Action 목록 조회
        """
        actions = get_available_actions(state, fail_count)
        if not actions:
            return None

        # 탐험
        if random.random() < epsilon:
            return random.choice(actions)

        # 활용: Q값 가장 높은 Action 선택
        return max(actions, key=lambda a: self.get(state, fail_count, a))

    def update(self, state: PitchState, fail_count: int, action: PitchAction,
               reward: float, next_state: PitchState, next_fail_count: int):
        """Q(S,A) <- Q(S,A) + α[R + γ·maxQ(S',A') - Q(S,A)]"""
        current_q  = self._table_get(self.personal_table, state, fail_count, action)
        max_next_q = self._max_q(next_state, next_fail_count)
        new_q      = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self._table_set(self.personal_table, state, fail_count, action, new_q)

    def get_personal(self, state: PitchState, fail_count: int, action: PitchAction) -> float:
        """개인 테이블 Q값 직접 조회 (테스트용)"""
        return self._table_get(self.personal_table, state, fail_count, action)

    def save(self):
        os.makedirs(Q_TABLE_DIR, exist_ok=True)
        path = USER_Q_TABLE_PATH.format(user_id=self.user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.personal_table, f, ensure_ascii=False, indent=2)

    def save_common(self):
        os.makedirs(Q_TABLE_DIR, exist_ok=True)
        with open(COMMON_Q_TABLE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.common_table, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _get_personalization_ratio(self) -> dict:
        ratio = {"common": 1.0, "personal": 0.0}
        for schedule in PERSONALIZATION_SCHEDULE:
            if self.session_count >= schedule["min_sessions"]:
                ratio = {"common": schedule["common"], "personal": schedule["personal"]}
        return ratio

    def _max_q(self, state: PitchState, fail_count: int) -> float:
        actions = get_available_actions(state, fail_count)
        if not actions:
            return 0.0
        return max(self.get(state, fail_count, a) for a in actions)

    def _make_key(self, state: PitchState, fail_count: int) -> str:
        """(state, fail_count) 튜플을 JSON 키 문자열로 변환"""
        return f"{state.value}:{fail_count}"

    def _table_get(self, table: dict, state: PitchState, fail_count: int, action: PitchAction) -> float:
        key = self._make_key(state, fail_count)
        return table.get(key, {}).get(action.value, 0.0)

    def _table_set(self, table: dict, state: PitchState, fail_count: int, action: PitchAction, value: float):
        key = self._make_key(state, fail_count)
        if key not in table:
            table[key] = {}
        table[key][action.value] = value

    def _load(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
