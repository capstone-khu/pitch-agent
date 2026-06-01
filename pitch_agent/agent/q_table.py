import json
import os
import random

from ..state.pitch_state import PitchState
from ..action.pitch_action import PitchAction, get_available_actions
from ..config import ALPHA, GAMMA, PERSONALIZATION_SCHEDULE, COMMON_Q_TABLE_PATH, USER_Q_TABLE_PATH, Q_TABLE_DIR


class QTable:
    """
    Q테이블 관리
    - State = PitchState (fail_count 제거)
    - Q(S, A) <- Q(S, A) + α[R + γ·maxQ(S', A') - Q(S, A)]
    """

    def __init__(self, user_id: str, session_count: int = 0):
        self.user_id       = user_id
        self.session_count = session_count
        self.alpha         = ALPHA
        self.gamma         = GAMMA

        self.common_table   = self._load(COMMON_Q_TABLE_PATH)
        self.personal_table = self._load(USER_Q_TABLE_PATH.format(user_id=user_id))

    def get(self, state: PitchState, action: PitchAction) -> float:
        ratio        = self._get_personalization_ratio()
        common_val   = self._table_get(self.common_table,   state, action)
        personal_val = self._table_get(self.personal_table, state, action)

        if personal_val != 0.0:
            personal_ratio = max(ratio["personal"], 0.3)
            return (1.0 - personal_ratio) * common_val + personal_ratio * personal_val

        return ratio["common"] * common_val

    def best_action(self, state: PitchState, epsilon: float = 0.1) -> PitchAction | None:
        actions = get_available_actions(state)
        if not actions:
            return None

        if random.random() < epsilon:
            return random.choice(actions)

        return max(actions, key=lambda a: self.get(state, a))

    def update(self, state: PitchState, action: PitchAction,
               reward: float, next_state: PitchState):
        current_q  = self._table_get(self.personal_table, state, action)
        max_next_q = self._max_q(next_state)
        new_q      = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self._table_set(self.personal_table, state, action, new_q)

    def get_personal(self, state: PitchState, action: PitchAction) -> float:
        return self._table_get(self.personal_table, state, action)

    def save(self):
        os.makedirs(Q_TABLE_DIR, exist_ok=True)
        path = USER_Q_TABLE_PATH.format(user_id=self.user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.personal_table, f, ensure_ascii=False, indent=2)

    def _get_personalization_ratio(self) -> dict:
        ratio = {"common": 1.0, "personal": 0.0}
        for schedule in PERSONALIZATION_SCHEDULE:
            if self.session_count >= schedule["min_sessions"]:
                ratio = {"common": schedule["common"], "personal": schedule["personal"]}
        return ratio

    def _max_q(self, state: PitchState) -> float:
        actions = get_available_actions(state)
        if not actions:
            return 0.0
        return max(self.get(state, a) for a in actions)

    def _make_key(self, state: PitchState) -> str:
        return state.value

    def _table_get(self, table: dict, state: PitchState, action: PitchAction) -> float:
        return table.get(self._make_key(state), {}).get(action.value, 0.0)

    def _table_set(self, table: dict, state: PitchState, action: PitchAction, value: float):
        key = self._make_key(state)
        if key not in table:
            table[key] = {}
        table[key][action.value] = value

    def _load(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}