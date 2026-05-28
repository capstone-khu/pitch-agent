import json
import os
from ..config import SESSION_HISTORY_DIR, SESSION_HISTORY_PATH, SESSION_FAIL_THRESHOLD


class SessionHistory:
    """
    세션(실행) 단위 마디별 실패 카운트 관리
    
    삼진아웃 정의:
    - 여러 세션에 걸쳐 같은 마디에서 같은 문제가 반복될 때
    - 세션 1: 마디3 실패 → count=1
    - 세션 2: 마디3 실패 → count=2
    - 세션 3: 마디3 실패 → count=3 → 삼진아웃!
    - 성공 시 해당 마디 카운트 초기화
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._path   = SESSION_HISTORY_PATH.format(user_id=user_id)
        self._data   = self._load()

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def record_measure_result(self, measure: int, success: bool, from_repeat_lesson: bool = False):
        """
        마디 결과 기록
        success:             해당 마디 성공 여부
        from_repeat_lesson:  반복 레슨 직후 정규 레슨인지 여부
        """
        key = str(measure)

        if success:
            # 성공 → 카운트 초기화
            self._data["pitch_fail"][key] = 0
            if from_repeat_lesson:
                # 반복 레슨 후 성공 → 보너스 리워드 플래그
                self._data["repeat_lesson_success"].append(measure)
        else:
            # 실패 → 카운트 +1
            current = self._data["pitch_fail"].get(key, 0)
            self._data["pitch_fail"][key] = current + 1

    def get_fail_count(self, measure: int) -> int:
        """특정 마디의 누적 실패 카운트 반환"""
        return self._data["pitch_fail"].get(str(measure), 0)

    def is_tripled_out(self, measure: int) -> bool:
        """특정 마디가 삼진아웃 상태인지 확인"""
        return self.get_fail_count(measure) >= SESSION_FAIL_THRESHOLD

    def get_tripled_out_measures(self) -> list:
        """삼진아웃 상태인 마디 목록 반환"""
        return [
            int(k) for k, v in self._data["pitch_fail"].items()
            if v >= SESSION_FAIL_THRESHOLD
        ]

    def set_session_type(self, session_type: str, repeat_measure: int = None):
        """
        세션 타입 설정
        session_type: 'normal' 또는 'repeat'
        repeat_measure: 반복 레슨 대상 마디 번호
        """
        self._data["last_session_type"]      = session_type
        self._data["repeat_target_measure"]  = repeat_measure

    def get_session_type(self) -> str:
        return self._data.get("last_session_type", "normal")

    def get_repeat_target(self) -> int | None:
        return self._data.get("repeat_target_measure")

    def is_repeat_lesson_success(self, measure: int) -> bool:
        """반복 레슨 후 해당 마디 성공 여부"""
        return measure in self._data.get("repeat_lesson_success", [])

    def clear_repeat_lesson_success(self):
        """반복 레슨 성공 기록 초기화"""
        self._data["repeat_lesson_success"] = []

    def save(self):
        os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def summary(self) -> dict:
        """현재 세션 히스토리 요약"""
        return {
            "user_id":              self.user_id,
            "pitch_fail":           self._data["pitch_fail"],
            "tripled_out_measures": self.get_tripled_out_measures(),
            "last_session_type":    self._data["last_session_type"],
            "repeat_target":        self._data["repeat_target_measure"],
        }

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "pitch_fail":             {},   # {마디번호: 연속 실패 세션 수}
            "last_session_type":      "normal",
            "repeat_target_measure":  None,
            "repeat_lesson_success":  [],
        }