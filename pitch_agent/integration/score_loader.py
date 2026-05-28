import json
import numpy as np


class ScoreLoader:
    """
    메타데이터 로더
    - 현재 타임스탬프 → 목표 음정(Hz) 반환
    - 목표 Hz와 실제 Hz 비교 → cents 편차 계산
    - 옥타브 정규화: 실제 Hz를 목표 Hz와 가장 가까운 옥타브로 자동 보정
    """

    def __init__(self, metadata_path: str):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.notes = data["notes"]
        print(f"악보 로드 완료: {len(self.notes)}개 음표")

    def get_target_at(self, timestamp: float) -> dict | None:
        """현재 타임스탬프에 해당하는 목표 음표 반환"""
        for note in self.notes:
            if note["start"] <= timestamp < note["end"]:
                return note
        return None

    def calc_cents_deviation(self, actual_hz: float, timestamp: float) -> float | None:
        """
        실제 Hz와 목표 Hz의 cents 편차 계산
        옥타브 정규화 적용:
        → 실제 Hz를 목표 Hz와 가장 가까운 옥타브로 자동 조정 후 비교
        """
        target = self.get_target_at(timestamp)
        if target is None or actual_hz <= 0:
            return None

        target_hz = target["hz_exact"]
        if target_hz <= 0:
            return None

        # 옥타브 정규화
        normalized_hz = self._normalize_octave(actual_hz, target_hz)

        cents = 1200 * np.log2(normalized_hz / target_hz)
        return round(cents, 2)

    def get_note_sequence(self) -> list:
        return self.notes

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _normalize_octave(self, actual_hz: float, target_hz: float) -> float:
        """
        실제 Hz를 목표 Hz와 가장 가까운 옥타브로 정규화
        예) 목표: A4(440Hz), 실제: A5(880Hz) → 440Hz로 변환
            목표: A4(440Hz), 실제: A3(220Hz) → 440Hz로 변환
        """
        normalized = actual_hz
        # 옥타브 올리기 (실제가 목표보다 너무 낮은 경우)
        while normalized < target_hz / 1.5:
            normalized *= 2
        # 옥타브 내리기 (실제가 목표보다 너무 높은 경우)
        while normalized > target_hz * 1.5:
            normalized /= 2
        return normalized


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loader   = ScoreLoader(os.path.join(base_dir, "score_metadata.json"))

    print("\n[옥타브 정규화 테스트]")
    # A4 목표(440Hz)에서 다른 옥타브 입력 시
    test_cases = [
        (880.0, 6.5),   # A5 → A4로 정규화
        (220.0, 6.5),   # A3 → A4로 정규화
        (440.0, 6.5),   # A4 → 그대로
        (438.0, 6.5),   # A4 약간 낮음 → -8cents
    ]
    for hz, ts in test_cases:
        cents = loader.calc_cents_deviation(hz, ts)
        target = loader.get_target_at(ts)
        print(f"  목표: {target['note']}({target['hz_exact']}Hz) | "
              f"입력: {hz}Hz | 정규화 후 편차: {cents:+.1f}cents")