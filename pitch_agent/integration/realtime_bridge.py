"""
실시간 연주 → SwiftF0 피치 디텍션 → cents 계산 → PitchAgent 연결
- 음표 단위: 목표Hz vs 실제Hz 로그 출력 (개발용)
- 마디 단위: PitchAgent 피드백 출력
- 삼진아웃: 세션 단위 누적 실패 카운트 기반
"""
import numpy as np
import threading
import queue
import time
import os
import noisereduce as nr
from swift_f0 import core

from .score_loader import ScoreLoader
from ..agent.pitch_agent import PitchAgent
from ..agent.session_history import SessionHistory
from ..config import SESSION_FAIL_THRESHOLD, REPEAT_LESSON_BONUS

SAMPLE_RATE    = 48000
FRAME_MS       = 50
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_MS / 1000)
CONF_THRESHOLD = 0.5


class RealtimeBridge:

    def __init__(self, user_id: str, metadata_path: str, session_count: int = 0,
                 session_type: str = "normal"):
        self.score_loader    = ScoreLoader(metadata_path)
        self.pitch_agent     = PitchAgent(user_id, session_count)
        self.session_history = SessionHistory(user_id)
        self.model           = core.SwiftF0()
        self.session_type    = session_type  # "normal" or "repeat"

        self._audio_queue  = queue.Queue()
        self._is_running   = False
        self._start_time   = None
        self._use_pygame   = False

        self._prev_f0      = 0.0
        self._prev_result  = None
        self._prev_hz      = None

        # 마디 단위 피드백 버퍼
        self._current_measure    = None
        self._measure_cents      = []
        self._measure_states     = []

        # 세션 내 마디별 결과 기록 (세션 종료 시 session_history에 반영)
        self._session_measure_results = {}  # {마디번호: 성공여부}

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def start(self, reference_audio: str = None):
        try:
            import pyaudio
        except ImportError:
            print("pyaudio 설치 필요: pip install pyaudio")
            return

        if reference_audio and os.path.exists(reference_audio):
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(reference_audio)
                pygame.mixer.music.play()
                self._use_pygame = True
                print(f"음원 재생 시작: {reference_audio}")
            except ImportError:
                print("pygame 없음 → 자체 타이머로 실행")
                self._use_pygame = False
                self._start_time = time.time()
        else:
            self._use_pygame = False
            self._start_time = time.time()
            print("reference.mp3 없음 → 마이크만으로 실행")

        self._is_running = True

        analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        analysis_thread.start()

        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
            stream_callback=self._audio_callback
        )

        print(f"마이크 감지 시작 [{self.session_type} 레슨] (Ctrl+C로 종료)")
        print("=" * 60)

        try:
            stream.start_stream()
            while stream.is_active() and self._is_running:
                time.sleep(0.01)
                if self._use_pygame:
                    try:
                        import pygame
                        if not pygame.mixer.music.get_busy():
                            print("\n음원 재생 완료 → 자동 종료")
                            break
                    except:
                        pass
        except KeyboardInterrupt:
            print("\n종료")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            if self._use_pygame:
                try:
                    import pygame
                    pygame.mixer.music.stop()
                except:
                    pass
            # 마지막 마디 피드백
            if self._current_measure is not None and self._measure_cents:
                self._flush_measure_feedback()
            self._is_running = False
            self._end_session()

    def process_frame(self, audio_chunk: np.ndarray, timestamp: float) -> dict | None:
        chunk_denoised = nr.reduce_noise(
            y=audio_chunk, sr=SAMPLE_RATE, prop_decrease=0.9, stationary=True
        )

        res_f0     = self.model.detect_from_array(chunk_denoised, sample_rate=SAMPLE_RATE)
        actual_hz  = 0.0
        confidence = 0.0

        if len(res_f0.pitch_hz) > 0:
            actual_hz  = float(np.median(res_f0.pitch_hz))
            confidence = float(np.median(res_f0.confidence))

            if self._prev_f0 > 0:
                if abs(actual_hz - (self._prev_f0 * 2)) < (self._prev_f0 * 0.1):
                    actual_hz /= 2
            if actual_hz > 1100:
                actual_hz /= 2
            self._prev_f0 = actual_hz

        if confidence < CONF_THRESHOLD or actual_hz <= 0:
            return None

        target_note = self.score_loader.get_target_at(timestamp)
        if target_note is None:
            return None

        cents = self.score_loader.calc_cents_deviation(actual_hz, timestamp)
        if cents is None:
            return None

        # 음표 단위 로그
        print(f"  [음표] {target_note['note']:<4} | "
              f"목표: {target_note['hz_exact']:>7.2f}Hz | "
              f"실제: {actual_hz:>7.2f}Hz | "
              f"편차: {cents:+.1f}cents")

        # 마디 단위 피드백 처리
        measure = target_note.get("measure")
        if measure is not None:
            if measure != self._current_measure:
                if self._current_measure is not None and self._measure_cents:
                    self._flush_measure_feedback()
                self._current_measure = measure
                self._measure_cents   = []
                self._measure_states  = []
                self.pitch_agent.reset_analyzer()

            self._measure_cents.append(cents)

        if self._prev_result is not None and self._prev_hz is not None:
            self.pitch_agent.receive_next_state(actual_hz)

        result = self.pitch_agent.run(cents)
        if result["state"] != "GOOD":
            self._measure_states.append(result["state"])

        self._prev_result = result
        self._prev_hz     = actual_hz

        return result

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _get_timestamp(self) -> float:
        if self._use_pygame:
            try:
                import pygame
                pos = pygame.mixer.music.get_pos()
                return pos / 1000.0 if pos >= 0 else 0.0
            except:
                pass
        return time.time() - self._start_time

    def _flush_measure_feedback(self):
        """마디 끝: 누적 cents 평균 + State로 피드백 출력 + 세션 히스토리 반영"""
        if not self._measure_cents:
            # 유효 프레임 없음 → 측정 불가, 세션 히스토리 반영 X
            print(f"\n{'='*60}")
            print(f"[마디 {self._current_measure}] 측정 불가 (소리 감지 없음)")
            print(f"{'='*60}\n")
            return

        valid_cents = [c for c in self._measure_cents if -200 <= c <= 200]

        # 최소 유효 프레임 수 기준 (마디 내 50ms 프레임 기준 최소 5개 = 0.25초 이상)
        MIN_VALID_FRAMES = 5

        if not valid_cents:
            # 이상치만 있고 유효값 없음 → 측정 불가
            print(f"\n{'='*60}")
            print(f"[마디 {self._current_measure}] 측정 불가 (유효 음정 감지 없음)")
            print(f"{'='*60}\n")
            return

        if len(valid_cents) < MIN_VALID_FRAMES:
            # 유효 프레임 수 부족 → 측정 불가
            print(f"\n{'='*60}")
            print(f"[마디 {self._current_measure}] 측정 불가 "
                  f"(유효 프레임 {len(valid_cents)}개 < 최소 {MIN_VALID_FRAMES}개)")
            print(f"{'='*60}\n")
            return

        avg_cents = float(np.mean(valid_cents))

        from collections import Counter
        if self._measure_states:
            dominant_state = Counter(self._measure_states).most_common(1)[0][0]
        else:
            dominant_state = "GOOD"

        # 마디 성공/실패 판단 (GOOD이면 성공)
        measure_success = (dominant_state == "GOOD")

        # 세션 히스토리 업데이트
        from_repeat = (self.session_type == "repeat")
        self.session_history.record_measure_result(
            self._current_measure, measure_success, from_repeat
        )
        self._session_measure_results[self._current_measure] = measure_success

        # 세션 누적 실패 카운트 조회
        fail_count  = self.session_history.get_fail_count(self._current_measure)
        tripled_out = self.session_history.is_tripled_out(self._current_measure)

        # 피드백 결정
        from ..action.pitch_action import FEEDBACK_MESSAGE, get_available_actions
        from ..state.pitch_state import PitchState
        try:
            state_enum  = PitchState(dominant_state)
            q_fail      = SESSION_FAIL_THRESHOLD if tripled_out else 0
            best_action = self.pitch_agent.q_table.best_action(state_enum, q_fail, epsilon=0.0)
            feedback    = FEEDBACK_MESSAGE.get(best_action, "잘 하고 있습니다. 계속 유지하세요")
        except Exception:
            feedback = "잘 하고 있습니다. 계속 유지하세요"

        # 반복 레슨 후 성공 시 보너스 리워드
        repeat_bonus = (from_repeat and measure_success)

        print(f"\n{'='*60}")
        print(f"[마디 {self._current_measure}] "
              f"평균 편차: {avg_cents:+.1f}cents | "
              f"주요 State: {dominant_state} | "
              f"세션 실패 누적: {fail_count}회")
        print(f"  → 피드백: {feedback}")
        if repeat_bonus:
            print(f"  🎉 반복 레슨 후 개선! 보너스 리워드 +{REPEAT_LESSON_BONUS}")
        if tripled_out:
            print(f"  ⚠️  삼진아웃! 슈퍼바이저 개입 필요 (마디 {self._current_measure} 반복 레슨 권장)")
        print(f"{'='*60}\n")

    def _end_session(self):
        """세션 종료 처리"""
        self.session_history.save()
        self.pitch_agent.save()

        # 전체 마디 목록
        all_measures = sorted(set(
            n.get("measure") for n in self.score_loader.notes
            if n.get("measure") is not None
        ))

        tripled = self.session_history.get_tripled_out_measures()
        print("\n" + "=" * 60)
        print("[세션 종료 요약]")
        for m in all_measures:
            if m in self._session_measure_results:
                success = self._session_measure_results[m]
                status  = "✅ 성공" if success else "❌ 실패"
                count   = self.session_history.get_fail_count(m)
                print(f"  마디 {m}: {status} | 누적 실패: {count}회")
            else:
                print(f"  마디 {m}: ⚪ 측정 불가")
        if tripled:
            print(f"\n  ⚠️  삼진아웃 마디: {tripled}")
            print(f"  → 다음 세션에서 마디 {tripled} 반복 레슨 권장")
        print("=" * 60)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        audio     = np.frombuffer(in_data, dtype=np.float32)
        timestamp = self._get_timestamp()
        self._audio_queue.put((timestamp, audio.copy()))
        return (None, pyaudio.paContinue)

    def _analysis_loop(self):
        while self._is_running:
            try:
                timestamp, audio_chunk = self._audio_queue.get(timeout=1.0)
                self.process_frame(audio_chunk, timestamp)
            except queue.Empty:
                continue


# ──────────────────────────────────────────
# 파일 기반 시뮬레이션 (테스트용)
# ──────────────────────────────────────────
def simulate_from_file(audio_path: str, metadata_path: str, user_id: str = "test",
                       session_type: str = "normal"):
    import librosa
    bridge = RealtimeBridge(user_id, metadata_path, session_type=session_type)
    y, sr  = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    print(f"시뮬레이션 시작: {audio_path} [{session_type} 레슨]\n")
    for i in range(0, len(y) - FRAME_SIZE, FRAME_SIZE):
        chunk     = y[i:i + FRAME_SIZE].astype(np.float32)
        timestamp = i / float(sr)
        bridge.process_frame(chunk, timestamp)
    if bridge._current_measure is not None and bridge._measure_cents:
        bridge._flush_measure_feedback()
    bridge._end_session()


if __name__ == "__main__":
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "score_metadata.json")
    reference     = os.path.join(base_dir, "reference.mp3")

    if not os.path.exists(metadata_path):
        print("score_metadata.json이 없습니다. 먼저 실행하세요:")
        print("py -m pitch_agent.integration.metadata_extractor --input 정석연주.mp3")
    else:
        bridge = RealtimeBridge(
            user_id       = "user_001",
            metadata_path = metadata_path,
            session_type  = "normal"
        )
        bridge.start(
            reference_audio = reference if os.path.exists(reference) else None
        )