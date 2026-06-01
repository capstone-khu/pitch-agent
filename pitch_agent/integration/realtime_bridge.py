"""
실시간 연주 → SwiftF0 피치 디텍션 → cents 계산 → PitchAgent 연결
- 50ms 프레임: cents 측정 + 로그 출력만
- 마디 단위: State 판단 → Q테이블 조회 → Action 선택 → 피드백 → Q업데이트
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
from ..config import REPEAT_LESSON_BONUS

SAMPLE_RATE    = 48000
FRAME_MS       = 50
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_MS / 1000)
CONF_THRESHOLD = 0.5
MIN_VALID_FRAMES = 5


class RealtimeBridge:

    def __init__(self, user_id: str, metadata_path: str, session_count: int = 0,
                 session_type: str = "normal"):
        self.score_loader    = ScoreLoader(metadata_path)
        self.pitch_agent     = PitchAgent(user_id, session_count)
        self.session_history = SessionHistory(user_id)
        self.model           = core.SwiftF0()
        self.session_type    = session_type

        self._audio_queue  = queue.Queue()
        self._is_running   = False
        self._start_time   = None
        self._use_pygame   = False
        self._prev_f0      = 0.0

        # 마디 단위 버퍼
        self._current_measure     = None
        self._measure_cents       = []   # 유효 cents 누적

        # 마디 단위 Q업데이트용
        self._prev_measure_state  = None  # 이전 마디 대표 State
        self._prev_measure_action = None  # 이전 마디 선택된 Action

        # 세션 결과
        self._session_measure_results = {}

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
                self._use_pygame = False
                self._start_time = time.time()
        else:
            self._use_pygame = False
            self._start_time = time.time()

        self._is_running = True
        analysis_thread  = threading.Thread(target=self._analysis_loop, daemon=True)
        analysis_thread.start()

        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=FRAME_SIZE,
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
            if self._current_measure is not None:
                self._flush_measure_feedback()
            self._is_running = False
            self._end_session()

    def process_frame(self, audio_chunk: np.ndarray, timestamp: float) -> dict | None:
        # 노이즈 제거 + 피치 디텍션
        chunk_denoised = nr.reduce_noise(
            y=audio_chunk, sr=SAMPLE_RATE, prop_decrease=0.9, stationary=True
        )
        res_f0     = self.model.detect_from_array(chunk_denoised, sample_rate=SAMPLE_RATE)
        actual_hz  = 0.0
        confidence = 0.0

        if len(res_f0.pitch_hz) > 0:
            actual_hz  = float(np.median(res_f0.pitch_hz))
            confidence = float(np.median(res_f0.confidence))
            if self._prev_f0 > 0 and abs(actual_hz - self._prev_f0 * 2) < self._prev_f0 * 0.1:
                actual_hz /= 2
            if actual_hz > 1100:
                actual_hz /= 2
            self._prev_f0 = actual_hz

        # 목표 음표 조회 (소리 없어도 항상 로그 출력)
        target_note = self.score_loader.get_target_at(timestamp)

        if confidence < CONF_THRESHOLD or actual_hz <= 0:
            if target_note is not None:
                print(f"  [음표] {target_note['note']:<4} | "
                      f"목표: {target_note['hz_exact']:>7.2f}Hz | "
                      f"실제: 입력 없음")
            return None

        if target_note is None:
            return None

        # cents 편차 계산 (옥타브 정규화 포함)
        cents = self.score_loader.calc_cents_deviation(actual_hz, timestamp)
        if cents is None:
            return None

        # 옥타브 정규화된 Hz 계산 (로그용)
        normalized_hz = self.score_loader._normalize_octave(actual_hz, target_note['hz_exact'])

        # 음표 단위 로그 (정규화된 Hz 출력)
        print(f"  [음표] {target_note['note']:<4} | "
              f"목표: {target_note['hz_exact']:>7.2f}Hz | "
              f"실제: {normalized_hz:>7.2f}Hz | "
              f"편차: {cents:+.1f}cents")

        # 마디 전환 감지
        measure = target_note.get("measure")
        if measure is not None:
            if measure != self._current_measure:
                if self._current_measure is not None:
                    self._flush_measure_feedback()
                self._current_measure = measure
                self._measure_cents   = []

            # cents 누적 (전부 사용 - 옥타브 정규화 이후 값)
            self._measure_cents.append(cents)

        return None  # 피드백은 마디 단위로만 출력

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
        """마디 끝: cents 평균 → State 판단 → Action 선택 → 피드백 → Q업데이트"""
        if not self._measure_cents:
            print(f"\n{'='*60}")
            print(f"[마디 {self._current_measure}] 측정 불가 (소리 감지 없음)")
            print(f"{'='*60}\n")
            return

        if len(self._measure_cents) < MIN_VALID_FRAMES:
            print(f"\n{'='*60}")
            print(f"[마디 {self._current_measure}] 측정 불가 "
                  f"(유효 프레임 {len(self._measure_cents)}개 < 최소 {MIN_VALID_FRAMES}개)")
            print(f"{'='*60}\n")
            return

        avg_cents = float(np.mean(self._measure_cents))

        # 평균 cents → State 판단
        from ..analyzer.pitch_analyzer import PitchAnalyzer
        from ..state.pitch_state import PitchState
        from ..action.pitch_action import PitchAction, FEEDBACK_MESSAGE

        analyzer    = PitchAnalyzer()
        curr_state  = analyzer.analyze(avg_cents)

        # Q테이블 조회 → Action 선택 (마디 단위 1번)
        action   = self.pitch_agent.q_table.best_action(curr_state)
        feedback = FEEDBACK_MESSAGE.get(action, "잘 하고 있습니다. 계속 유지하세요")

        # 마디 성공/실패
        measure_success = (curr_state == PitchState.GOOD)

        # ── 마디 단위 Q업데이트 ──────────────────
        reward = 0.0
        if self._prev_measure_state is not None and self._prev_measure_action is not None:
            was_supervisor = (self._prev_measure_action == PitchAction.CALL_SUPERVISOR)
            reward = self.pitch_agent.reward_calculator.calculate(
                self._prev_measure_state, curr_state, was_supervisor
            )
            self.pitch_agent.q_table.update(
                self._prev_measure_state,
                self._prev_measure_action,
                reward,
                curr_state
            )
            print(f"  [Q업데이트] {self._prev_measure_state.value} x "
                  f"{self._prev_measure_action.value} → reward: {reward:+.1f} "
                  f"({self._prev_measure_state.value} → {curr_state.value})")

        # 현재 마디 State/Action 저장 (다음 마디 Q업데이트용)
        self._prev_measure_state  = curr_state
        self._prev_measure_action = action

        # 세션 히스토리 업데이트
        from_repeat = (self.session_type == "repeat")
        self.session_history.record_measure_result(
            self._current_measure, measure_success, from_repeat
        )
        self._session_measure_results[self._current_measure] = measure_success

        # 슈퍼바이저 호출 페이로드 생성
        if action == PitchAction.CALL_SUPERVISOR:
            reward_for_supervisor = reward if (
                self._prev_measure_state is not None and
                self._prev_measure_action is not None
            ) else 0.0
            q_value = self.pitch_agent.q_table.get(curr_state, action)
            supervisor_payload = {
                "agent":    "pitch",
                "measure":  self._current_measure,
                "state":    curr_state.value,
                "action_id":"SA-04",
                "action":   "CALL_SUPERVISOR",
                "feedback": feedback,
                "reward":   round(reward_for_supervisor, 3),
                "q":        round(q_value, 3),
                "meta":     {}
            }
            self._call_supervisor(supervisor_payload)

        # 출력
        print(f"\n{'='*60}")
        print(f"[마디 {self._current_measure}] "
              f"평균 편차: {avg_cents:+.1f}cents | "
              f"State: {curr_state.value}")
        print(f"  → 피드백: {feedback}")
        print(f"{'='*60}\n")

    def _call_supervisor(self, payload: dict):
        """
        슈퍼바이저 에이전트 호출 (현재는 로그 출력으로 대체)
        실제 연동 시 이 메서드에서 슈퍼바이저 API 호출
        """
        import json
        print(f"  → [SUPERVISOR CALL] 페이로드 전달:")
        print(f"    {json.dumps(payload, ensure_ascii=False, indent=4)}")

    def _end_session(self):
        self.session_history.save()
        self.pitch_agent.save()

        all_measures = sorted(set(
            n.get("measure") for n in self.score_loader.notes
            if n.get("measure") is not None
        ))

        print("\n" + "=" * 60)
        print("[세션 종료 요약]")
        for m in all_measures:
            if m in self._session_measure_results:
                success = self._session_measure_results[m]
                status  = "성공" if success else "실패"
                print(f"  마디 {m:>2}: {status}")
            else:
                print(f"  마디 {m:>2}: 측정 불가")
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
    if bridge._current_measure is not None:
        bridge._flush_measure_feedback()
    bridge._end_session()


if __name__ == "__main__":
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "score_metadata.json")
    reference     = os.path.join(base_dir, "reference.mp3")

    if not os.path.exists(metadata_path):
        print("score_metadata.json이 없습니다.")
    else:
        bridge = RealtimeBridge(
            user_id       = "user_001",
            metadata_path = metadata_path,
            session_type  = "normal"
        )
        bridge.start(
            reference_audio = reference if os.path.exists(reference) else None
        )