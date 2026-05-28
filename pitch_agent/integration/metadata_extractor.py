"""
정석 연주 mp3 → 음표 단위 메타데이터 추출
사용법: py -m pitch_agent.integration.metadata_extractor --input 음원파일.mp3
"""
import json
import numpy as np
import os
import argparse
import librosa
import noisereduce as nr
import threading
import queue
from swift_f0 import core

SAMPLE_RATE    = 48000
FRAME_MS       = 50
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_MS / 1000)
CONF_THRESHOLD = 0.6   # 신뢰도 기준 상향

# 바이올린 멜로디 유효 Hz 범위 (A4 ~ G5)
HZ_MIN = 400.0   # A4 아래는 노이즈/반주로 간주
HZ_MAX = 800.0   # G5 위는 이상치로 간주

MIN_DURATION = 0.15  # 최소 음표 길이 (초)

audio_queue     = queue.Queue()
pitch_results   = []
denoised_chunks = []


def frequency_to_note(f):
    if f <= 0:
        return "N/A"
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    try:
        midi = int(round(12 * np.log2(f / 440.0) + 69))
        return f"{notes[midi % 12]}{midi // 12 - 1}"
    except:
        return "N/A"

def hz_to_midi(f):
    if f <= 0: return -1
    return int(round(12 * np.log2(f / 440.0) + 69))

def midi_to_hz(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def thread_preprocessing(y, sr):
    for i in range(0, len(y) - FRAME_SIZE, FRAME_SIZE):
        chunk          = y[i:i + FRAME_SIZE]
        chunk_denoised = nr.reduce_noise(y=chunk, sr=sr, prop_decrease=0.9, stationary=True)
        timestamp      = i / float(sr)
        audio_queue.put((timestamp, chunk_denoised))
        denoised_chunks.append(chunk_denoised)
    audio_queue.put(None)

def thread_analysis(sr):
    model   = core.SwiftF0()
    prev_f0 = 0.0
    while True:
        item = audio_queue.get()
        if item is None:
            break
        timestamp, chunk = item
        res_f0           = model.detect_from_array(chunk, sample_rate=sr)
        current_f0, conf = 0.0, 0.0
        if len(res_f0.pitch_hz) > 0:
            current_f0 = float(np.median(res_f0.pitch_hz))
            conf       = float(np.median(res_f0.confidence))
            # 옥타브 보정
            if prev_f0 > 0 and abs(current_f0 - (prev_f0 * 2)) < (prev_f0 * 0.1):
                current_f0 /= 2
            if current_f0 > 1100:
                current_f0 /= 2
            prev_f0 = current_f0

        note = frequency_to_note(current_f0) if conf > 0.4 else "N/A"
        pitch_results.append({
            "timestamp":  round(timestamp, 3),
            "hz":         round(current_f0, 2),
            "confidence": round(conf, 3),
            "note":       note
        })
        audio_queue.task_done()


def extract_notes(raw_path: str, output_path: str):
    with open(raw_path, "r") as f:
        data = json.load(f)

    # 1. 신뢰도 + Hz 범위 필터링 (멜로디 음역대만)
    valid = [
        fr for fr in data["frames"]
        if fr["confidence"] >= CONF_THRESHOLD
        and fr["note"] != "N/A"
        and HZ_MIN <= fr["hz"] <= HZ_MAX
    ]

    print(f"전체 프레임: {len(data['frames'])}개")
    print(f"유효 프레임 (conf>={CONF_THRESHOLD}, {HZ_MIN}~{HZ_MAX}Hz): {len(valid)}개")

    if not valid:
        print("유효한 프레임이 없습니다.")
        return

    # 2. 연속된 같은 MIDI 음표로 묶기
    notes         = []
    current_midi  = hz_to_midi(valid[0]["hz"])
    note_start    = valid[0]["timestamp"]
    hz_bucket     = [valid[0]["hz"]]

    for fr in valid[1:]:
        midi = hz_to_midi(fr["hz"])
        if midi == current_midi:
            hz_bucket.append(fr["hz"])
        else:
            duration = fr["timestamp"] - note_start
            if duration >= MIN_DURATION:
                avg_hz    = float(np.mean(hz_bucket))
                note_name = frequency_to_note(avg_hz)
                notes.append({
                    "index":    len(notes),
                    "note":     note_name,
                    "midi":     current_midi,
                    "hz":       round(avg_hz, 2),
                    "hz_exact": round(midi_to_hz(current_midi), 2),
                    "start":    round(note_start, 3),
                    "end":      round(fr["timestamp"], 3),
                    "duration": round(duration, 3),
                    "measure":  None
                })
            current_midi = midi
            note_start   = fr["timestamp"]
            hz_bucket    = [fr["hz"]]

    # 마지막 음표
    if hz_bucket:
        last_ts  = valid[-1]["timestamp"]
        duration = last_ts - note_start
        if duration >= MIN_DURATION:
            avg_hz    = float(np.mean(hz_bucket))
            note_name = frequency_to_note(avg_hz)
            notes.append({
                "index":    len(notes),
                "note":     note_name,
                "midi":     current_midi,
                "hz":       round(avg_hz, 2),
                "hz_exact": round(midi_to_hz(current_midi), 2),
                "start":    round(note_start, 3),
                "end":      round(last_ts, 3),
                "duration": round(duration, 3),
                "measure":  None
            })

    print(f"\n추출된 음표 {len(notes)}개:")
    for n in notes:
        print(f"  [{n['index']:>2}] {n['note']:<4} | {n['hz']:>7.2f}Hz | "
              f"{n['start']:.2f}s ~ {n['end']:.2f}s ({n['duration']:.2f}s)")

    # 3. 마디 번호 자동 할당
    notes = assign_measures(notes)

    # 4. 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "source":      raw_path,
            "total_notes": len(notes),
            "hz_range":    [HZ_MIN, HZ_MAX],
            "notes":       notes
        }, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")
    return notes


def assign_measures(notes: list) -> list:
    """
    반짝반짝 작은별 A장조 마디 구조에 맞게 음표에 마디 번호 할당
    음표 순서와 시간 기반으로 자동 할당
    """
    # 반짝반짝 A장조 예상 음표 시퀀스 (MIDI 번호)
    # A4=69, B4=71, C#5=73, D5=74, E5=76, F#5=78
    TWINKLE_SEQUENCE = [
        # 마디 1: A4 A4 E5 E5 F#5 F#5 E5
        (69, 1), (69, 1), (76, 1), (76, 1), (78, 1), (78, 1), (76, 1),
        # 마디 2: D5 D5 C#5 C#5 B4 B4 A4
        (74, 2), (74, 2), (73, 2), (73, 2), (71, 2), (71, 2), (69, 2),
        # 마디 3: E5 E5 D5 D5 C#5 C#5 B4
        (76, 3), (76, 3), (74, 3), (74, 3), (73, 3), (73, 3), (71, 3),
        # 마디 4: E5 E5 D5 D5 C#5 C#5 B4
        (76, 4), (76, 4), (74, 4), (74, 4), (73, 4), (73, 4), (71, 4),
        # 마디 5: A4 A4 E5 E5 F#5 F#5 E5
        (69, 5), (69, 5), (76, 5), (76, 5), (78, 5), (78, 5), (76, 5),
        # 마디 6: D5 D5 C#5 C#5 B4 B4 A4
        (74, 6), (74, 6), (73, 6), (73, 6), (71, 6), (71, 6), (69, 6),
    ]

    # 추출된 음표와 시퀀스 매칭 (시간 순서 기반)
    seq_idx = 0
    for note in notes:
        if seq_idx >= len(TWINKLE_SEQUENCE):
            break
        expected_midi, measure = TWINKLE_SEQUENCE[seq_idx]
        # MIDI가 일치하거나 인접 음표(±1)면 할당
        if abs(note["midi"] - expected_midi) <= 1:
            note["measure"] = measure
            seq_idx += 1
        else:
            # 다음 시퀀스와 비교
            for lookahead in range(1, 4):
                if seq_idx + lookahead < len(TWINKLE_SEQUENCE):
                    exp_midi, meas = TWINKLE_SEQUENCE[seq_idx + lookahead]
                    if abs(note["midi"] - exp_midi) <= 1:
                        note["measure"] = meas
                        seq_idx += lookahead + 1
                        break

    # 마디 번호 없는 음표 → 인접 마디로 채우기
    for i, note in enumerate(notes):
        if note["measure"] is None:
            prev_m = next((notes[j]["measure"] for j in range(i-1, -1, -1) if notes[j]["measure"]), None)
            next_m = next((notes[j]["measure"] for j in range(i+1, len(notes)) if notes[j]["measure"]), None)
            note["measure"] = prev_m or next_m

    # 마디별 확인 출력
    print("\n마디별 음표:")
    measures = sorted(set(n["measure"] for n in notes if n["measure"]))
    for m in measures:
        ns = [n for n in notes if n["measure"] == m]
        print(f"  마디 {m}: " + " → ".join(f"{n['note']}({n['start']:.1f}s)" for n in ns))

    return notes


def run_full_pipeline(input_mp3: str):
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    raw_path      = os.path.join(base_dir, "raw_pitch_data.json")
    metadata_path = os.path.join(base_dir, "score_metadata.json")

    print(f"음원 로딩: {input_mp3}")
    y, sr = librosa.load(input_mp3, sr=SAMPLE_RATE, mono=True)
    print(f"총 길이: {len(y)/sr:.2f}초\n")

    t1 = threading.Thread(target=thread_preprocessing, args=(y, sr))
    t2 = threading.Thread(target=thread_analysis,      args=(sr,))
    t1.start(); t2.start()
    t1.join();  t2.join()

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": input_mp3, "sample_rate": sr, "frame_ms": FRAME_MS,
            "total_frames": len(pitch_results), "frames": pitch_results
        }, f, ensure_ascii=False, indent=2)

    extract_notes(raw_path, metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="정석 연주 mp3로 악보 메타데이터 생성")
    parser.add_argument("--input", type=str, required=True, help="정석 연주 mp3 파일 경로")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"파일을 찾을 수 없습니다: {args.input}")
    else:
        run_full_pipeline(args.input)