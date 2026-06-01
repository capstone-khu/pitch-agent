# Pitch Agent

바이올린 협주 레슨 AI 플랫폼의 음정 분석 에이전트입니다.
실시간 연주 음정을 감지하고, Q-Learning 기반으로 마디 단위 피드백을 제공합니다.

---

## 프로젝트 구조

```
pitch_agent/
├── state/
│   └── pitch_state.py          # PitchState 정의
├── action/
│   └── pitch_action.py         # PitchAction 정의 + 피드백 메시지
├── analyzer/
│   └── pitch_analyzer.py       # cents 편차 → PitchState 변환
├── reward/
│   └── reward_calculator.py    # Reward 계산
├── agent/
│   ├── pitch_agent.py          # 에이전트 핵심 로직
│   ├── q_table.py              # Q테이블 관리 (저장/로드/업데이트)
│   └── session_history.py      # 세션 단위 마디별 성공/실패 기록
├── integration/
│   ├── metadata_extractor.py   # 정석 연주 mp3 → 악보 메타데이터 추출
│   ├── score_loader.py         # 메타데이터 로드 + 옥타브 정규화 + cents 계산
│   ├── realtime_bridge.py      # 실시간 마이크 입력 → PitchAgent 연결
│   └── score_metadata.json     # 반짝반짝 작은별 D장조 메타데이터 (12마디)
├── data/
│   ├── q_tables/               # 사용자별 Q테이블 (자동 생성)
│   └── session_history/        # 세션별 마디 기록 (자동 생성)
├── tests/
│   └── test_pitch_agent.py     # 단위 테스트 (35개)
├── config.py                   # 하이퍼파라미터 및 경로 설정
└── requirements.txt
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행

```bash
# capstone-design 폴더에서 실행
cd capstone-design

py -m pitch_agent.integration.realtime_bridge
```

`pitch_agent/integration/reference.mp3` 파일이 있으면 자동 재생 + 마이크 감지 시작합니다.

---

## 설계 구조

### State

| State | 설명 | cents 기준 |
|-------|------|-----------|
| GOOD | 정상 | ±30cents 이내 |
| SHARP_SLIGHT | 음정 약간 높음 | +30 ~ +100cents |
| SHARP_MAJOR | 음정 많이 높음 | +100cents 이상 |
| FLAT_SLIGHT | 음정 약간 낮음 | -30 ~ -100cents |
| FLAT_MAJOR | 음정 많이 낮음 | -100cents 이하 |

State는 마디 내 유효 cents의 **평균값**으로 판단합니다.

### Action

| Action ID | 액션명 | 피드백 | 발동 조건 |
|-----------|--------|--------|----------|
| SA-01 | PITCH_UP | "음정을 올리세요" | FLAT_SLIGHT / FLAT_MAJOR |
| SA-02 | PITCH_DOWN | "음정을 내리세요" | SHARP_SLIGHT / SHARP_MAJOR |
| SA-03 | POSITIVE_PITCH | "잘 하고 있습니다. 계속 유지하세요" | GOOD |
| SA-04 | CALL_SUPERVISOR | "슈퍼바이저에게 도움을 요청합니다" | SHARP / FLAT (Q값 기준 자연 선택) |

### Reward

| 조건 | 값 |
|------|----|
| GOOD 전환 | +1.0 |
| 심각 → 경미 개선 (MAJOR → SLIGHT) | +0.5 |
| 변화 없음 | -0.3 |
| 악화 (SLIGHT → MAJOR) | -0.8 |
| CALL_SUPERVISOR 후 개선 | +0.8 |
| CALL_SUPERVISOR 후 개선 없음 | -0.5 |

### Q-Learning

```
Q(S, A) ← Q(S, A) + α[R + γ·maxQ(S', A') - Q(S, A)]

S  = PitchState (마디 평균 cents 기반)
A  = PitchAction
α  = 0.1 (학습률)
γ  = 0.9 (할인율)
```

Q테이블은 **마디 단위**로 1회 업데이트됩니다.
(50ms 프레임은 cents 측정만 수행)

### 개인화 Q테이블

```
0세션  → 공통 Q테이블 100%
10세션 → 공통 70% + 개인 30%
30세션 → 공통 30% + 개인 70%
50세션 → 개인 Q테이블 100%
```

### 옥타브 정규화

바이올린 기종마다 감지되는 옥타브가 다를 수 있어 목표 Hz와 가장 가까운 옥타브로 자동 보정합니다.

```
목표: A4 (440Hz) / 실제 감지: A5 (880Hz)
→ 880 / 2 = 440Hz 로 정규화 후 cents 계산
```

---

## 악보 메타데이터

반짝반짝 작은별 D장조, 0.5마디 단위 12마디 구성

```
마디  1: D4 D4 A4 A4   (도도솔솔)
마디  2: B4 B4 A4      (라라솔)
마디  3: G4 G4 F#4 F#4 (파파미미)
마디  4: E4 E4 D4      (레레도)
마디  5: A4 A4 G4 G4   (솔솔파파)
마디  6: F#4 F#4 E4    (미미레)
마디  7: A4 A4 G4 G4   (솔솔파파)
마디  8: F#4 F#4 E4    (미미레)
마디  9: D4 D4 A4 A4   (도도솔솔)
마디 10: B4 B4 A4      (라라솔)
마디 11: G4 G4 F#4 F#4 (파파미미)
마디 12: E4 E4 D4      (레레도)
```

---

## 에이전트 출력 JSON 양식

### 마디 피드백

```json
{
    "agent": "pitch",
    "measure": 4,
    "state": "FLAT_MAJOR",
    "action_id": "SA-01",
    "action": "PITCH_UP",
    "feedback": "음정을 올리세요",
    "reward": -0.3,
    "q": -0.027,
    "meta": {}
}
```

### CALL_SUPERVISOR 발동 시

```json
{
    "agent": "pitch",
    "measure": 4,
    "state": "FLAT_MAJOR",
    "action_id": "SA-04",
    "action": "CALL_SUPERVISOR",
    "feedback": "슈퍼바이저에게 도움을 요청합니다",
    "reward": -0.8,
    "q": -0.064,
    "meta": {}
}
```

### 세션 종료 요약

```json
{
    "user_id": "user_001",
    "session_type": "normal",
    "results": {
        "1": "성공",
        "2": "성공",
        "3": "실패",
        "4": "측정 불가"
    }
}
```

---

## 테스트

```bash
cd capstone-design
py -m pytest pitch_agent/tests/test_pitch_agent.py -v
```

35개 테스트 통과

---

## 의존성

```
librosa        오디오 로딩
noisereduce    노이즈 제거
swift-f0       피치 디텍션
numpy          수치 계산
pyaudio        실시간 마이크 입력
pygame         음원 재생
```
