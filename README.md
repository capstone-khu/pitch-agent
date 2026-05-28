# Pitch Agent (음정 에이전트)

바이올린 협주 레슨 AI 플랫폼의 **음정 분석 에이전트**입니다.  
실시간 연주 음정을 감지하고, Q-Learning 기반으로 피드백을 제공합니다.

---

## 프로젝트 구조

```
pitch_agent/
├── state/
│   └── pitch_state.py          # PitchState 정의 (GOOD, SHARP_SLIGHT 등)
├── action/
│   └── pitch_action.py         # PitchAction 정의 + 피드백 메시지
├── analyzer/
│   └── pitch_analyzer.py       # cents 편차 → PitchState 변환
├── reward/
│   └── reward_calculator.py    # Reward/Penalty 계산
├── agent/
│   ├── pitch_agent.py          # 에이전트 핵심 로직
│   ├── q_table.py              # Q테이블 관리 (저장/로드/업데이트)
│   └── session_history.py      # 세션 단위 실패 카운트 관리
├── integration/
│   ├── metadata_extractor.py   # 정석 연주 mp3 → 악보 메타데이터 추출
│   ├── score_loader.py         # 메타데이터 로드 + 옥타브 정규화 + cents 계산
│   ├── realtime_bridge.py      # 실시간 마이크 입력 → PitchAgent 연결
│   └── score_metadata.json     # 반짝반짝 작은별 A장조 메타데이터
├── data/
│   ├── q_tables/               # 사용자별 Q테이블 (자동 생성)
│   └── session_history/        # 세션별 실패 카운트 (자동 생성)
├── tests/
│   └── test_pitch_agent.py     # 단위 테스트 (35개)
├── config.py                   # 하이퍼파라미터 설정
└── requirements.txt            # 의존성 패키지
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

# 실시간 레슨 실행
py -m pitch_agent.integration.realtime_bridge
```

`pitch_agent/integration/reference.mp3` 파일이 있으면 자동 재생 + 마이크 감지 시작

---

## 설계 구조

### 1. State 정의 (음정 에이전트)

| State ID | 값             | 설명                            |
| -------- | -------------- | ------------------------------- |
| P-S0     | `GOOD`         | 정상 (±30cents 이내)            |
| P-S1     | `SHARP_SLIGHT` | 음정 약간 높음 (30~100cents)    |
| P-S2     | `SHARP_MAJOR`  | 음정 많이 높음 (100cents 이상)  |
| P-S3     | `FLAT_SLIGHT`  | 음정 약간 낮음 (-30~-100cents)  |
| P-S4     | `FLAT_MAJOR`   | 음정 많이 낮음 (-100cents 이하) |

### 2. Action 정의

| Action ID | 액션명                    | 피드백                                                  | 발동 조건                  |
| --------- | ------------------------- | ------------------------------------------------------- | -------------------------- |
| SA-01     | `PITCH_UP`                | "음정을 올리세요"                                       | FLAT_SLIGHT / FLAT_MAJOR   |
| SA-02     | `PITCH_DOWN`              | "음정을 내리세요"                                       | SHARP_SLIGHT / SHARP_MAJOR |
| SA-04     | `POSITIVE_PITCH`          | "잘 하고 있습니다. 계속 유지하세요"                     | GOOD                       |
| SA-05     | `SWITCH_PITCH_TO_POSTURE` | "음정 교정이 반복 실패하고 있습니다. 자세를 점검하세요" | 삼진아웃                   |

### 3. Reward / Penalty

| 조건               | 값   |
| ------------------ | ---- |
| GOOD 전환          | +1.0 |
| 심각 → 경미 개선   | +0.5 |
| 변화 없음          | -0.3 |
| 동일 액션 2회 실패 | -0.6 |
| 동일 액션 3회 실패 | -1.0 |
| 악화               | -0.8 |

### 4. Q-Learning 공식

```
Q(S, A) ← Q(S, A) + α[R + γ·maxQ(S', A') - Q(S, A)]

S  = (PitchState, fail_count) 튜플
α  = 0.1 (학습률)
γ  = 0.9 (할인율)
```

### 5. 삼진아웃 (세션 단위)

```
세션 1: 마디 N 실패 → fail_count = 1
세션 2: 마디 N 실패 → fail_count = 2
세션 3: 마디 N 실패 → fail_count = 3 → 삼진아웃!
→ 슈퍼바이저 개입 + 해당 마디 반복 레슨 권장
```

### 6. 개인화 Q테이블

```
신규 유저  → 공통 Q테이블 100%
10세션 후  → 공통 70% + 개인 30%
30세션 후  → 공통 30% + 개인 70%
50세션 후  → 개인 Q테이블 100%
```

---

## 에이전트 출력 양식

### 음정 분석 결과 (analysis)

```json
{
  "timestamp": 12.35,
  "target_note": "E5",
  "target_hz": 659.26,
  "actual_hz": 654.85,
  "cents_deviation": -11.6
}
```

### 상태 + Reward (rl_state)

```json
{
  "previous_state": ["FLAT_SLIGHT", 0],
  "current_state": ["FLAT_SLIGHT", 1],
  "action": "SA-01_PITCH_UP",
  "reward": -0.3,
  "reason": "state unchanged after feedback",
  "updated_q": -0.027
}
```

### 에이전트 출력 (agent output)

```json
{
  "agent": "pitch",
  "state": "FLAT_SLIGHT",
  "fail_count": 1,
  "action": "PITCH_UP",
  "feedback": "음정을 올리세요",
  "tripled_out": false
}
```

### 마디 단위 피드백 (measure feedback)

```json
{
  "measure": 2,
  "avg_cents": -9.0,
  "dominant_state": "FLAT_MAJOR",
  "session_fail_count": 2,
  "feedback": "음정을 올리세요",
  "tripled_out": false
}
```

### 세션 종료 요약 (session summary)

```json
{
  "user_id": "user_001",
  "session_type": "normal",
  "results": {
    "1": { "success": false, "cumulative_fail": 2 },
    "2": { "success": false, "cumulative_fail": 2 },
    "3": { "success": true, "cumulative_fail": 0 }
  },
  "tripled_out_measures": [1, 2]
}
```

---

## 테스트 실행

```bash
cd capstone-design
py -m pytest pitch_agent/tests/test_pitch_agent.py -v
```

총 **35개** 테스트 통과

---

## 의존성

```
librosa        # 오디오 로딩
soundfile      # 오디오 저장
noisereduce    # 노이즈 제거
swift-f0       # 피치 디텍션
numpy          # 수치 계산
pyaudio        # 실시간 마이크 입력
pygame         # 음원 재생
matplotlib     # 시각화 (선택)
```
