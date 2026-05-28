from pitch_agent.agent.pitch_agent import PitchAgent


def main():
    """
    실행 예시
    실제 환경에서는 오디오 분석 모듈로부터 cents_deviation을 받아옴
    """
    agent = PitchAgent(user_id="user_001", session_count=0)

    # 시뮬레이션: 마디별 음정 편차 입력 (cents 단위)
    # 양수 = 높음(sharp), 음수 = 낮음(flat)
    session_data = [
        110.0,   # SHARP_MAJOR
        110.0,   # SHARP_MAJOR (실패 카운트 +1)
        110.0,   # SHARP_MAJOR (실패 카운트 +2, 삼진아웃 임박)
        50.0,    # SHARP_SLIGHT (부분 개선)
        10.0,    # GOOD
        -110.0,  # FLAT_MAJOR
        35.0,    # DRIFT 진행 중
        40.0,
        45.0,
    ]

    print("=" * 60)
    print(f"[Pitch Agent 세션 시작] 사용자: {agent.user_id}")
    print("=" * 60)

    for i, deviation in enumerate(session_data):
        print(f"\n[마디 {i+1}] 음정 편차: {deviation:+.1f} cents")
        result = agent.run(deviation)
        print(f"  State     : {result['state']}")
        print(f"  Action    : {result['action']}")
        print(f"  피드백     : {result['feedback']}")
        print(f"  실패 카운트 : {result['fail_count']}")
        print(f"  삼진아웃   : {result['tripled_out']}")

        # 다음 마디가 있으면 Reward 계산
        if i + 1 < len(session_data):
            reward = agent.receive_next_state(session_data[i + 1])
            if reward != 0.0:
                print(f"  Reward    : {reward:+.1f}")

    # 세션 종료 후 Q테이블 저장
    agent.save()
    print("\n" + "=" * 60)
    print("Q테이블 저장 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
