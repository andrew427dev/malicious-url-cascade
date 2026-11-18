# 정제 재실험의 공개 집계 증거

같은 검증15% 전체를 모델 선택·보정·임계값 선택에 재사용한2026-11-16 실행이다. 이전5/5/5 결과를 이미 확인한 시험의 추가 평가이며 독립 맹검 시험이 아니다.

- 주/도메인 학습은 각각20epoch 완료. 새 학습 결과를 사용하고 각 분할의GRU는 해당 분할 내 비교 정책에서 공유한다.
- data-audit: 최종717,793행, 클래스 수·분할·URL/인코딩 유일성 PASS.
- result-audit:283개 정책 기록×2분할, 실제 선택적 예측·호출·시간35회×2분할 PASS.
- diagnostics: 오류/복구 사례, 평가 수정 전후 가중치·시험 행·경량 점수 불변 PASS. 수정으로 달라진 시험GRU 판정은 두 분할 모두0건. 모든 τ도 동일하다.
- learning-curves:10개 적합 기록, 중첩 학습 부분집합·전체 학습량 앵커 PASS.
- 패키지:125멤버 ZIP의 CRC·124파일 SHA-256 및 경로 검사 PASS. 압축 해제한 코드 self-test PASS. 새 환경의 전체 재학습은 NOT_RUN.
- Windows 단위 검사:56개 중53 PASS/3 FAIL/0 ERROR. 새 계약 검사11개는 PASS. 실패3개는 기반ba96c12에서도 재현한 Windows 경로/기존 cloud ZIP 차이이며 이번 코드의 신규 실패가 아니다.
- Ruff0.17.5 검사 PASS. 공식 PyPI에서 작업 폴더에만 설치했고 기존 Python 환경은 바꾸지 않았다.

가중치·행별 점수·정제 배열·URL 코퍼스는 Git에 포함하지 않았다. 집계 JSON의 해시는 원래 실행 산출물을 식별하며 누락된 큰 파일의 실제 재검사는 로컬 결과 또는 제출 ZIP에서 수행한다. `environment-summary.json`은 로컬 실행 파일 경로만 제거한 환경 요약이다. 원래환경기록은 ZIP에 들어 있다. 학습/평가/데이터 생성 원문 snapshot은 실제 사용된 코드를 구분한다.

[보고서](../../docs/clean-rerun/report-ko.pdf) · [재현 명령](../../experiments/clean_rerun/README.md) · [피드백 이행 확인](../../docs/clean-rerun/feedback-check.md)
