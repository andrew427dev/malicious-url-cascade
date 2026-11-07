# 실험 재현

## 코드와 데이터의 출처

재현 대상은 Munna Ram의 `URL_CLASSIFIER` 커밋 `835bd5e1710531407c0b5ca09da3d7c3de8d1bbc`입니다. GRU·수치 결합 모델과 실험 A의 전처리는 이 커밋의 구현을 사용합니다. 학습 실행기는 필요한 Python 모듈 3개를 원저자의 저장소에서 받아 SHA-256을 확인합니다. [고정 원본](https://github.com/munna0912/URL_CLASSIFIER/tree/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc) · [모듈·데이터 명세](../configs/upstream.json)

원본 Git LFS에서 내려받을 수 없었던 CSV 3개는 같은 저장소의 고정된 이전 리비전에서, 참고 가중치는 원저자의 Hugging Face 리비전에서 확보했습니다. 파일 크기와 SHA-256을 원래 LFS 포인터와 대조했습니다. B 학습은 자체 가중치와 어휘를 생성합니다. [파일별 출처 및 검증 기록](../evidence/source-integrity.json)

## 로컬에서 확인할 수 있는 것

저장소 루트에서 실행합니다. 첫 명령은 표준 라이브러리만 사용합니다. 원본 비교 테스트는 모듈이 준비된 환경에서 실행됩니다.

```bash
python3.10 -m unittest discover -s tests -v
python3.10 scripts/fetch_reference.py
python3.10 -m unittest discover -s tests -p test_reference_parity.py -v
```

마지막 검사는 예약 예시 도메인, IPv4·IPv6, 유니코드 숫자, 쿼리 등을 조합한 입력 192개에 대해 팀 코드가 만든 18개 수치와 정제 문자열을 고정 원본과 비교합니다. 학습 데이터나 TensorFlow는 필요하지 않습니다.

## Kaggle CPU에서 전체 실험을 재현하는 순서

Kaggle Notebook에 아래 파일을 가져오고 인터넷 연결을 켭니다. 초기 설치는 Python 3.10 전용 환경에 TensorFlow 2.12.0, NumPy 1.23.5, pandas 1.5.3, scikit-learn 1.2.2를 구성합니다. 전체 패키지 버전은 [CPU](../environment/w1-cpu.txt)와 [GPU](../environment/w1-gpu.txt) 실측 목록에서 확인할 수 있습니다.

1. **데이터 준비와 파일럿** · [준비 노트북](../notebooks/HUFS_CompSec_Cloud.ipynb)에서 `STAGE`를 바꿔 같은 작업 디렉터리에서 `diagnose → setup → fetch → prepare-b → audit-b → smoke → pilot` 순서로 실행합니다. `prepare-b`가 출력한 `runs/<run-id>/split` 절대경로를 이후 단계의 `SPLIT_DIR`에 지정합니다. `smoke`와 `pilot`은 동결된 B 학습 행에서만 표본을 뽑습니다. [기준 분할 명세](../evidence/b-split-manifest.json)
2. **A 재현** · 새 Kaggle CPU 세션에서 [A 노트북](../notebooks/HUFS_CompSec_Experiment_A.ipynb)을 실행합니다. 자체 환경과 데이터를 준비하고 A 전용 어휘·모델을 학습합니다. 과거 smoke 복원 ZIP이 연결돼 있으면 해시를 확인하고 복원 검사도 실행합니다. 과거 실행 버전과 현재 재현용 노트북은 각각의 파일·코드 해시로 구분합니다. [과거 제출본 식별자](../evidence/experiment-a-version.json)
3. **B 모델 선택** · 별도 Kaggle CPU 세션에서 [B 노트북](../notebooks/HUFS_CompSec_Experiment_B.ipynb)을 실행합니다. 기존 분할 체크포인트를 연결하면 해시를 대조해 복원합니다. 체크포인트가 없으면 공개 원천에서 데이터를 다시 확보하고 분할을 생성·감사합니다. 동결 당시 매니페스트와 학습·모델 선택 파일의 해시가 일치해야 학습이 시작됩니다. [선택 코드](../scripts/train_b.py)
4. **게이트와 최종 시험** · 모델 선택 후 게이트 전용 집합에서 후보 `0, .001, .005, .01, .02, .05, .10, .20`을 비교합니다. 모델을 한 번 로드하고 특징 계산부터 결과 순서 복원까지 측정합니다. 선택한 게이트와 모델을 고정한 뒤 내부 시험과 외부 평가 결과를 산출하는 것이 다음 실험 단계입니다.

노트북의 `setup`은 고정된 원본 모듈을, `fetch`는 데이터 파일을 받습니다. 인터넷 연결은 이 준비 작업에 사용합니다. 입력 데이터에 든 주소로 요청을 보내는 단계는 없습니다. 노트북은 결과가 담기지 않은 실행용 파일입니다. 각 실행의 `status.json`, 명령 로그, 해시, 집계 지표를 해당 실험 버전과 함께 보관합니다.
