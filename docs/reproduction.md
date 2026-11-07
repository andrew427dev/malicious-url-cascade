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

## 실험 A 내부 재현 결과

| 20번째 epoch | 원본 노트북 저장 출력 | 팀 실행 |
| --- | ---: | ---: |
| 검증 정확도 | 98.34% | 98.40% |
| 검증 BCE | 0.0565 | 0.05769 |
| 내부 악성 Recall | 내부 클래스별 수치 미기록 | 68,895 / 70,394 (97.87%) |

원본의 첫 학습 블록과 비교했습니다. 팀 실행의 전처리 후 검증 표본은 156,334건이며 정상 오탐은 1,003건, 악성 미탐은 1,499건입니다. 원본은 층화 분할 호출에 random_state를 지정하지 않았고 TPU에서 실행했습니다. 팀은 seed 42와 Kaggle CPU를 사용했습니다. 약 0.06%p의 정확도 차이를 같은 검증 표본에서의 개선으로 해석할 수 없습니다. 원본의 별도 데이터 평가값은 이 표에서 제외했습니다. [원본 노트북](https://github.com/munna0912/URL_CLASSIFIER/blob/835bd5e1710531407c0b5ca09da3d7c3de8d1bbc/Model_Training.ipynb) · [팀 실행·검증 파일 해시](../evidence/experiment-a-internal.json)

## 동결 모델로 게이트 선택과 오프라인 시연 실행

Kaggle CPU의 고정 환경에서 B 학습 출력의 `experiment-b` 디렉터리와 W2의 `split` 디렉터리를 복원합니다. 검증한 모델 묶음에서 계산한 `models-frozen.json`의 SHA-256을 아래 첫 명령에 사용합니다. 실행 디렉터리는 매번 새로 지정합니다.

```bash
python evaluation/gate.py --models "$MODELS" --models-sha256 "$MODELS_SHA256" \
  --split "$SPLIT" --out "$GATE_OUT"
python evaluation/predict.py --models "$MODELS" --policy "$GATE_OUT/policy.json" \
  --policy-sha256 "$POLICY_SHA256" --url 'https://example.org/demo'
```

게이트 선택은 전체 게이트 집합의 8개 후보에서 추가 미탐을 계산합니다. 허용 후보만 같은 최대 1,000건에서 실제 경로를 5회 측정하고, 최속 후보에서 5% 이내인 후보 중 가장 작은 임계값을 고릅니다. 최종 후보는 전체 게이트 집합에서 실제 선택적 호출로 다시 검증합니다. `gate-candidates.json`에는 후보별 성능, `gate-latencies.json`에는 원본 URL 문자열을 제외한 표본 ID와 반복별 지연이 남습니다. `policy.json`의 SHA-256을 기록한 후 두 번째 명령에 사용합니다. 여러 입력은 `--url` 대신 `--jsonl inputs.jsonl`로 지정합니다. 각 줄에는 고유한 `input_id`와 `raw_url`을 넣습니다.

## 동결 후 시험과 외부 평가

게이트 선택과 최종 평가는 각각 독립된 비공개 Kaggle CPU 노트북에서 실행합니다. 먼저 검증을 마친 B 모델 ZIP의 SHA-256과 Kaggle 버전 번호로 게이트 노트북을 생성합니다. 노트북 식별자는 각 실행의 `owner/slug`입니다.

```bash
python3.10 evaluation/build_notebook.py --stage gate \
  --model-notebook "$B_NOTEBOOK" --model-version "$B_VERSION" \
  --archive-sha256 "$B_ZIP_SHA256" --output /tmp/HUFS_URL_Gate.ipynb
```

첫 파일을 Kaggle에 가져오면 환경과 분할을 재구성하고, 게이트선택 집합에서 후보를 비교해 `hufs-b-gate-policy.zip`을 저장합니다. 완료된 실행의 ZIP을 내려받아 로그의 SHA-256과 대조한 뒤 최종 평가 노트북을 생성합니다.

```bash
python3.10 evaluation/build_notebook.py --stage final \
  --model-notebook "$B_NOTEBOOK" --model-version "$B_VERSION" \
  --archive-sha256 "$B_ZIP_SHA256" --reference-notebook "$A_NOTEBOOK" \
  --gate-notebook "$GATE_NOTEBOOK" --gate-version "$GATE_VERSION" \
  --gate-sha256 "$GATE_ZIP_SHA256" --output /tmp/HUFS_URL_Final.ipynb
```

최종 평가는 지정한 버전에서 정책·모델을 복원하고 무결성을 검증합니다. 최종 평가를 재시도해도 같은 게이트를 복원합니다. 입력 ZIP은 Kaggle에 연결한 입력에서 찾거나 지정한 노트북 버전에서 가져옵니다. 내부 시험과 독립 외부 집합에서는 네 가지 추론 경로를 실행합니다. 원본과 같은 외부 CSV 전처리를 사용하는 A 대조에는 A 모델과 어휘를 씁니다.

`evaluation-output` 아래에 선택한 정책, 후보별 게이트 비용, 내부·외부 성능표, 도메인 단위 2,000회 부트스트랩 결과, 사전 정의 오류 그룹, 순차 지연과 배치 처리량이 남습니다. 모델 로딩과 데이터 읽기 시간은 별도로 기록합니다. 외부에 정상 표본이 없으면 FPR과 Balanced Accuracy는 `null`입니다. 공개용 지표를 옮길 때는 집계표와 해시를 사용하고, 행별 예측과 전체 시간 로그는 비공개 실행 출력에 보관합니다.
