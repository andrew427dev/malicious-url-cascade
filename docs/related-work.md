# 관련 연구와 보고서 인용 근거

현재 보고서의 참고문헌에는 원저자 저장소와 도구 문서만 있고 학술 논문이 없다. 아래 6편은 서론, 확장 설계, 한계 절에서 우리 판단을 뒷받침하는 근거다.

**원문 확인 방법(2026-11-17):** 각 논문의 공개 PDF를 직접 내려받아 제목, 저자, 게재처와 아래에 요약한 내용을 원문에서 확인했다. 쪽 번호는 내려받은 PDF의 쪽이다. 저작권 때문에 PDF는 저장소에 넣지 않고 링크만 둔다. 인용은 요약이며, 따옴표 안의 짧은 구절만 원문 그대로다.

## 1. 문제 배경

### [R1] Sahoo, Liu, Hoi — 악성 URL 탐지 개관

- **서지:** Doyen Sahoo, Chenghao Liu, Steven C.H. Hoi. *Malicious URL Detection using Machine Learning: A Survey.* arXiv:1701.07179, 2017 (확인한 판: v3, 2019년 8월).
- **원문:** https://arxiv.org/abs/1701.07179
- **확인한 내용:**
  - 초록(PDF 1쪽): 블랙리스트는 모든 주소를 담을 수 없고 새로 생긴 악성 URL을 잡지 못한다. 이 때문에 머신러닝 기반 탐지가 연구되어 왔다.
  - 4절: 배치 학습은 개념 변화(concept drift)에 약하다. 클래스 불균형도 연구 과제로 다룬다.
- **우리 보고서에서 쓰는 곳:** 서론의 문제 정의. 한계 절에서 "수집 시점이 다른 URL에 대한 성능은 보장하지 않는다"는 문장의 근거.

### [R2] Ma, Saul, Savage, Voelker — URL 문자열만으로 하는 분류

- **서지:** Justin Ma, Lawrence K. Saul, Stefan Savage, Geoffrey M. Voelker. *Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs.* KDD '09, Paris, 2009. ACM.
- **원문:** https://cseweb.ucsd.edu/~savage/papers/KDD09.pdf
- **확인한 내용:**
  - 1쪽: 페이지 내용을 받지 않고 들어오는 URL만으로 판단하는 "lightweight URL classification"을 제안한다.
  - 페이지를 받지 않으면 사용자에게 더 안전하고, 분류 비용도 낮다.
  - 2.1절: 특징을 어휘(lexical) 특징과 호스트 기반 특징으로 나눈다.
  - 초록: 분류 정확도는 95–99%다.
- **우리 보고서에서 쓰는 곳:** 원 모델의 18개 수치 특징이 어휘 특징 계열이라는 설명. 실험에서 사이트에 접속하지 않는 이유. 단, 이 논문은 호스트 기반 특징도 쓰고 우리는 쓰지 않는다는 차이를 함께 적는다.

### [R3] Le, Pham, Sahoo, Hoi — URLNet

- **서지:** Hung Le, Quang Pham, Doyen Sahoo, Steven C.H. Hoi. *URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection.* arXiv:1802.03162, 2018 (확인한 판: v2).
- **원문:** https://arxiv.org/abs/1802.03162
- **확인한 내용:**
  - 초록(PDF 1쪽): URL의 문자와 단어 양쪽에 CNN을 적용하는 end-to-end 모델을 제안한다.
  - 기존 bag-of-words 특징은 문자 순서를 반영하지 못하고 사람이 특징을 설계해야 한다고 지적한다.
- **우리 보고서에서 쓰는 곳:** 원 모델(문자 3-gram + GRU + 수치 특징)이 이 계열의 문자 단위 딥러닝 접근이라는 위치 설정. 이 판의 PDF에는 학회 게재 정보가 없으므로 arXiv로만 인용한다.

## 2. 확장(캐스케이드)의 근거

### [R4] Viola, Jones — 캐스케이드의 원형

- **서지:** Paul Viola, Michael Jones. *Rapid Object Detection using a Boosted Cascade of Simple Features.* IEEE CVPR 2001.
- **원문:** https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
- **확인한 내용:**
  - 초록: 점점 복잡해지는 분류기를 "cascade"로 묶는다. 배경 영역은 빨리 버리고, 대상이 있을 법한 영역에만 연산을 더 쓴다.
  - 서론(PDF 2쪽)에서는 앞단 필터가 탐지 대상의 거의 전부를 통과시켜야 하므로 미탐률이 중요하다고 설명한다.
  - 4절 *The Attentional Cascade*(PDF 4쪽)에서는 단순 분류기의 임계값을 조정해 미탐률을 0에 가깝게 유지하면서 대부분의 배경 영역을 먼저 걸러 낸다.
- **우리 보고서에서 쓰는 곳:** 캐스케이드 설계의 출처.
  - 역할 대응: 이 논문의 "버리는 배경"이 우리의 "조기 종료하는 정상 URL"이다. 앞단이 놓치면 안 되는 대상은 악성 URL이다.
  - 따라서 "추가 미탐 1%p 이내"라는 우리 성공 기준은 이 논문이 말하는 앞단 거짓 음성 제약과 같은 성격이다.

## 3. 데이터 문제와 평가 한계의 근거

### [R5] Arp 외 — 보안 분야 머신러닝의 함정 (가장 중요)

- **서지:** Daniel Arp, Erwin Quiring, Feargus Pendlebury, Alexander Warnecke, Fabio Pierazzi, Christian Wressnegger, Lorenzo Cavallaro, Konrad Rieck. *Dos and Don'ts of Machine Learning in Computer Security.* 31st USENIX Security Symposium, 2022.
- **원문:** https://www.usenix.org/system/files/sec22-arp.pdf
- **확인한 내용 (2절 "Pitfalls in Machine Learning"):** 보안 머신러닝 논문에서 흔한 함정 10개를 정리했다. 그중 우리와 직접 관련된 것은 아래와 같다.

| 함정 | 원문 요지 | 우리 프로젝트에서 대응되는 사실 |
|---|---|---|
| P2 Label Inaccuracy | 정답 라벨이 부정확하거나 불안정하면 시스템 성능 평가 전체가 흔들린다 | 세 원파일에서 같은 URL에 서로 다른 라벨이 붙은 경우(PR #3 보고서 기준 96,007개 URL) |
| P3 Data Snooping | 시험 집합을 최종 평가 이전 실험에 쓰면(test snooping), 실제로는 없는 정보가 학습 설정에 들어간다 | PR #3의 이전 5/5/5 실행에서 같은 시험 집합 결과를 이미 확인함 |
| P4 Spurious Correlations | 보안 문제와 무관한 인공물이 클래스를 가르는 지름길이 되어, 모델이 과제 대신 인공물을 학습한다 | URL의 scheme 표기 유무가 수집 출처를 구별하는 단서일 가능성 |
| P6 Inappropriate Baseline | 기준선 방법이 없거나 적으면 기존 방법 대비 개선을 보일 수 없다. 거의 같은 모델끼리만 비교하는 것도 맥락이 부족하다 | PR #3의 URL 길이 기준선·경량 모델 단독 비교가 이에 대한 대응 |

- **우리 보고서에서 쓰는 곳:**
  - 데이터 정제 절: 왜 정제가 필요했는지의 근거(P2).
  - 한계 절: 공개한 제한 사항(P3, P4)을 이 분류에 맞춰 정리한다. "알고 있는 함정을 확인하고 공개했다"는 형태로 쓴다.

### [R6] Northcutt, Athalye, Mueller — 시험 집합의 라벨 오류

- **서지:** Curtis G. Northcutt, Anish Athalye, Jonas Mueller. *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.* NeurIPS 2021 Track on Datasets and Benchmarks. arXiv:2103.14749.
- **원문:** https://arxiv.org/abs/2103.14749 (확인한 판: v4)
- **확인한 내용 (초록과 그림 4):**
  - 널리 쓰이는 10개 데이터셋의 시험 집합에 평균 최소 3.3%의 라벨 오류가 있다.
  - 그림 4(PDF 8쪽)에서 원래 라벨이 틀렸던 표본의 비중을 높이자, 수정한 라벨로 측정한 ImageNet 정확도 순위가 바뀌었다. 원래 비중은 2.9%였고, 약 9%에서 ResNet-18과 ResNet-50의 정확도가 교차했다.
- **우리 보고서에서 쓰는 곳:**
  - 원 데이터에서 실험 A의 98.40%는 오류가 섞인 라벨 기준의 점수다. 따라서 탐지 성능의 근거가 아니라 원 결과 재현의 근거로만 쓴다는 설명의 근거.
  - 정제 후 GRU와 경량 모델의 차이가 작아진 현상을 해석할 때의 참고.

## 보고서 반영 제안

| 보고서 절 | 넣을 인용 |
|---|---|
| Introduction & Motivation | R1, R2 |
| Original Work Summary | R3 (원 모델의 위치) |
| Experiments (확장 설계) | R4 |
| 데이터 정제 | R5-P2, R6 |
| Limitations & Future Work | R5-P3·P4, R1(개념 변화) |
