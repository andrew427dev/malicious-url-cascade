"""Present each measured result in the team report's readable 3–5 column format."""

from dataclasses import dataclass
import re


@dataclass
class ReportTable:
    caption: str
    headers: list
    rows: list
    widths: tuple


# Each column is an original column index or a tuple of identifying fields.
# Widths are centimetres on the reference report's 16-cm text area.
LAYOUTS = {
    "파일": [("", (0, 1, 2, 3), (6.1, 2.4, 2.4, 5.1))],
    "학습 블록": [("", (0, 1, 2, 3, 4), (2.1, 5.4, 2.9, 2.8, 2.8))],
    "단계": [("", (0, 1, 2), (9.3, 3.1, 3.6))],
    "분할|역할": [("", (0, 1, 2, 3, 4), (3.2, 2.7, 3.5, 3.3, 3.3))],
    "순서": [("", (0, 1, 2), (1.35, 4.25, 10.4))],
    "분할|집합 쌍": [("", (0, 1, 2, 3, 4), (2.3, 3.5, 3.0, 3.8, 3.4))],
    "분할|표기|GRU 회차": [
        ("로지스틱 회귀", ((0, 1), 2, 3, 5), (5.5, 3.4, 3.7, 3.4)),
        ("히스토그램 기반 그래디언트 부스팅", ((0, 1), 2, 4, 6), (5.5, 3.4, 3.7, 3.4)),
    ],
    "평가 집합": [("", (0, 1, 2, 3), (3.8, 5.2, 3.4, 3.6))],
    "구성|정확도 %": [
        ("탐지 성능", (0, 1, 2, 3), (5.4, 3.4, 3.8, 3.4)),
        ("미탐과 GRU 호출", (0, 4, 5, 6, 7), (4.2, 2.6, 3.2, 3.0, 3.0)),
    ],
    "구성|원래 재현율 %": [
        ("악성 재현율", (0, 1, 2), (5.4, 5.3, 5.3)),
        ("판정이 바뀐 URL 수", (0, 3, 4, 5, 6), (3.9, 2.6, 3.1, 3.1, 3.3)),
    ],
    "구성|GRU 건": [
        ("처리 시간", (0, 3, 4, 5), (4.9, 3.4, 4.1, 3.6)),
        ("GRU 호출", (0, 1, 2), (6.0, 5.0, 5.0)),
    ],
    "분할|표기|모델|정확도 %": [
        ("탐지율", ((0, 1), 2, 3, 4, 5), (4.1, 2.1, 3.1, 3.6, 3.1)),
        ("미탐과 평균 정밀도", ((0, 1), 2, 6, 7), (4.9, 3.1, 3.6, 4.4)),
    ],
    "분할|모델": [
        ("원래 표기", (0, 1, 2, 3, 4), (3.2, 2.4, 3.5, 3.6, 3.3)),
        ("재학습 후 평균 정밀도", (0, 1, 5, 6), (3.6, 2.8, 4.3, 5.3)),
    ],
    "분할|겹침 행": [("", (0, 1, 2, 3, 4), (3.2, 3.2, 3.2, 3.2, 3.2))],
    "학습 비율": [("", (0, 1, 2, 3), (4.4, 4.2, 3.7, 3.7))],
    "분할|표기|모델|재검사 건": [
        ("", ((0, 1, 2), 3, 4, 5, 6), (4.9, 2.7, 2.6, 2.9, 2.9))
    ],
    "재검사 비율": [("", (0, 1, 2, 3, 4), (3.1, 3.0, 3.6, 3.1, 3.2))],
    "분할|표기|구성": [("", ((0, 1), 2, 3, 4, 5), (3.7, 3.5, 2.6, 3.1, 3.1))],
}

LABELS = {
    "재현율 %": "악성 재현율",
    "미탐": "미탐 수",
    "미탐 건": "미탐 수",
    "GRU 건": "GRU 호출 수",
    "호출 %": "GRU 호출률",
    "호출 절감 %": "호출 감소율",
    "시간 단축 %": "시간 단축률",
    "추가 미탐 건": "추가 미탐",
    "중앙값 ms": "중앙값 (ms)",
    "5회 범위 ms": "5회 범위 (ms)",
    "절약 ms": "절약 시간 (ms)",
    "GRU 호출 %": "GRU 호출률",
    "이번 단계 제외": "제외한 행",
    "원래 재현율 %": "원래 표기",
    "제거 재현율 %": "접두어 제거",
}


def present(caption, heads, rows):
    names = "|".join(heads) + "|"
    key = next(k for k in LAYOUTS if names.startswith(k + "|"))
    specs = LAYOUTS[key]
    covered = set()
    result = []
    for i, (topic, columns, widths) in enumerate(specs, 1):
        blocks = [c if isinstance(c, tuple) else (c,) for c in columns]
        covered.update(j for block in blocks for j in block)
        headers, values = [], []
        for block in blocks:
            if len(block) > 1:
                headers.append(" · ".join(heads[j] for j in block))
            else:
                name = heads[block[0]]
                headers.append(
                    LABELS.get(name, name.replace(" (%)", "").removesuffix(" %"))
                )
        for row in rows:
            cells = []
            for block in blocks:
                value = " · ".join(str(row[j]) for j in block)
                if len(block) == 1 and heads[block[0]] == "구성":
                    value = re.sub(r"^(LR|HGB) ", r"\1 + GRU ", value)
                if (
                    len(block) == 1
                    and "%" in heads[block[0]]
                    and value != "—"
                    and not value.endswith("%")
                ):
                    value += "%"
                cells.append(value)
            values.append(cells)
        label = caption
        if topic:
            label = re.sub(r"^표 (\d+)\.", lambda m: f"표 {m[1]}-{i}.", caption)
            label += " · " + topic
        assert len(widths) == len(columns) and abs(sum(widths) - 16) < 0.01
        result.append(ReportTable(label, headers, values, widths))
    assert covered == set(range(len(heads))), (caption, covered)
    return result
