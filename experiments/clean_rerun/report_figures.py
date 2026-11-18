"""Vector figures for the Korean technical report (ReportLab, no generated imagery)."""

from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.lib import colors

INK = colors.HexColor("#17334F")
PALE = colors.HexColor("#EEF3F7")
GRAY = colors.HexColor("#68737D")


def box(d, x, y, w, h, title, detail="", fill=PALE):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=INK, strokeWidth=0.7))
    d.add(
        String(
            x + w / 2,
            y + h / 2 + (4 if detail else -3),
            title,
            fontName="KoreanBold",
            fontSize=8.3,
            textAnchor="middle",
            fillColor=INK,
        )
    )
    if detail:
        d.add(
            String(
                x + w / 2,
                y + h / 2 - 10,
                detail,
                fontName="Korean",
                fontSize=7.2,
                textAnchor="middle",
                fillColor=GRAY,
            )
        )


def arrow(d, points, label=None, at=None):
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        d.add(Line(x1, y1, x2, y2, strokeColor=INK, strokeWidth=0.8))
    x, y = points[-1]
    px, py = points[-2]
    if x == px:
        sign = 1 if y > py else -1
        tri = [x, y, x - 3, y - sign * 6, x + 3, y - sign * 6]
    else:
        sign = 1 if x > px else -1
        tri = [x, y, x - sign * 6, y - 3, x - sign * 6, y + 3]
    d.add(Polygon(tri, fillColor=INK, strokeColor=INK))
    if label:
        xx, yy = at
        d.add(String(xx, yy, label, fontName="Korean", fontSize=7.4, textAnchor="middle"))


def cascade():
    d = Drawing(396, 300)
    box(d, 3, 250, 108, 42, "정규화 URL", "오프라인 문자열 입력")
    box(d, 144, 250, 108, 42, "18개 수치 특징", "공통 특징 추출")
    box(d, 285, 250, 108, 42, "1단계 경량 모델", "LR 또는 트리 + 확률 보정")
    arrow(d, [(111, 271), (144, 271)])
    arrow(d, [(252, 271), (285, 271)])
    box(d, 285, 180, 108, 42, "악성 점수 p₁", "검증에서 고정한 임계값 τ")
    arrow(d, [(339, 250), (339, 222)])
    box(d, 144, 180, 108, 42, "조기 종료 후보", "p₁ < τ")
    arrow(d, [(285, 201), (252, 201)])
    box(d, 144, 105, 108, 42, "재검사 후보 선택", "같은 수의 무작위 또는 경계")
    arrow(d, [(198, 180), (198, 147)])
    box(d, 3, 105, 108, 42, "정상으로 종료", "2단계 호출 없음")
    arrow(d, [(144, 126), (111, 126)], "미선택", (127, 133))
    box(d, 285, 105, 108, 42, "2단계 GRU 결합 모델", "선택된 URL만 실행")
    arrow(d, [(339, 180), (339, 147)], "p₁ ≥ τ", (366, 161))
    arrow(d, [(252, 126), (285, 126)], "선택", (268, 133))
    box(d, 285, 30, 108, 42, "정상 또는 악성", "정상 점수 ≥ 0.5이면 정상")
    arrow(d, [(339, 105), (339, 72)])
    d.add(String(3, 68, "기본 정책: 재검사 비율 0%", fontName="Korean", fontSize=8))
    d.add(String(3, 49, "주 비교: 조기 종료 후보의 5% 재검사", fontName="Korean", fontSize=8))
    d.add(String(3, 30, "LR과 트리는 서로 다른 비교 구성에서 사용", fontName="Korean", fontSize=8))
    return d


def gru():
    d = Drawing(396, 340)
    box(d, 15, 287, 160, 43, "문자 3-gram 입력", "최대 50 token / 사전 10,000")
    box(d, 221, 287, 160, 43, "수치 특징 입력", "18개 특징 / 원래 척도")
    box(d, 15, 212, 160, 43, "Embedding", "10,000 × 16")
    arrow(d, [(95, 287), (95, 255)])
    box(d, 15, 137, 160, 43, "GRU 16", "dropout 0.2 / recurrent 0.2")
    arrow(d, [(95, 212), (95, 180)])
    box(d, 221, 137, 160, 43, "Dense 16 + ReLU", "dropout 0.2")
    arrow(d, [(301, 287), (301, 180)])
    box(d, 132, 67, 132, 43, "결합 32 → Dense 16", "ReLU + dropout 0.2")
    arrow(d, [(95, 137), (95, 88), (132, 88)])
    arrow(d, [(301, 137), (301, 88), (264, 88)])
    box(d, 132, 0, 132, 39, "Dense 1 + Sigmoid", "정상일 점수 출력")
    arrow(d, [(198, 67), (198, 39)])
    return d


def timing(summary, labels):
    d = Drawing(396, 210)
    maxval = max(v["amortized_ms_per_url"] for v in summary.values()) * 1.1
    for i, (key, label) in enumerate(labels.items()):
        y = 185 - i * 26
        v = summary[key]["amortized_ms_per_url"]
        width = 235 * v / maxval
        d.add(String(0, y + 4, label, fontName="Korean", fontSize=8.3))
        d.add(
            Rect(
                96,
                y,
                width,
                15,
                fillColor=INK if key == "gru" else colors.HexColor("#6A94B5"),
                strokeColor=None,
            )
        )
        d.add(String(100 + width, y + 4, f"{v:.4f}", fontName="Korean", fontSize=8))
    d.add(
        String(96, 0, "배치 처리 시간 / URL (ms), 5회 측정 중앙값", fontName="Korean", fontSize=8)
    )
    return d


def recall_curve(results, family):
    rows = results["sensitivity"][family]
    d = Drawing(396, 210)
    tau = results["policies"][family + "_basic"]["tau"]
    subset = [r for r in rows if r["tau"] == tau]
    target = results["gru"]["recall"] * 100 - 1
    lo = min(min(r["recall"] for r in subset) * 100, target) - 0.1
    hi = max(r["recall"] for r in subset) * 100 + 0.1
    if hi - lo < 0.5:
        hi = lo + 0.5
    x0, y0, w, h = 45, 38, 325, 145
    d.add(Line(x0, y0, x0 + w, y0, strokeColor=GRAY))
    d.add(Line(x0, y0, x0, y0 + h, strokeColor=GRAY))
    for rate in (0, 0.05, 0.1, 0.2):
        xx = x0 + w * rate / 0.2
        d.add(
            String(xx, 20, f"{rate * 100:g}%", fontName="Korean", fontSize=8, textAnchor="middle")
        )
    for v in (lo, (lo + hi) / 2, hi):
        yy = y0 + (v - lo) / (hi - lo) * h
        d.add(String(x0 - 7, yy - 3, f"{v:.2f}", fontName="Korean", fontSize=8, textAnchor="end"))
    target_y = y0 + (target - lo) / (hi - lo) * h
    target_color = colors.HexColor("#A05F45")
    d.add(
        Line(
            x0,
            target_y,
            x0 + w,
            target_y,
            strokeColor=target_color,
            strokeDashArray=[3, 3],
            strokeWidth=0.8,
        )
    )
    d.add(
        String(
            x0 + w,
            target_y - 10,
            "GRU 대비 1%p 감소 기준",
            fontName="Korean",
            fontSize=7,
            textAnchor="end",
            fillColor=target_color,
        )
    )
    for policy, color, label in [
        ("random", colors.HexColor("#7C8A96"), "무작위"),
        ("boundary", INK, "경계 점수"),
    ]:
        points = []
        for r in sorted(
            [r for r in subset if r["policy"] in (policy, "basic")], key=lambda r: r["rate"]
        ):
            points.append((x0 + w * r["rate"] / 0.2, y0 + (r["recall"] * 100 - lo) / (hi - lo) * h))
        for a, b in zip(points, points[1:]):
            d.add(Line(*a, *b, strokeColor=color, strokeWidth=1.4))
        for a, b in points:
            d.add(Circle(a, b, 2.3, fillColor=color, strokeColor=color))
        d.add(
            String(
                245 if policy == "random" else 315,
                199,
                label,
                fontName="Korean",
                fontSize=8,
                fillColor=color,
            )
        )
    d.add(String(0, 199, "악성 recall (%)", fontName="Korean", fontSize=8))
    d.add(String(180, 1, "재검사 비율", fontName="Korean", fontSize=8, textAnchor="middle"))
    return d


def first_stage_models():
    d = Drawing(396, 210)
    d.add(String(0, 195, "로지스틱 회귀: 선형 점수", fontName="KoreanBold", fontSize=9))
    box(d, 0, 137, 108, 40, "18개 특징 표준화", "학습 집합에서 평균·표준편차")
    box(d, 144, 137, 108, 40, "가중 합 + 절편", "z = Σ(w·x) + b")
    box(d, 288, 137, 108, 40, "sigmoid 보정", "공통 검증 15%로 적합")
    arrow(d, [(108, 157), (144, 157)])
    arrow(d, [(252, 157), (288, 157)])
    d.add(String(0, 104, "트리 부스팅: 비선형 분기의 합", fontName="KoreanBold", fontSize=9))
    box(d, 0, 46, 108, 40, "18개 수치 특징", "표준화 없이 사용")
    box(d, 144, 46, 108, 40, "초기 점수 + 60개 트리", "손실을 줄이는 순차 학습")
    box(d, 288, 46, 108, 40, "sigmoid 보정", "공통 검증 15%로 적합")
    arrow(d, [(108, 66), (144, 66)])
    arrow(d, [(252, 66), (288, 66)])
    d.add(
        String(
            0,
            8,
            "p₁은 악성 점수이며, 임계값·재검사 정책을 거쳐 GRU 실행 여부를 결정한다.",
            fontName="Korean",
            fontSize=8,
        )
    )
    return d


def training_size_curve(rows):
    d = Drawing(396, 220)
    x0, y0, w, h = 49, 42, 321, 145
    values = [r[k] * 100 for r in rows for k in ("train_ap", "validation_ap")]
    lo = min(values) - 0.03
    hi = max(values) + 0.03
    if hi - lo < 0.15:
        hi = lo + 0.15
    maxrows = rows[-1]["training_rows"]
    d.add(Line(x0, y0, x0 + w, y0, strokeColor=GRAY))
    d.add(Line(x0, y0, x0, y0 + h, strokeColor=GRAY))
    for value in (lo, (lo + hi) / 2, hi):
        yy = y0 + (value - lo) / (hi - lo) * h
        d.add(
            String(x0 - 6, yy - 3, f"{value:.2f}", fontName="Korean", fontSize=8, textAnchor="end")
        )
    for row in rows:
        xx = x0 + w * row["training_rows"] / maxrows
        d.add(
            String(
                xx,
                25,
                f"{row['training_rows'] / 1000:.0f}천",
                fontName="Korean",
                fontSize=8,
                textAnchor="middle",
            )
        )
    for key, color, label, lx in [
        ("train_ap", GRAY, "학습", 250),
        ("validation_ap", INK, "공통 검증", 305),
    ]:
        points = [
            (x0 + w * r["training_rows"] / maxrows, y0 + (r[key] * 100 - lo) / (hi - lo) * h)
            for r in rows
        ]
        for a, b in zip(points, points[1:]):
            d.add(Line(*a, *b, strokeColor=color, strokeWidth=1.4))
        for x, y in points:
            d.add(Circle(x, y, 2.3, fillColor=color, strokeColor=color))
        d.add(String(lx, 204, label, fontName="Korean", fontSize=8, fillColor=color))
    d.add(String(0, 204, "악성 AP (%)", fontName="Korean", fontSize=8))
    d.add(
        String(
            196, 3, "실제 학습 데이터 수 (행)", fontName="Korean", fontSize=8, textAnchor="middle"
        )
    )
    return d
