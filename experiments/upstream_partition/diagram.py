"""Draw the actual cascade in the two-panel grayscale design of the November 23 report."""

from PIL import Image, ImageDraw, ImageFont


def draw(path, regular, bold):
    scale = 2
    image = Image.new("RGB", (1250 * scale, 473 * scale), "white")
    canvas = ImageDraw.Draw(image)
    ink, dark, panel, line = "#131313", "#242525", "#F3F3F2", "#434343"

    def points(coords):
        return [tuple(round(v * scale) for v in p) for p in coords]

    def rectangle(rect, fill, border, radius=5):
        canvas.rounded_rectangle(
            tuple(int(x * scale) for x in rect),
            radius * scale,
            fill=fill,
            outline=border,
            width=scale,
        )

    def label(x, y, text, size, heavy=False, color=ink, anchor="mm"):
        font = ImageFont.truetype(str(bold if heavy else regular), round(size * scale))
        canvas.text((x * scale, y * scale), text, font=font, fill=color, anchor=anchor)

    def box(rect, title, lines, black=False):
        rectangle(rect, dark if black else "white", dark if black else "#777777")
        x = (rect[0] + rect[2]) / 2
        center = (rect[1] + rect[3]) / 2
        layout = [(title, 22, True)] + [(line, 17, False) for line in lines]
        first = center - (len(layout) - 1) * 12
        for i, (text, size, heavy) in enumerate(layout):
            label(x, first + i * 25, text, size, heavy, "white" if black else ink)

    def arrow(coords):
        canvas.line(points(coords), fill=line, width=2 * scale)
        (x0, y0), (x, y) = coords[-2:]
        dx, dy = x - x0, y - y0
        length = (dx**2 + dy**2) ** 0.5
        dx, dy = dx / length, dy / length
        canvas.polygon(
            points(
                [
                    (x, y),
                    (x - 9 * dx + 4 * dy, y - 9 * dy - 4 * dx),
                    (x - 9 * dx - 4 * dy, y - 9 * dy + 4 * dx),
                ]
            ),
            fill=line,
        )

    rectangle((5, 5, 531, 453), panel, "#AAAAAA")
    rectangle((629, 5, 1244, 453), "white", "#AAAAAA")
    label(23, 38, "1단계  |  팀이 추가한 경량 분류", 27, True, anchor="lm")
    label(653, 38, "2단계  |  원저자의 GRU 분류기", 27, True, anchor="lm")
    box(
        (40, 74, 493, 141),
        "원본 URL에서 수치 특징 18개 계산",
        ["문자 전처리 전의 길이·기호·경로 사용"],
    )
    arrow([(267, 141), (267, 161)])
    box(
        (40, 162, 493, 230),
        "로지스틱 회귀 또는 트리 부스팅",
        ["두 후보의 악성 확률을 각각 평가"],
        True,
    )
    arrow([(267, 230), (267, 250)])
    box(
        (40, 251, 493, 333),
        "GRU 검사 대상 선택",
        ["임계값 이상 전부 + 임계값 미만 중 재검사", "기본 0건 · 무작위 또는 경계 5%"],
        True,
    )
    arrow([(267, 333), (267, 376)])
    label(286, 352, "나머지 주소", 16, anchor="lm")
    box((84, 377, 450, 438), "정상으로 조기 종료", ["GRU 검사 생략"])
    label(560, 268, "검사 대상", 18, True)
    canvas.line(points([(493, 289), (680, 289)]), fill=line, width=2 * scale)
    arrow([(680, 289), (680, 147), (727, 147)])
    arrow([(680, 289), (680, 355), (727, 355)])
    box(
        (729, 100, 970, 194),
        "선택된 URL의 문자",
        ["정제 → 문자 3그램 50칸", "임베딩 → GRU"],
    )
    box((729, 310, 970, 402), "수치 특징 18개", ["앞서 계산한 값을", "밀집층에 입력"])
    canvas.line(
        points([(970, 147), (1004, 147), (1004, 355), (970, 355)]),
        fill=line,
        width=2 * scale,
    )
    arrow([(1004, 253), (1034, 253)])
    box(
        (1036, 199, 1227, 307),
        "두 출력 결합",
        ["정상 확률 ≥ 0.5 정상", "미만이면 악성"],
    )
    image.save(path)
