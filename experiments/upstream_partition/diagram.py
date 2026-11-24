"""Draw the source-to-decision paths with separate team and upstream stages."""

from PIL import Image, ImageDraw, ImageFont


def draw(path, regular, bold):
    image = Image.new("RGB", (1540, 1100), "white")
    canvas = ImageDraw.Draw(image)
    navy, blue, green, ink = "#19466B", "#EAF2FA", "#E6F3EF", "#172C3A"

    def box(rect, title, body, fill, edge):
        canvas.rounded_rectangle(rect, 18, fill=fill, outline=edge, width=3)
        x = (rect[0] + rect[2]) // 2
        y = (rect[1] + rect[3]) // 2
        canvas.text(
            (x, y - 18),
            title,
            font=ImageFont.truetype(str(bold), 31),
            anchor="mm",
            fill=ink,
        )
        canvas.text(
            (x, y + 26),
            body,
            font=ImageFont.truetype(str(regular), 24),
            anchor="mm",
            fill=ink,
        )

    def arrow(points):
        canvas.line(points, fill=navy, width=5)
        (x0, y0), (x, y) = points[-2:]
        dx, dy = x - x0, y - y0
        d = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / d, dy / d
        canvas.polygon(
            [
                (x, y),
                (x - 18 * dx + 9 * dy, y - 18 * dy - 9 * dx),
                (x - 18 * dx - 9 * dy, y - 18 * dy + 9 * dx),
            ],
            fill=navy,
        )

    box(
        (370, 15, 1170, 112),
        "원본 URL → 수치 특징 18개",
        "길이·기호·경로 등을 문자 전처리 전에 계산",
        blue,
        navy,
    )
    arrow([(770, 112), (770, 155)])
    box(
        (315, 156, 1225, 258),
        "팀의 1단계  ·  로지스틱 회귀 또는 트리 부스팅",
        "18개 수치로 악성 확률을 계산하고 검증 집합에서 정한 임계값과 비교",
        green,
        "#377565",
    )
    arrow([(540, 258), (540, 285), (332, 285), (332, 325)])
    arrow([(1000, 258), (1000, 285), (1174, 285), (1174, 325)])
    box(
        (40, 326, 625, 430),
        "확률 < 임계값",
        "정상으로 조기 종료할 후보",
        green,
        "#377565",
    )
    box(
        (865, 326, 1485, 430),
        "확률 ≥ 임계값",
        "모두 GRU 검사 대상으로 선택",
        green,
        "#377565",
    )
    arrow([(332, 430), (332, 478)])
    box(
        (40, 479, 625, 589),
        "재검사 정책",
        "기본 0건 / 무작위 또는 경계 5%",
        green,
        "#377565",
    )
    arrow([(202, 589), (202, 688)])
    box((15, 690, 395, 792), "선택되지 않은 후보", "여기서 정상 판정", green, "#377565")
    arrow([(485, 589), (485, 621), (1000, 621), (1000, 662)])
    arrow([(1174, 430), (1174, 662)])
    box(
        (590, 663, 1490, 753),
        "팀의 2단계  ·  원저자 GRU 모델",
        "재검사에 뽑힌 URL과 임계값 이상인 URL을 검사",
        blue,
        navy,
    )
    arrow([(806, 753), (806, 792)])
    arrow([(1260, 753), (1260, 792)])
    box(
        (543, 793, 1056, 900),
        "문자 분기",
        "원문 정제 → 3그램 50칸 → 임베딩 → GRU",
        blue,
        navy,
    )
    box((1075, 793, 1528, 900), "수치 분기", "앞서 계산한 18개 값 → 밀집층", blue, navy)
    arrow([(806, 900), (806, 942), (961, 942), (961, 978)])
    arrow([(1260, 900), (1260, 942), (1125, 942), (1125, 978)])
    box(
        (616, 979, 1483, 1085),
        "두 분기 결합 → 정상 확률",
        "0.5 이상 정상 / 0.5 미만 악성",
        blue,
        navy,
    )
    image.save(path)
