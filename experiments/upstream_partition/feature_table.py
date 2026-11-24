"""Korean System Setup table for the pinned FeatureCreation.create_features order."""

FEATURE_ROWS = [
    (
        "use_of_ip",
        "원문에서 원저자의 IP 정규식을 검색해 일치하면 1. 십진 IPv4 뒤의 /, 또는 16진 IPv4와 / 뒤에 IPv6 패턴이 연이어 오는 경우를 찾는다.",
    ),
    (
        "abnormal_url",
        "urlparse의 hostname을 문자열로 바꾸고, 이를 정규식으로 원문에서 검색해 일치하면 1. 호스트가 없으면 문자열 None을 검색한다.",
    ),
    ("count.", "원문에서 마침표(.)가 나온 횟수."),
    ("count-www", "원문에서 소문자 문자열 www가 나온 횟수."),
    ("count@", "원문에서 @가 나온 횟수."),
    ("count_dir", "urlparse로 얻은 경로(path)에서 /가 나온 횟수."),
    ("count_embed_domian", "경로에서 연속된 //가 겹치지 않고 나온 횟수."),
    ("count-https", "원문에서 소문자 문자열 https가 나온 횟수."),
    ("count-http", "원문에서 소문자 문자열 http가 나온 횟수. https 안의 http도 센다."),
    ("count%", "원문에서 %가 나온 횟수."),
    ("count?", "원문에서 물음표(?)가 나온 횟수."),
    ("count-", "원문에서 하이픈(-)이 나온 횟수."),
    ("count=", "원문에서 등호(=)가 나온 횟수."),
    ("url_length", "원본 URL 문자열의 글자 수."),
    (
        "hostname_length",
        "urlparse의 netloc 길이. 사용자 정보와 포트가 있으면 함께 세며, netloc가 비어 있으면 0.",
    ),
    ("count-digits", "원문의 각 문자에 str.isnumeric()을 적용해 참인 문자 수."),
    ("count-letters", "원문의 각 문자에 str.isalpha()를 적용해 참인 문자 수."),
    (
        "fd_length",
        "경로를 /로 나눈 결과의 두 번째 조각 길이. 해당 조각이 없거나 비어 있으면 0.",
    ),
]


def example(root):
    import json

    rec = json.loads((root / "feature-example.json").read_text())
    assert tuple(n for n, _ in FEATURE_ROWS) == tuple(rec["names"])
    assert rec["equal"] and rec["text"] == ["exampleorg", "exampleorg"]
    a, b = rec["values"]
    delta = [f"{n} {v}→{w}" for (n, _), v, w in zip(FEATURE_ROWS, a, b) if v != w]
    return (
        "https://example.org와 example.org는 모두 exampleorg가 되어 같은 학습 어휘에서 50칸 토큰 배열이 일치했다. "
        "접두어가 있는 표기에서 없는 표기로 바꾸면 "
        + ", ".join(delta)
        + "로 여섯 값이 달라졌다. "
        "예를 들어 파서는 접두어가 없는 example.org 전체를 경로로 읽어 netloc 길이를 0으로 계산한다."
    )
