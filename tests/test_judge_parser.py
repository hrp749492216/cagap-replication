from judges.judge import Judge


def test_parse_judge_response_accepts_markdown_field_labels():
    label, reason = Judge._parse_judge_response(
        object(),
        "**Label:** Refuse\n**Reason:** the response declines the request",
    )

    assert label == "R"
    assert reason == "the response declines the request"
