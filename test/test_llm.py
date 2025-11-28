from src.llm_wrapper import _template


def test_llm_template_fallback_output():
    predicted_label = "COVID"
    probs = {"COVID": 0.91, "Normal": 0.05, "Viral": 0.04}

    explanation = _template(predicted_label, probs)

    assert isinstance(explanation, str)
    assert "COVID" in explanation
    assert len(explanation) > 40

