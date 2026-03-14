from web_search.utils import build_search_prompt, classify_query_complexity


def test_classify_query_complexity_marks_simple_fact_query_as_direct():
    profile = classify_query_complexity("What is the capital of France?")

    assert profile.level == 1
    assert profile.mode == "direct"
    assert profile.query_type == "factual"
    assert profile.min_source_count == 1


def test_classify_query_complexity_marks_comparative_analysis_as_deep():
    profile = classify_query_complexity(
        "Compare Redis vs RabbitMQ for event-driven microservices architecture tradeoffs and best practices."
    )

    assert profile.level == 3
    assert profile.mode == "deep"
    assert profile.query_type == "comparative"
    assert profile.min_source_count == 4


def test_classify_query_complexity_marks_balanced_query_with_higher_quota():
    profile = classify_query_complexity(
        "What are the best options for deploying a small internal knowledge base with access control and low maintenance?"
    )

    assert profile.level == 2
    assert profile.mode == "balanced"
    assert profile.min_source_count == 3


def test_build_search_prompt_includes_complexity_specific_guidance():
    prompt = build_search_prompt(
        "请分析 RAG 与 fine-tuning 的区别、适用场景、成本权衡以及最佳实践"
    )

    assert "Complexity Level: 3 (deep)" in prompt
    assert "Minimum Source Target: 4" in prompt
    assert "Examine 4-6 relevant dimensions." in prompt
    assert "Do not continue searching just because more sources might exist." in prompt
    assert "Sources` section containing 4-7" in prompt
