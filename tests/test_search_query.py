"""Cleaning the model's reply into a search query. Pure Python."""

from adaptive_tutor.research.search_query import MAX_QUERY_CHARS, clean_search_query, shorten


def test_plain_query_is_kept():
    assert clean_search_query("mitochondria energy production", "q") == "mitochondria energy production"


def test_quotes_and_labels_are_removed():
    assert clean_search_query('Search query: "mitochondria energy"', "q") == "mitochondria energy"
    assert clean_search_query("`photosynthesis light reactions`", "q") == "photosynthesis light reactions"
    assert clean_search_query('"Query: vector database"', "q") == "vector database"


def test_intro_lines_and_list_markers_are_skipped():
    raw = "Here is a short search query:\n\n1. LangGraph conditional edges"
    assert clean_search_query(raw, "q") == "LangGraph conditional edges"


def test_empty_reply_falls_back_to_the_question():
    assert clean_search_query("   \n  ", "What is   RAG?") == "What is RAG?"


def test_long_text_is_cut_at_a_word_boundary():
    query = clean_search_query("word " * 50, "q")
    assert len(query) <= MAX_QUERY_CHARS
    assert not query.endswith(" ")
    assert set(query.split()) == {"word"}


def test_shorten_leaves_short_text_alone():
    assert shorten("short") == "short"