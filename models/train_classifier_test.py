from train_classifier import tokenize


def test_SHOULD_tokenize_text_WHEN_called():
    # GIVEN: An example input string.
    example_input = "This is a test. It should tokenize this sentence."

    # WHEN: The tokenize function is called with the example input.
    tokens = tokenize(example_input)

    # THEN: The tokens should be as expected.
    expected_tokens = ['test', 'tokenize', 'sentence']
    assert tokens == expected_tokens
