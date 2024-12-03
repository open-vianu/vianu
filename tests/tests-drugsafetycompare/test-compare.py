# tests/tests-drugsafetycompare/test-compare.py

from unittest.mock import patch, MagicMock
from vianu.drugsafetycompare.src.compare import compare_drugs_with_gpt


@patch("vianu.drugsafetycompare.src.compare.OpenAI")
def test_compare_drugs_with_gpt_success(mock_openai):
    # Mock OpenAI response
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """```markdown
| Affected SOC                | Germany                 | Switzerland            |
|-----------------------------|-------------------------|------------------------|
| Cardiovascular disorders    | Hypertension, Arrhythmia| Tachycardia            |
| Gastrointestinal disorders  | Nausea                  | Vomiting, Diarrhea     |
```"""
    mock_client_instance.chat.completions.create.return_value = mock_response

    # Define test inputs
    token = "test_token"
    drug_name = "Aspirin"
    german_description = "Hypertension and Arrhythmia are common."
    swiss_description = "Tachycardia is observed."

    # Call the function
    result = compare_drugs_with_gpt(
        token, drug_name, german_description, swiss_description
    )

    # Define expected output
    expected_output = """| Affected SOC                | Germany                 | Switzerland            |
|-----------------------------|-------------------------|------------------------|
| Cardiovascular disorders    | Hypertension, Arrhythmia| Tachycardia            |
| Gastrointestinal disorders  | Nausea                  | Vomiting, Diarrhea     |"""

    # Assert the result matches expected output
    assert result.strip() == expected_output.strip()

    # Ensure OpenAI was called with correct parameters
    mock_openai.assert_called_with(api_key=token)
    mock_client_instance.chat.completions.create.assert_called_once()


@patch("vianu.drugsafetycompare.src.compare.OpenAI")
def test_compare_drugs_with_gpt_api_exception(mock_openai):
    # Mock OpenAI to raise an exception
    mock_client_instance = MagicMock()
    mock_openai.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.side_effect = Exception("API Error")

    # Define test inputs
    token = "test_token"
    drug_name = "Aspirin"
    german_description = "Hypertension and Arrhythmia are common."
    swiss_description = "Tachycardia is observed."

    # Call the function
    result = compare_drugs_with_gpt(
        token, drug_name, german_description, swiss_description
    )

    # Define expected output
    expected_output = "Error calling OpenAI API: API Error"

    # Assert the result matches expected output
    assert result == expected_output

    # Ensure OpenAI was called with correct parameters
    mock_openai.assert_called_with(api_key=token)
    mock_client_instance.chat.completions.create.assert_called_once()
