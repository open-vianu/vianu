from openai import OpenAI

import logging

logger = logging.getLogger("fraudcrawler_logger")


def compare_drugs_with_gpt(token, drug_name, german_description, swiss_description):
    """
    Compare drug descriptions from Germany and Switzerland using GPT-4.

    Args:
        token (str): The OpenAI API token.
        drug_name (str): Name of the drug being compared.
        german_description (str): Drug description from Germany.
        swiss_description (str): Drug description from Switzerland.

    Returns:
        str: A Markdown string representing the comparison table.
    """
    # Refined prompt to prevent hallucinations
    prompt = f"""
    You are a medical expert helping to compare drug descriptions across countries on the level of MedDRAs System Organ Class (SOC).
    MedDRA categorizes adverse events into 27 high-level organ system groups for uniform reporting, such as “Cardiac disorders,” “Gastrointestinal disorders,” and “Nervous system disorders.” Use only SOC that are existing, do not invent new ones.

    Please create a comparison table in Markdown format for the drug '{drug_name}' based on the following descriptions:
    
    **Germany:**
    {german_description}
    
    **Switzerland:**
    {swiss_description}
    
    The table should include:
    - A column for the SOC.
    - A column for the mention in Germany's description.
    - A column for the mention in Switzerland's description.

    The first row should contain headers: "Affected SOC", "Germany", "Switzerland".
    Use only the exact terms and phrases mentioned in the descriptions. Do not infer, interpret, or hallucinate additional information.
    """
    try:
        client = OpenAI(api_key=token)
        logger.info(prompt)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        # Extract the content from the response
        markdown_table = response.choices[0].message.content
        markdown_table = markdown_table.replace("```markdown", "")
        markdown_table = markdown_table.replace("```", "")
        logger.info(markdown_table)
        return markdown_table
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"
