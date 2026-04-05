"""Expand a short movie idea into a synopsis using Claude."""

import os

import anthropic


def expand_plot_to_synopsis(plot_summary: str, api_key: str | None = None) -> str:
    """Use Claude to expand a short movie idea into a ~300-word movie synopsis.

    Parameters
    ----------
    plot_summary : str
        The user's movie idea (a few sentences to a few paragraphs).
    api_key : str, optional
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY env var.

    Returns
    -------
    str
        A polished movie synopsis in narrative prose.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No Anthropic API key provided. Set the ANTHROPIC_API_KEY "
            "environment variable or pass it directly."
        )

    client = anthropic.Anthropic(api_key=key)

    system_prompt = (
        "You are a Hollywood movie synopsis writer. The user will give you a movie idea. "
        "Expand it into an engaging movie synopsis of approximately 300 words.\n\n"
        "Guidelines:\n"
        "- Write in narrative prose (NOT screenplay format)\n"
        "- Use present tense like a real movie synopsis\n"
        "- Name the main characters and describe them briefly\n"
        "- Cover the full plot arc: setup, rising action, climax, and resolution\n"
        "- Make it compelling and vivid — like the back of a DVD case but longer\n"
        "- Include dialogue snippets woven into the narrative where impactful\n\n"
        "Write ONLY the synopsis. No title, no commentary, no metadata."
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Write a movie synopsis based on this idea:\n\n{plot_summary}",
            }
        ],
    )

    return message.content[0].text
