"""Expand a short movie idea into screenplay-format text using Claude."""

import os

import anthropic


def expand_plot_to_script(plot_summary: str, api_key: str | None = None) -> str:
    """Use Claude to expand a short movie idea into a ~2000-word screenplay.

    Parameters
    ----------
    plot_summary : str
        The user's movie idea (a few sentences to a few paragraphs).
    api_key : str, optional
        Anthropic API key. Falls back to the ANTHROPIC_API_KEY env var.

    Returns
    -------
    str
        Screenplay-format text with scene headings, dialogue, and action lines.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No Anthropic API key provided. Set the ANTHROPIC_API_KEY "
            "environment variable or pass it directly."
        )

    client = anthropic.Anthropic(api_key=key)

    system_prompt = (
        "You are a professional screenwriter. The user will give you a movie idea. "
        "Expand it into a screenplay excerpt of approximately 2000 words.\n\n"
        "Follow standard screenplay formatting:\n"
        "- Scene headings in ALL CAPS starting with INT. or EXT. (e.g., INT. CLASSROOM - DAY)\n"
        "- Character names in ALL CAPS, centered (indented with spaces) before their dialogue\n"
        "- Dialogue indented beneath character names\n"
        "- Action/description lines in normal case\n"
        "- Include at least 10 scenes with varied locations\n"
        "- Include dialogue between multiple characters\n\n"
        "Write ONLY the screenplay text. No commentary, no title page, no notes."
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Write a screenplay based on this idea:\n\n{plot_summary}",
            }
        ],
    )

    return message.content[0].text
