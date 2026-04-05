"""Expand a short movie idea into a synopsis using Claude."""

import json
import os

import anthropic

VIBE_CATEGORIES = {
    "😂 Laughs": {
        "low": "Waiting for your Hot Pocket to heat up",
        "high": "Laughing so hard you shart a little",
    },
    "😢 Tears": {
        "low": "Bone dry, not a tear in sight",
        "high": "Ugly crying in public",
    },
    "💓 Romance": {
        "low": "Awkward handshake",
        "high": "Steamy enough to fog your screen",
    },
    "😱 Scares": {
        "low": "Night light off, no big deal",
        "high": "Sleeping with every light on for a week",
    },
    "🔥 Thrills": {
        "low": "Sunday afternoon nap",
        "high": "White-knuckle grip on your seat",
    },
}


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


def expand_plot_to_screenplay(plot_summary: str, api_key: str | None = None) -> str:
    """Use Claude to expand a short movie idea into a ~2000-word screenplay.

    This generates screenplay-format text used internally for feature extraction.
    It is NOT shown to the user.

    Parameters
    ----------
    plot_summary : str
        The user's movie idea.
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
        "- Character names in ALL CAPS, centered (indented with at least 10 spaces) before their dialogue\n"
        "- Dialogue indented beneath character names (indented with at least 10 spaces)\n"
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


def rate_vibes(synopsis: str, api_key: str | None = None) -> dict[str, int]:
    """Have Claude rate a movie synopsis on fun vibe categories (0-100).

    Parameters
    ----------
    synopsis : str
        The movie synopsis text.
    api_key : str, optional
        Anthropic API key.

    Returns
    -------
    dict mapping vibe category names to scores (0-100).
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No Anthropic API key provided.")

    client = anthropic.Anthropic(api_key=key)

    categories_str = ", ".join(VIBE_CATEGORIES.keys())

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=(
            "You are a movie vibe analyst. Rate the following movie synopsis on each category "
            "from 0 to 100. Respond ONLY with a JSON object mapping category name to integer score.\n\n"
            f"Categories: {categories_str}\n\n"
            'Example response: {{"😂 Laughs": 75, "😢 Tears": 20, ...}}'
        ),
        messages=[
            {"role": "user", "content": synopsis}
        ],
    )

    try:
        return json.loads(message.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return {cat: 50 for cat in VIBE_CATEGORIES}
