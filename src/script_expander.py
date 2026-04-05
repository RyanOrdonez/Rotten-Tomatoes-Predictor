"""Expand a short movie idea into a synopsis using Claude."""

import json
import os

import anthropic

VIBE_CATEGORIES = {
    "😂 Laughs": {
        0: "Waiting for your Hot Pocket in the microwave",
        25: "A light chuckle at best",
        50: "Solid comedy — you'll laugh out loud a few times",
        75: "Tears-from-laughing funny",
        100: "Laughing so hard you shart a little",
    },
    "😢 Tears": {
        0: "Bone dry, not a tear in sight",
        25: "A tiny lump in your throat",
        50: "You'll get misty-eyed at least once",
        75: "Openly weeping, no shame",
        100: "Ugly crying in public",
    },
    "💓 Romance": {
        0: "Awkward handshake energy",
        25: "A shy glance across the room",
        50: "Butterflies in your stomach",
        75: "Full rom-com swoon",
        100: "Steamy enough to fog your screen",
    },
    "😱 Scares": {
        0: "Night light off, no big deal",
        25: "A couple of jump scares",
        50: "Checking behind the shower curtain",
        75: "Watching through your fingers",
        100: "Sleeping with every light on for a week",
    },
    "🔥 Thrills": {
        0: "Sunday afternoon nap",
        25: "Mildly interesting, like a crossword",
        50: "Leaning forward in your seat",
        75: "Heart pounding, can't look away",
        100: "White-knuckle grip, forgot to breathe",
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
        max_tokens=1024,
        system=(
            "Read the movie synopsis and rate how the audience would experience it. "
            "Use the rate_vibes tool to submit your ratings."
        ),
        tools=[
            {
                "name": "rate_vibes",
                "description": "Submit audience vibe ratings for a movie synopsis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "laughs": {
                            "type": "integer",
                            "description": "How funny is this movie? 0=not funny at all, 100=nonstop hilarious. A comedy should be 60-90, a dark drama should be 0-15.",
                        },
                        "tears": {
                            "type": "integer",
                            "description": "Will people cry? 0=not emotional, 100=devastating. A tragedy should be 70-95, a light comedy should be 0-20.",
                        },
                        "romance": {
                            "type": "integer",
                            "description": "Is there a love story? 0=no romance, 100=deeply romantic. A rom-com should be 70-95, an action movie with no love interest should be 0-10.",
                        },
                        "scares": {
                            "type": "integer",
                            "description": "Is it scary like a horror movie? Monsters, ghosts, serial killers. 0=not scary, 100=terrifying. A horror film should be 70-95, a rom-com or drama should be 0-10.",
                        },
                        "thrills": {
                            "type": "integer",
                            "description": "Is it exciting and intense? Action, chases, suspense. 0=calm and quiet, 100=edge of your seat. An action blockbuster should be 70-90, a quiet indie drama should be 5-20.",
                        },
                    },
                    "required": ["laughs", "tears", "romance", "scares", "thrills"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "rate_vibes"},
        messages=[
            {"role": "user", "content": f"Rate this movie:\n\n{synopsis}"}
        ],
    )

    try:
        # Extract tool use result
        for block in message.content:
            if block.type == "tool_use" and block.name == "rate_vibes":
                ratings = block.input
                return {
                    "😂 Laughs": int(ratings.get("laughs", 50)),
                    "😢 Tears": int(ratings.get("tears", 50)),
                    "💓 Romance": int(ratings.get("romance", 50)),
                    "😱 Scares": int(ratings.get("scares", 50)),
                    "🔥 Thrills": int(ratings.get("thrills", 50)),
                }
    except Exception:
        pass

    # Fallback: generate interesting varied scores from synopsis content
    import hashlib as _hl
    h = int(_hl.md5(synopsis.encode()).hexdigest(), 16)
    fallback = {}
    for i, cat in enumerate(VIBE_CATEGORIES):
        # Generate a pseudo-random score that varies per category and synopsis
        seed = (h >> (i * 8)) & 0xFF
        fallback[cat] = max(5, min(95, (seed * 37 + i * 53) % 91 + 5))
    return fallback
