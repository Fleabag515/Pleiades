"""Host a Pleiades character as a Discord bot.

One bot process per character. The token comes from the character's vault
(`discord.token`). On a DM or a mention, the message is run through the engine and the
reply is sent back. Requires the [discord] extra and the Message Content intent enabled
in the Discord Developer Portal.
"""

from __future__ import annotations

from typing import Optional

from ..engine import Engine
from ..profiles import Profile


def run_discord_bot(name: str, *, engine: Optional[Engine] = None) -> None:
    try:
        import discord
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("discord.py not installed. Install the [discord] extra.") from e

    engine = engine or Engine()
    manager = engine.manager
    profile: Profile = manager.get(name)

    with manager.open_vault(name) as vault:
        token = vault.get("discord.token")
    if not token:
        raise RuntimeError(
            f"No discord.token in the vault for '{name}'. "
            f"Add it with: pleiades vault set {name} discord.token"
        )

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[pleiades] {name} connected to Discord as {client.user}.")

    @client.event
    async def on_message(message: "discord.Message"):
        if message.author == client.user:
            return
        is_dm = message.guild is None
        is_mention = client.user in getattr(message, "mentions", [])
        if not (is_dm or is_mention):
            return

        content = message.content
        if is_mention and client.user is not None:
            content = content.replace(client.user.mention, "").strip()
        if not content:
            return

        async with message.channel.typing():
            # engine.run is blocking; run it off the event loop.
            import asyncio

            reply = await asyncio.to_thread(engine.run, profile, content)

        for chunk in _split(reply or "(no response)"):
            await message.channel.send(chunk)

    client.run(token)


def _split(text: str, limit: int = 1900):
    """Discord caps messages at 2000 chars; chunk safely."""
    for i in range(0, len(text), limit):
        yield text[i : i + limit]
