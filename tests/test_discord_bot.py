"""Discord connector: no rate-limiting meant N inbound DMs fanned out into N
concurrent multi-round agent loops, and exec_policy="allow" (a real, plausible
per-character convenience setting) applied unmodified to messages from anyone
who could DM the bot. See HANDOFF's 2026-07-25 correctness audit, finding #6."""

from pleiades.connectors.discord_bot import _floor_exec_policy_for_discord, _InFlightGate
from pleiades.profiles import Profile


def test_floor_clamps_allow_to_ask_without_mutating_original():
    p = Profile(name="test", exec_policy="allow")
    floored = _floor_exec_policy_for_discord(p)
    assert floored.exec_policy == "ask"
    assert p.exec_policy == "allow"  # the original profile is untouched
    assert floored is not p


def test_floor_leaves_other_policies_unchanged():
    for policy in (None, "ask", "deny", "preset"):
        p = Profile(name="test", exec_policy=policy)
        assert _floor_exec_policy_for_discord(p) is p  # no copy needed/made


def test_in_flight_gate_blocks_second_entry_from_same_author():
    gate = _InFlightGate()
    assert gate.try_enter("author-1") is True
    assert gate.try_enter("author-1") is False  # already in flight
    assert gate.try_enter("author-2") is True  # different author, unaffected


def test_in_flight_gate_allows_reentry_after_leave():
    gate = _InFlightGate()
    assert gate.try_enter("author-1") is True
    gate.leave("author-1")
    assert gate.try_enter("author-1") is True
