"""Gateway visible thread fork tests.

A visible fork is different from /branch: it must keep the current thread on its
original session and bind a newly-created platform thread to a copied child
session inside the live SessionStore.  This prevents the running gateway from
later overwriting sessions.json with a fresh timestamp session for the fork.
"""
from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _conversation_core(messages):
    """Strip volatile SessionDB metadata such as timestamps for assertions."""
    keys = ("role", "content", "message_id")
    return [{k: msg[k] for k in keys if k in msg} for msg in messages]


@pytest.fixture
def session_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "state.db")
    yield db
    db.close()


class FakeSessionStore:
    def __init__(self, source: SessionSource, session_id: str, history: list[dict]):
        self.entries: dict[str, SessionEntry] = {}
        self.bound: list[tuple[str, str]] = []
        self.histories: dict[str, list[dict]] = {session_id: history}
        key = build_session_key(source)
        self.entries[key] = SessionEntry(
            session_key=key,
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            display_name=source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
        )

    def get_or_create_session(self, source: SessionSource):
        return self.entries[build_session_key(source)]

    def load_transcript(self, session_id: str):
        return list(self.histories.get(session_id, []))

    def bind_session(self, source: SessionSource, target_session_id: str, display_name: str | None = None):
        key = build_session_key(source)
        entry = SessionEntry(
            session_key=key,
            session_id=target_session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            display_name=display_name or source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
        )
        self.entries[key] = entry
        self.bound.append((key, target_session_id))
        return entry


@pytest.fixture
def current_source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="source-thread",
        chat_name="AI Hub / work / Source Lane",
        chat_type="thread",
        user_id="user-1",
        user_name="Tester",
        thread_id="source-thread",
        guild_id="guild-1",
        parent_chat_id="parent-forum",
        chat_topic="Durable Agent work surface.",
    )


@pytest.fixture
def runner(session_db, current_source):
    from gateway.run import GatewayRunner

    history = [
        {"role": "user", "content": "Remember sentinel VF-CONTINUITY."},
        {"role": "assistant", "content": "Sentinel VF-CONTINUITY stored."},
    ]
    source_session_id = "source-session"
    session_db.create_session(session_id=source_session_id, source="discord", user_id="user-1")
    session_db.set_session_title(source_session_id, "Source Lane")
    for msg in history:
        session_db.append_message(session_id=source_session_id, role=msg["role"], content=msg["content"])

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value="fork-thread")
    adapter.export_visible_fork_context = None
    adapter.send = AsyncMock()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = FakeSessionStore(current_source, source_session_id, history)
    runner._session_db = session_db
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    return runner, adapter


@pytest.mark.asyncio
async def test_visible_fork_creates_new_thread_and_binds_child_session_in_live_store(runner, current_source, session_db):
    runner, adapter = runner
    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    adapter.create_handoff_thread.assert_awaited_once_with("parent-forum", "Fork Lane")
    assert "Fork Lane" in response
    assert "fork-thread" in response

    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    assert session_db.get_session_title(child_id) == "Fork Lane"
    assert _conversation_core(session_db.get_messages_as_conversation(child_id)) == [
        {"role": "user", "content": "Remember sentinel VF-CONTINUITY."},
        {"role": "assistant", "content": "Sentinel VF-CONTINUITY stored."},
    ]

    fork_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="fork-thread",
        chat_name="Fork Lane",
        chat_type="thread",
        user_id=current_source.user_id,
        user_name=current_source.user_name,
        thread_id="fork-thread",
        guild_id=current_source.guild_id,
        parent_chat_id="parent-forum",
        chat_topic=current_source.chat_topic,
    )
    fork_key = build_session_key(fork_source)
    assert runner.session_store.entries[fork_key].session_id == child_id
    assert runner.session_store.bound == [(fork_key, child_id)]

    # Original visible thread remains bound to the original session.
    assert runner.session_store.entries[build_session_key(current_source)].session_id == "source-session"
    assert session_db.get_session("source-session")["ended_at"] is None


@pytest.mark.asyncio
async def test_visible_fork_requires_thread_parent(runner, current_source):
    runner, _adapter = runner
    current_source.parent_chat_id = None
    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "parent" in response.lower() or "thread" in response.lower()


@pytest.mark.asyncio
async def test_visible_fork_without_name_defaults_to_current_thread_title(runner, current_source, session_db):
    runner, adapter = runner
    event = MessageEvent(text="/fork", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    adapter.create_handoff_thread.assert_awaited_once_with("parent-forum", "Source Lane Fork")
    assert "Source Lane Fork" in response

    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    assert session_db.get_session_title(branch_sessions[0]["id"]) == "Source Lane Fork"


@pytest.mark.asyncio
async def test_fork_default_title_strips_discord_channel_prefix(runner, current_source):
    runner, adapter = runner
    current_source.chat_name = "AI Hub / work / #1P · Ju-Young"
    event = MessageEvent(text="/fork", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    adapter.create_handoff_thread.assert_awaited_once_with("parent-forum", "1P · Ju-Young Fork")
    assert "#1P" not in response
    assert "1P · Ju-Young Fork" in response


@pytest.mark.asyncio
async def test_fork_appends_visible_platform_context_tail(runner, current_source, session_db):
    runner, adapter = runner
    adapter.export_visible_fork_context = AsyncMock(return_value=[
        {"role": "user", "content": "[OMELET] current visible question"},
        {"role": "assistant", "content": "current visible answer"},
    ])
    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "No conversation" not in response
    adapter.export_visible_fork_context.assert_awaited_once()
    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    assert _conversation_core(session_db.get_messages_as_conversation(child_id)) == [
        {"role": "user", "content": "Remember sentinel VF-CONTINUITY."},
        {"role": "assistant", "content": "Sentinel VF-CONTINUITY stored."},
        {"role": "user", "content": "[OMELET] current visible question"},
        {"role": "assistant", "content": "current visible answer"},
    ]


@pytest.mark.asyncio
async def test_fork_uses_message_id_dedupe_for_visible_tail(runner, current_source, session_db):
    runner, adapter = runner
    runner.session_store.histories["source-session"] = [
        {"role": "user", "content": "repeat", "message_id": "discord-1"},
        {"role": "user", "content": "repeat", "message_id": "discord-2"},
    ]
    adapter.export_visible_fork_context = AsyncMock(return_value=[
        {"role": "user", "content": "repeat", "message_id": "discord-1"},
        {"role": "user", "content": "repeat", "message_id": "discord-3"},
    ])
    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "No conversation" not in response
    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    copied = session_db.get_messages_as_conversation(child_id)
    assert [m["message_id"] for m in copied] == ["discord-1", "discord-2", "discord-3"]


@pytest.mark.asyncio
async def test_fork_of_fork_uses_immediate_parent_session(runner, current_source, session_db):
    runner, _adapter = runner
    source_key = build_session_key(current_source)
    fork_parent_id = "fork-parent-session"
    session_db.create_session(
        session_id=fork_parent_id,
        source="discord",
        user_id="user-1",
        parent_session_id="source-session",
    )
    runner.session_store.entries[source_key].session_id = fork_parent_id
    runner.session_store.histories[fork_parent_id] = [
        {"role": "user", "content": "message added inside first fork"},
        {"role": "assistant", "content": "first fork context stored"},
    ]
    event = MessageEvent(text="/fork Second Fork", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "Second Fork" in response
    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == fork_parent_id
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    assert _conversation_core(session_db.get_messages_as_conversation(child_id)) == [
        {"role": "user", "content": "message added inside first fork"},
        {"role": "assistant", "content": "first fork context stored"},
    ]


@pytest.mark.asyncio
async def test_oversized_fork_preflights_to_curated_active_history(runner, current_source, session_db):
    runner, _adapter = runner
    runner.config = {
        "model": {"default": "test-model", "context_length": 80},
        "compression": {"threshold": 0.5, "protect_first_n": 1, "protect_last_n": 2},
    }
    runner.session_store.histories["source-session"] = [
        {"role": "user", "content": "first anchor"},
        {"role": "assistant", "content": "middle " + ("x" * 1000)},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    event = MessageEvent(text="/fork Curated Fork", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "Fork mode: `curated_active_with_raw_archive`" in response
    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    copied = _conversation_core(session_db.get_messages_as_conversation(child_id))
    assert copied[0] == {"role": "user", "content": "first anchor"}
    assert "Visible fork preflight" in copied[1]["content"]
    assert copied[-2:] == [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]


@pytest.mark.asyncio
async def test_fork_reports_failure_on_partial_copy(runner, current_source, session_db, monkeypatch):
    runner, _adapter = runner
    original_append = session_db.append_message
    calls = {"count": 0}

    def flaky_append(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(session_db, "append_message", flaky_append)
    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "Visible fork failed" in response
    assert "copied only 1/2" in response
    assert runner.session_store.bound == []


@pytest.mark.asyncio
async def test_fork_uses_durable_route_context_when_current_thread_session_is_fresh(
    runner, current_source, session_db
):
    runner, _adapter = runner
    source_key = build_session_key(current_source)
    fresh_session_id = "fresh-route-slip-session"
    session_db.create_session(session_id=fresh_session_id, source="discord", user_id="user-1")
    runner.session_store.entries[source_key].session_id = fresh_session_id
    runner.session_store.histories[fresh_session_id] = [
        {"role": "user", "content": "fresh route-slip prompt only"},
    ]

    hermes_home = os.environ["HERMES_HOME"]
    routes_path = os.path.join(hermes_home, "discord_routes.json")
    with open(routes_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "routes": {
                    "source-lane": {
                        "discord_thread_id": current_source.thread_id,
                        "current_hermes_session_id": "source-session",
                    }
                }
            },
            f,
        )

    event = MessageEvent(text="/fork Fork Lane", source=current_source, message_id="m1")

    response = await runner._handle_visible_fork_command(event)

    assert "No conversation" not in response
    branch_sessions = [
        s for s in session_db.list_sessions_rich(limit=20)
        if s["parent_session_id"] == "source-session"
    ]
    assert len(branch_sessions) == 1
    child_id = branch_sessions[0]["id"]
    assert _conversation_core(session_db.get_messages_as_conversation(child_id)) == [
        {"role": "user", "content": "Remember sentinel VF-CONTINUITY."},
        {"role": "assistant", "content": "Sentinel VF-CONTINUITY stored."},
        {"role": "user", "content": "fresh route-slip prompt only"},
    ]
    # The original thread binding is not silently rebound during the fork.
    assert runner.session_store.entries[source_key].session_id == fresh_session_id


def test_fork_command_registry_no_legacy_aliases():
    from hermes_cli.commands import resolve_command

    assert resolve_command("fork").name == "fork"
    assert resolve_command("visible-fork") is None
    assert resolve_command("fork-thread") is None
    assert resolve_command("thread-fork") is None
