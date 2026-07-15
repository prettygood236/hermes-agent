"""Gateway /rewind visible-message cleanup tests."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="12345",
        chat_name="Home / #thread",
        chat_type="thread",
        user_id="111",
        user_name="Tester",
        thread_id="12345",
        parent_chat_id="999",
    )


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])

    from gateway.run import GatewayRunner

    source = _source()
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source)
    db = store._db
    assert db is not None
    db.append_message(entry.session_id, "user", "first question", platform_message_id="u1")
    db.append_message(entry.session_id, "assistant", "first answer", platform_message_id="b1")

    gw = object.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    gw._evict_cached_agent = MagicMock()
    gw.adapters = {Platform.DISCORD: SimpleNamespace(rewind_visible_turns=AsyncMock(return_value={
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "matched_user_message_ids": ["u1"],
        "matched_user_message_contents": ["first question"],
        "planned": 3,
        "planned_bot": 1,
        "planned_status": 1,
        "planned_user": 1,
        "deleted_bot": 1,
        "deleted_status": 1,
        "deleted_user": 1,
        "kept_user": 0,
        "failed": 0,
    }))}
    setattr(gw, "hooks", SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False))
    gw._running_agents = {}
    gw._running_agents_ts = {}
    gw._pending_messages = {}
    gw._pending_approvals = {}
    gw._update_prompt_pending = {}
    gw._startup_restore_in_progress = False
    gw._scale_to_zero_note_real_inbound = MagicMock()
    def _authorized(source):
        return True
    setattr(gw, "_is_user_authorized", _authorized)
    gw._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("/rewind must dispatch before normal agent turn")
    )
    yield gw, source, store, entry
    db.close()


@pytest.mark.asyncio
async def test_rewind_dispatches_from_gateway_message_handler(runner):
    gw, source, store, entry = runner
    event = MessageEvent(text="/rewind 1", source=source, message_id="cmd1")

    response = await gw._handle_message(event)

    assert "Rewound 1 turn" in response
    assert "bot replies deleted 1" in response
    assert store.load_transcript(entry.session_id) == []
    gw._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewind_rewinds_history_and_delegates_visible_cleanup(runner):
    gw, source, store, entry = runner
    event = MessageEvent(text="/rewind", source=source, message_id="cmd1")

    response = await gw._handle_rewind_command(event)

    assert "Rewound 1 turn" in response
    assert "bot replies deleted 1" in response
    assert "status messages deleted 1" in response
    assert store.load_transcript(entry.session_id) == []
    gw._evict_cached_agent.assert_called_once()
    gw.adapters[Platform.DISCORD].rewind_visible_turns.assert_awaited_once_with(
        source=source,
        event=event,
        turns=1,
        delete_user_messages=True,
        dry_run=False,
    )


@pytest.mark.asyncio
async def test_rewind_dry_run_does_not_mutate_history(runner):
    gw, source, store, entry = runner
    gw.adapters[Platform.DISCORD].rewind_visible_turns = AsyncMock(return_value={
        "supported": True,
        "dry_run": True,
        "matched_turns": 1,
        "planned": 2,
        "planned_bot": 1,
        "planned_user": 1,
    })
    event = MessageEvent(text="/rewind dry-run", source=source, message_id="cmd1")

    response = await gw._handle_rewind_command(event)

    assert "dry-run" in response
    assert "Session history: unchanged" in response
    assert len(store.load_transcript(entry.session_id)) == 2
    gw._evict_cached_agent.assert_not_called()
    gw.adapters[Platform.DISCORD].rewind_visible_turns.assert_awaited_once_with(
        source=source,
        event=event,
        turns=1,
        delete_user_messages=True,
        dry_run=True,
    )


@pytest.mark.asyncio
async def test_rewind_keeps_history_when_visible_user_was_not_committed(runner):
    gw, source, store, entry = runner
    gw.adapters[Platform.DISCORD].rewind_visible_turns = AsyncMock(return_value={
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "matched_user_message_ids": ["u2"],
        "matched_user_message_contents": ["second question stopped before commit"],
        "planned": 3,
        "planned_bot": 1,
        "planned_status": 1,
        "planned_user": 1,
        "deleted_bot": 1,
        "deleted_status": 1,
        "deleted_user": 1,
        "kept_user": 0,
        "failed": 0,
    })
    before = store.load_transcript(entry.session_id)
    event = MessageEvent(text="/rewind", source=source, message_id="cmd2")

    response = await gw._handle_rewind_command(event)

    assert "Session history: unchanged" in response
    assert "visible user turn was not the last committed Hermes prompt" in response
    assert "Backed-up prompt preview" not in response
    assert store.load_transcript(entry.session_id) == before
    gw._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_rewind_matches_routed_marker_payload_to_committed_prompt(runner):
    gw, source, store, entry = runner
    db = store._db
    assert db is not None
    db.append_message(
        entry.session_id,
        "user",
        "[Routed from Discord #home to default / 2A · Health]\n\nshoulder question from home",
    )
    db.append_message(entry.session_id, "assistant", "routed answer")
    gw.adapters[Platform.DISCORD].rewind_visible_turns = AsyncMock(return_value={
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "matched_user_message_ids": ["routed-marker-1"],
        "matched_user_message_contents": ["shoulder question from home"],
        "planned": 3,
        "planned_bot": 1,
        "planned_status": 1,
        "planned_user": 1,
        "deleted_bot": 1,
        "deleted_status": 1,
        "deleted_user": 1,
        "kept_user": 0,
        "failed": 0,
    })
    event = MessageEvent(text="/rewind", source=source, message_id="cmd-routed")

    response = await gw._handle_rewind_command(event)

    assert "Rewound 1 turn" in response
    remaining = store.load_transcript(entry.session_id)
    remaining_contents = [message["content"] for message in remaining]
    assert not any("shoulder question from home" in content for content in remaining_contents)
    assert "routed answer" not in remaining_contents
    assert "first question" in remaining_contents
    assert "first answer" in remaining_contents
    gw._evict_cached_agent.assert_called_once()


@pytest.mark.asyncio
async def test_rewind_keeps_history_for_bot_only_visible_cleanup(runner):
    gw, source, store, entry = runner
    gw.adapters[Platform.DISCORD].rewind_visible_turns = AsyncMock(return_value={
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "bot_only_visible_turn": True,
        "matched_user_message_ids": [],
        "matched_user_message_contents": [],
        "planned": 2,
        "planned_bot": 1,
        "planned_status": 1,
        "planned_user": 0,
        "deleted_bot": 1,
        "deleted_status": 1,
        "deleted_user": 0,
        "kept_user": 0,
        "failed": 0,
    })
    before = store.load_transcript(entry.session_id)
    event = MessageEvent(text="/rewind", source=source, message_id="cmd3")

    response = await gw._handle_rewind_command(event)

    assert "Session history: unchanged" in response
    assert "visible cleanup matched only bot/status artifacts" in response
    assert store.load_transcript(entry.session_id) == before
    gw._evict_cached_agent.assert_not_called()


class _AsyncHistory:
    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeChannel:
    def __init__(self, messages, manage_messages=None, bot_member=None):
        self.messages = messages
        self.history_kwargs = None
        self._manage_messages = manage_messages
        self.guild = SimpleNamespace(me=bot_member) if bot_member is not None else None

    def history(self, **kwargs):
        self.history_kwargs = kwargs
        return _AsyncHistory(self.messages)

    def permissions_for(self, _member):
        if self._manage_messages is None:
            raise AttributeError("permission check unavailable")
        return SimpleNamespace(manage_messages=self._manage_messages)


class _FakeMessage:
    def __init__(self, message_id, author, content="", delete_error=None, created_at=None):
        self.id = int(message_id)
        self.author = author
        self.content = content
        self.clean_content = content
        self.created_at = created_at or datetime.now()
        self.delete = AsyncMock()
        if delete_error is not None:
            self.delete.side_effect = delete_error


def _discord_adapter_with_history(messages, bot=None, manage_messages=None):
    from plugins.platforms.discord.adapter import DiscordAdapter

    bot = bot or SimpleNamespace(id=999, name="HermesBot", bot=True)
    channel = _FakeChannel(messages, manage_messages=manage_messages, bot_member=bot)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        user=bot,
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    return adapter, channel


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_deletes_bot_and_user_messages():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, channel = _discord_adapter_with_history([
        _FakeMessage(30, bot, "Hermes answer chunk 2"),
        _FakeMessage(29, bot, "Hermes answer chunk 1"),
        _FakeMessage(28, user, "user question"),
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="31")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 1
    assert result["matched_user_message_ids"] == ["28"]
    assert result["matched_user_message_contents"] == ["user question"]
    assert result["planned_bot"] == 2
    assert result["planned_user"] == 1
    assert result["deleted_bot"] == 2
    assert result["deleted_user"] == 1
    assert channel.history_kwargs["before"] is not None


@pytest.mark.asyncio
async def test_discord_rewind_counts_routed_user_marker_as_turn_boundary():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    old_answer = _FakeMessage(45, bot, "old answer must stay")
    old_question = _FakeMessage(44, user, "old direct question")
    routed_marker = _FakeMessage(
        47,
        bot,
        "↪ Routed user turn from #home · Tester\n\nshoulder question from home",
    )
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(50, bot, "new direct answer"),
        _FakeMessage(49, user, "new direct question"),
        _FakeMessage(48, bot, "routed answer"),
        routed_marker,
        old_answer,
        old_question,
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind 2", source=source, message_id="51")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=2)

    assert result["matched_turns"] == 2
    assert result["matched_user_message_ids"] == ["49", "47"]
    assert result["matched_user_message_contents"] == [
        "new direct question",
        "shoulder question from home",
    ]
    assert result["planned_user"] == 1
    assert result["planned_bot"] == 2
    assert result["planned_status"] == 1
    assert result["deleted_user"] == 1
    assert result["deleted_bot"] == 2
    assert result["deleted_status"] == 1
    old_answer.delete.assert_not_called()
    old_question.delete.assert_not_called()


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_keeps_cron_deliveries():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    cron = _FakeMessage(
        31,
        bot,
        "Cronjob Response: Ju-Young Intake Control Board\n(job_id: aee9fcaa05fb)\n-------------\n\nboard update",
    )
    answer = _FakeMessage(30, bot, "Hermes answer")
    question = _FakeMessage(29, user, "user question")
    adapter, _channel = _discord_adapter_with_history([cron, answer, question], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="32")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 1
    assert result["matched_user_message_ids"] == ["29"]
    assert result["planned_bot"] == 1
    assert result["deleted_bot"] == 1
    assert result["planned_user"] == 1
    assert result["deleted_user"] == 1
    assert result["kept_cron"] == 1
    cron.delete.assert_not_called()
    answer.delete.assert_awaited_once()
    question.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_keeps_cron_only_history():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    cron = _FakeMessage(
        31,
        bot,
        "Cronjob Response: Ju-Young Intake Control Board\n(job_id: aee9fcaa05fb)\n-------------\n\nboard update",
    )
    adapter, _channel = _discord_adapter_with_history([cron], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="32")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 0
    assert result["planned"] == 0
    assert result["kept_cron"] == 1
    cron.delete.assert_not_called()


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_deletes_turn_status_messages():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(33, bot, "Hermes final answer"),
        _FakeMessage(32, bot, "📚 Reading skill ju-young-smartux-screen-guides"),
        _FakeMessage(31, bot, "status bump without legacy marker"),
        _FakeMessage(30, user, "user question"),
    ], bot=bot)
    adapter._nonconversational_messages._ids["31"] = None
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="34")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 1
    assert result["matched_user_message_ids"] == ["30"]
    assert result["matched_user_message_contents"] == ["user question"]
    assert result["planned_bot"] == 1
    assert result["planned_status"] == 2
    assert result["planned_user"] == 1
    assert result["deleted_bot"] == 1
    assert result["deleted_status"] == 2
    assert result["deleted_user"] == 1


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_deletes_bot_only_slash_goal_cluster():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    now = datetime.now()
    lane_seed = _FakeMessage(
        1,
        bot,
        "3R · Interior 세션 생성",
        created_at=now - timedelta(days=14),
    )
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(34, bot, "⚡ Stopped. You can continue this session.", created_at=now - timedelta(seconds=5)),
        _FakeMessage(33, bot, "⏳ Working — 3 min — iteration 9/150", created_at=now - timedelta(seconds=55)),
        _FakeMessage(32, bot, "📚 Reading skill obsidian", created_at=now - timedelta(seconds=70)),
        _FakeMessage(31, bot, "⊙ Goal set (20-turn budget): build the wiki", created_at=now - timedelta(seconds=80)),
        lane_seed,
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="35", timestamp=now)

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 1
    assert result["bot_only_visible_turn"] is True
    assert result["matched_user_message_ids"] == []
    assert result["matched_user_message_contents"] == []
    assert result["planned_bot"] == 2
    assert result["planned_status"] == 2
    assert result["planned_user"] == 0
    assert result["deleted_bot"] == 2
    assert result["deleted_status"] == 2
    assert result["deleted_user"] == 0
    lane_seed.delete.assert_not_called()


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_classifies_legacy_imweb_progress_and_user_failures():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(35, bot, "📋 Updating tasks planning 5 task(s)\n🖥️ browser_console..."),
        _FakeMessage(34, user, "승인", delete_error=PermissionError("Missing Permissions")),
        _FakeMessage(33, bot, "📚 Reading skill writing-plans"),
        _FakeMessage(32, user, "첫단계로 구현할거 제시해.", delete_error=PermissionError("Missing Permissions")),
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind 2", source=source, message_id="36")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=2)

    assert result["matched_turns"] == 2
    assert result["planned_status"] == 2
    assert result["planned_user"] == 2
    assert result["deleted_status"] == 2
    assert result["deleted_user"] == 0
    assert result["failed"] == 2
    assert result["failed_user"] == 2


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_reports_user_failure_without_manage_messages():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(30, bot, "Hermes answer"),
        _FakeMessage(29, user, "user question"),
    ], bot=bot, manage_messages=False)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="31")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["matched_turns"] == 1
    assert result["planned_bot"] == 1
    assert result["planned_user"] == 1
    assert result["deleted_bot"] == 1
    assert result["deleted_user"] == 0
    assert result["kept_user"] == 0
    assert result["failed"] == 1
    assert result["failed_user"] == 1
    assert result["user_delete_permission"] == "missing_manage_messages"
    assert result["user_delete_failure_reason"] == "missing Manage Messages"


@pytest.mark.asyncio
async def test_rewind_result_reports_kept_cron_deliveries(runner):
    gw, _source_obj, _store, _entry = runner

    lines = gw._format_visible_rewind_result({
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "planned": 2,
        "planned_bot": 1,
        "planned_status": 0,
        "planned_user": 1,
        "deleted_bot": 1,
        "deleted_status": 0,
        "deleted_user": 1,
        "kept_user": 0,
        "kept_cron": 2,
        "failed": 0,
    }, delete_user_messages=True)

    assert lines == [
        "Visible messages: bot replies deleted 1; user messages deleted 1; cron deliveries kept 2."
    ]


@pytest.mark.asyncio
async def test_rewind_result_reports_kept_cron_when_no_visible_turn(runner):
    gw, _source_obj, _store, _entry = runner

    lines = gw._format_visible_rewind_result({
        "supported": True,
        "dry_run": False,
        "matched_turns": 0,
        "planned": 0,
        "kept_cron": 1,
        "failed": 0,
    }, delete_user_messages=True)

    assert lines == ["Visible messages: no matching visible turn found to delete; cron deliveries kept 1."]


@pytest.mark.asyncio
async def test_rewind_result_reports_user_failure_for_missing_manage_messages(runner):
    gw, _source_obj, _store, _entry = runner

    lines = gw._format_visible_rewind_result({
        "supported": True,
        "dry_run": False,
        "matched_turns": 1,
        "planned": 1,
        "planned_bot": 1,
        "planned_status": 0,
        "planned_user": 1,
        "deleted_bot": 1,
        "deleted_status": 0,
        "deleted_user": 0,
        "kept_user": 0,
        "failed": 1,
        "failed_user": 1,
        "user_delete_permission": "missing_manage_messages",
        "user_delete_failure_reason": "missing Manage Messages",
    }, delete_user_messages=True)

    assert lines == [
        "Visible messages: bot replies deleted 1; user messages deleted 0; "
        "failed 1 (user 1: missing Manage Messages)."
    ]


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_bot_only_keeps_user_message():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(30, bot, "Hermes answer"),
        _FakeMessage(29, user, "user question"),
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind bot-only", source=source, message_id="31")

    result = await adapter.rewind_visible_turns(
        source=source,
        event=event,
        turns=1,
        delete_user_messages=False,
    )

    assert result["planned_bot"] == 1
    assert result["planned_user"] == 0
    assert result["deleted_bot"] == 1
    assert result["deleted_user"] == 0
    assert result["kept_user"] == 1


@pytest.mark.asyncio
async def test_discord_rewind_visible_turns_aborts_across_other_user():
    bot = SimpleNamespace(id=999, name="HermesBot", bot=True)
    other = SimpleNamespace(id=222, name="Other", bot=False)
    user = SimpleNamespace(id=111, name="Tester", bot=False)
    adapter, _channel = _discord_adapter_with_history([
        _FakeMessage(30, other, "someone else spoke"),
        _FakeMessage(29, bot, "Hermes answer"),
        _FakeMessage(28, user, "user question"),
    ], bot=bot)
    source = _source()
    event = MessageEvent(text="/rewind", source=source, message_id="31")

    result = await adapter.rewind_visible_turns(source=source, event=event, turns=1)

    assert result["aborted"] is True
    assert "ambiguous" in result["reason"]
    assert result["planned"] == 0
