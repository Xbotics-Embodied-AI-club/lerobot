"""录制控制循环改用实时调度（Xbotics）：何时开、用什么参数、没权限时照常录制。"""

import os

import pytest

from lerobot.utils import control_priority


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(
        os, "sched_setscheduler", lambda pid, policy, param: seen.append((pid, policy, param))
    )
    monkeypatch.setattr(control_priority.platform, "system", lambda: "Linux")
    return seen


def _board(monkeypatch, on: bool):
    from lerobot.datasets import rkmpp

    monkeypatch.setattr(rkmpp, "rkmpp_available", lambda: on)


def test_default_on_board_uses_fifo_and_resets_children(calls, monkeypatch):
    """板子上默认：本线程 SCHED_FIFO，带 RESET_ON_FORK（之后新建的线程与子进程回到普通调度）。"""
    monkeypatch.delenv("LEROBOT_LOOP_RT", raising=False)
    _board(monkeypatch, True)
    assert control_priority.raise_control_loop_priority()
    ((pid, policy, param),) = calls
    assert pid == 0
    assert policy == os.SCHED_FIFO | os.SCHED_RESET_ON_FORK
    assert param.sched_priority == control_priority.DEFAULT_PRIORITY


def test_default_off_elsewhere(calls, monkeypatch):
    monkeypatch.delenv("LEROBOT_LOOP_RT", raising=False)
    _board(monkeypatch, False)
    assert not control_priority.raise_control_loop_priority()
    assert calls == []


def test_env_turns_it_off_or_sets_priority(calls, monkeypatch):
    _board(monkeypatch, True)
    monkeypatch.setenv("LEROBOT_LOOP_RT", "0")
    assert not control_priority.raise_control_loop_priority()
    assert calls == []
    _board(monkeypatch, False)
    monkeypatch.setenv("LEROBOT_LOOP_RT", "30")
    assert control_priority.raise_control_loop_priority()
    assert calls[0][2].sched_priority == 30


def test_without_permission_recording_continues(monkeypatch, caplog):
    def denied(*args):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "sched_setscheduler", denied)
    monkeypatch.setenv("LEROBOT_LOOP_RT", "10")
    assert not control_priority.raise_control_loop_priority()
    assert "recording continues" in caplog.text


@pytest.fixture
def gc_state():
    import gc

    yield gc
    control_priority.release_gc()
    gc.unfreeze()
    gc.enable()


def test_gc_held_only_when_realtime_is_on(gc_state, monkeypatch):
    monkeypatch.setenv("LEROBOT_LOOP_RT", "0")
    assert not control_priority.hold_gc()
    assert gc_state.isenabled()
    control_priority.collect_gc_between_episodes()  # 没接管时什么都不做
    assert gc_state.isenabled()


def test_gc_moves_between_episodes(gc_state, monkeypatch):
    """开启后：录制中途不自动回收（已有对象冻结）；两条之间回收后仍关着；录制结束恢复。"""
    monkeypatch.setenv("LEROBOT_LOOP_RT", "10")
    assert control_priority.hold_gc()
    assert not gc_state.isenabled() and gc_state.get_freeze_count() > 0
    control_priority.collect_gc_between_episodes()
    assert not gc_state.isenabled()
    control_priority.release_gc()
    assert gc_state.isenabled() and gc_state.get_freeze_count() == 0
