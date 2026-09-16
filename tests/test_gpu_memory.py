import sys
from types import SimpleNamespace
import pytest
from robot_workspace_dexterous.gpu_memory import configure_gpu_memory


@pytest.mark.parametrize('free,expected', [(120, 90), (100, 70)])
def test_budget_preserves_headroom(monkeypatch, free, expected):
    calls = []
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
        mem_get_info=lambda device: (free*2**30, 120*2**30),
        set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, device)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    assert configure_gpu_memory(.75) == expected*2**30
    assert calls == [(expected/120, 0)]


def test_insufficient_headroom_does_not_start_allocation(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           mem_get_info=lambda device: (7, 120))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match='reserve'):
        configure_gpu_memory(.75)
