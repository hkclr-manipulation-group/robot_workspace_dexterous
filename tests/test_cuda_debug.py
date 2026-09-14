import sys
from types import SimpleNamespace

import pytest

from robot_workspace_dexterous.cuda_debug import cuda_stage


def test_async_error_is_attributed_to_stage_without_retry(monkeypatch, capsys):
    calls = []
    error = RuntimeError('CUDA error: an illegal memory access was encountered')

    def synchronize():
        calls.append('sync')
        raise error

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
    with pytest.raises(RuntimeError) as raised:
        with cuda_stage('IK goals 0:32', True):
            calls.append('solve')
    assert raised.value is error
    assert calls == ['solve', 'sync']
    log = capsys.readouterr().err
    assert 'FAILED IK goals 0:32' in log and 'OK' not in log


def test_immediate_error_does_not_touch_cuda_again(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: calls.append('sync'))))
    with pytest.raises(ValueError, match='bad input'):
        with cuda_stage('initialize IK', True):
            raise ValueError('bad input')
    assert not calls


def test_normal_execution_does_not_synchronize(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with cuda_stage('normal execution'):
        pass
