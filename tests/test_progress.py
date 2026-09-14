import json
import numpy as np
from robot_workspace_dexterous.progress import WorkspaceProgress
from robot_workspace_dexterous.sampling import DexterousWorkspace


def test_partial_and_final_snapshot(tmp_path):
    monitor = WorkspaceProgress(tmp_path, 'tool', {'resolution_m': .025})
    points = np.array([[0, 0, 0], [.025, 0, 0], [.05, 0, 0]])
    workspace = DexterousWorkspace(points, np.array([.5, .25, 0]), np.array([2, 1, 0]), 4)
    monitor.snapshot(workspace, 5, 12, 2.0)
    with np.load(tmp_path / 'partial.npz') as saved:
        assert saved['tested_orientations'].tolist() == [4, 1, 0]
        assert saved['reachable_orientations'].tolist() == [2, 1, 0]
        assert not saved['complete']
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['completed_cells'] == 1 and status['tested_cells'] == 2
    assert (tmp_path / 'preview.png').stat().st_size > 1000
    assert 'Unprocessed cells are unknown' in (tmp_path / 'index.html').read_text()
    monitor.snapshot(workspace, 12, 12, 3.0)
    with np.load(tmp_path / 'partial.npz') as saved:
        assert saved['tested_orientations'].tolist() == [4, 4, 4]
        assert saved['complete']
    assert json.loads((tmp_path / 'status.json').read_text())['state'] == 'complete'


def test_empty_reachable_preview(tmp_path):
    monitor = WorkspaceProgress(tmp_path, 'tool', {})
    workspace = DexterousWorkspace(np.array([[0, 0, 0], [.1, 0, 0]]), np.zeros(2), np.zeros(2, dtype=int), 4)
    monitor.snapshot(workspace, 1, 8, .5)
    assert (tmp_path / 'preview.png').is_file()
    assert json.loads((tmp_path / 'status.json').read_text())['reachable_cells_so_far'] == 0
