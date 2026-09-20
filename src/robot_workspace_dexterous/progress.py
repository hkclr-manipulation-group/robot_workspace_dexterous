"""CPU snapshots for inspecting a workspace while IK is still running."""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import time

import numpy as np

from .sampling import DexterousWorkspace


class WorkspaceProgress:
    def __init__(self, directory: Path, link: str, metadata: dict):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.link, self.metadata = link, metadata
        self.last_print = -float('inf')
        self._status({'state': 'initializing', 'done': 0, 'total': None}, False)

    def _status(self, status: dict, has_preview: bool):
        status = {**self.metadata, **status, 'ee_link': self.link,
                  'updated_utc': datetime.now(timezone.utc).isoformat()}
        temporary = self.directory / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(self.directory / 'status.json')
        total = status.get('total')
        percent = 100 * status['done'] / total if total else 0
        preview = '<img src="preview.png?t='+str(time.time_ns())+'" style="width:100%">' if has_preview else '<p>Initializing GPU model; waiting for the first completed batch.</p>'
        collision_note = ('Collision results use STL triangle intersections and closed-mesh containment; '
                          'open meshes are checked as surfaces.'
                          if self.metadata.get('collision_model', {}).get('mode') == 'stl'
                          else 'Collision results use the configured sphere approximation.')
        page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>Workspace progress</title>
<body style="font:16px system-ui;background:#f3f6fa;color:#172b43;margin:32px">
<h1>{escape(self.link)}</h1><h2>{escape(status['state'])} · {percent:.1f}%</h2>
<progress value="{percent}" max="100" style="width:100%"></progress>
<p>Updated {status['updated_utc']} · {status['done']:,} / {total or '?'} IK goals</p>
<p>Latest saved snapshot; this page refreshes every 5 seconds. An unchanged timestamp means no new snapshot has arrived.</p>
{preview}<p>XY, XZ and YZ sections at the labelled grid coordinates; no depth projection.
Unprocessed cells are unknown. Partial-cell dexterity is a lower bound until all orientations are tested.
{collision_note}</p></body></html>'''
        temporary = self.directory / 'index.tmp'
        temporary.write_text(page, encoding='utf-8')
        temporary.replace(self.directory / 'index.html')

    def progress(self, done: int, total: int, elapsed: float):
        now = time.monotonic()
        if done != total and now - self.last_print < 5:
            return
        self.last_print = now
        rate = done / elapsed if elapsed else 0
        eta = (total-done)/rate/60 if rate else float('inf')
        print(f'{self.link}: IK {done:,}/{total:,} ({100*done/total:.1f}%) · '
              f'{rate:.0f} goals/s · elapsed {elapsed/60:.1f} min · ETA {eta:.1f} min', flush=True)

    def snapshot(self, workspace: DexterousWorkspace, done: int, total: int, elapsed: float):
        # The solver visits cells sequentially and all orientations within each cell.
        tested = np.clip(done - np.arange(len(workspace.positions), dtype=np.int64)
                         * workspace.orientation_count, 0, workspace.orientation_count)
        temporary = self.directory / 'partial.tmp.npz'
        workspace.save(str(temporary))
        with np.load(temporary) as saved:
            arrays = dict(saved)
        np.savez_compressed(temporary, **arrays, tested_orientations=tested,
                            complete=done == total, done=done, total=total)
        temporary.replace(self.directory / 'partial.npz')
        self._preview(workspace, tested)
        self._status({'state': 'complete' if done == total else 'running',
                      'done': done, 'total': total, 'elapsed_seconds': elapsed,
                      'tested_cells': int(np.count_nonzero(tested)),
                      'completed_cells': int(np.count_nonzero(tested == workspace.orientation_count)),
                      'reachable_cells_so_far': int(np.count_nonzero(workspace.reachable_orientations))}, True)
        print(f'Snapshot: {self.directory / "index.html"}', flush=True)

    def _preview(self, workspace: DexterousWorkspace, tested: np.ndarray):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(12, 4), layout='constrained')
        FigureCanvasAgg(figure)
        keep = (tested > 0) & (workspace.reachable_orientations > 0)
        axes = figure.subplots(1, 3)
        from .visualize import section_mask
        sections = self.metadata.get('plot_sections', (0., 0., 0.))
        for axis, (i, j, label) in zip(axes, [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]):
            normal = 3 - i - j
            mask, selected = section_mask(workspace.positions, normal, sections[normal])
            visible = keep & mask
            xy = workspace.positions[visible][:, [i, j]]
            scores = workspace.dexterity[visible]
            artist = axis.scatter(xy[:, 0], xy[:, 1], c=scores, s=9, marker='s',
                                  linewidths=0, cmap='turbo', vmin=0, vmax=1)
            for dimension, setter in ((i, axis.set_xlim), (j, axis.set_ylim)):
                low, high = workspace.positions[:, dimension].min(), workspace.positions[:, dimension].max()
                setter(float(low)-.01, float(high)+.01)
            axis.set(title=f'{label} section at {"XYZ"[normal]}={selected:.3f} m', xlabel='XYZ'[i]+' (m)', ylabel='XYZ'[j]+' (m)', aspect='equal')
            axis.grid(alpha=.15)
        figure.colorbar(artist, ax=list(axes), label='Orientation coverage (lower bound while partial)', shrink=.8)
        figure.suptitle(f'{self.link} · {int(np.count_nonzero(keep)):,} reachable cells so far')
        temporary = self.directory / 'preview.tmp.png'
        figure.savefig(temporary, dpi=130)
        temporary.replace(self.directory / 'preview.png')
