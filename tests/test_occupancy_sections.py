import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from robot_workspace_dexterous.occupancy import fixed_occupancy
from robot_workspace_dexterous.visualize import section_mask


class OccupancyTests(unittest.TestCase):
    def test_fixed_material_surface_and_free_space(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / 'robot.urdf'
            urdf.write_text('<robot><link name="world"/><link name="base"/>'
                '<link name="moving"/><joint type="fixed"><parent link="world"/>'
                '<child link="base"/><origin xyz="2 0 0"/></joint>'
                '<joint type="revolute"><parent link="base"/>'
                '<child link="moving"/></joint></robot>')
            mesh = trimesh.creation.box()
            part = SimpleNamespace(link='base', vertices=mesh.vertices,
                                   faces=mesh.faces, watertight=True)
            moving = SimpleNamespace(link='moving', vertices=mesh.vertices,
                                     faces=mesh.faces, watertight=True)
            model = SimpleNamespace(parts=[part, moving], metadata={})
            points = np.array([[2., 0, 0], [2.5, 0, 0], [3, 0, 0], [0, 0, 0]])
            blocked, unresolved = fixed_occupancy(model, urdf, points)
            self.assertEqual(blocked.tolist(), [True, True, False, False])
            self.assertEqual(unresolved, [])
            # A nested shell is a cavity, not occupied material.
            inner = trimesh.creation.box(extents=[.4]*3)
            inner.invert()
            hollow = trimesh.util.concatenate([mesh, inner])
            part.vertices, part.faces = hollow.vertices, hollow.faces
            blocked, _ = fixed_occupancy(model, urdf, np.array([[2., 0, 0], [2.4, 0, 0]]))
            self.assertEqual(blocked.tolist(), [False, True])
            # Broken shells must not silently claim a known solid interior.
            part.faces, part.watertight = mesh.faces[:-1], False
            part.vertices = mesh.vertices
            _, unresolved = fixed_occupancy(model, urdf, points)
            self.assertEqual(unresolved, ['base'])

    def test_sections_do_not_project_other_depths(self):
        points = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 1e-7]])
        mask, selected = section_mask(points, 2, 0.)
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(selected, 0.)


if __name__ == '__main__':
    unittest.main()
