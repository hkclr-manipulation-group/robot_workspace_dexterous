import json
import unittest

import numpy as np

from robot_workspace_dexterous.layer_preview import xy_layers
from robot_workspace_dexterous.sampling import DexterousWorkspace


class LayerPreviewTests(unittest.TestCase):
    def test_depths_are_separate_and_unprocessed_points_are_hidden(self):
        workspace = DexterousWorkspace(
            np.array([[0., 0., -1.], [0., 0., 0.], [0., 0., 1.], [1., 0., 1.]]),
            np.array([.5, 0., 1., 1.]), np.array([1, 0, 2, 2]), 2)
        html = xy_layers(workspace, np.array([2, 2, 1, 0]), 0.)
        payload = html.split('<script id="xy-data" type="application/json">')[1].split('</script>')[0]
        data = json.loads(payload)
        self.assertEqual(data['center'], 1)
        self.assertEqual([x['z'] for x in data['layers']], [-1., 0., 1.])
        self.assertEqual([len(x['points']) for x in data['layers']], [1, 0, 1])
        self.assertEqual(data['layers'][2]['tested'], 1)
        self.assertEqual(data['layers'][2]['complete'], 0)
        self.assertEqual(data['layers'][2]['total'], 2)
        self.assertIn("location.hash", html)


if __name__ == '__main__':
    unittest.main()
