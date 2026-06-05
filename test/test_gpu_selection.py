import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


GPU_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "gpu.py"
GPU_SPEC = importlib.util.spec_from_file_location("gpu_utils_for_test", GPU_MODULE_PATH)
gpu_utils = importlib.util.module_from_spec(GPU_SPEC)
sys.modules[GPU_SPEC.name] = gpu_utils
GPU_SPEC.loader.exec_module(gpu_utils)

parse_gpu_id_list = gpu_utils.parse_gpu_id_list
resolve_gpu_selection = gpu_utils.resolve_gpu_selection


class GPUSelectionTests(unittest.TestCase):
    def test_default_four_visible_gpus_uses_all_gpus(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(device_count=4)

        self.assertEqual(selection.requested, "auto")
        self.assertEqual(selection.selected, [0, 1, 2, 3])
        self.assertEqual(selection.skipped, [])

    def test_default_eight_visible_gpus_uses_all_gpus(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(device_count=8)

        self.assertEqual(selection.selected, list(range(8)))
        self.assertEqual(selection.skipped, [])

    def test_legacy_four_gpu_behavior_with_explicit_gpus(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(["1", "2", "3"], device_count=4)

        self.assertEqual(selection.selected, [1, 2, 3])
        self.assertEqual(selection.skipped, [])

    def test_exclude_gpu0_preserves_old_default_when_requested(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(
                device_count=4,
                excluded_physical_gpus=[0],
            )

        self.assertEqual(selection.selected, [1, 2, 3])
        self.assertEqual(selection.skipped, [0])

    def test_allow_gpu0_keeps_full_range(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(["0-7"], allow_gpu0=True, device_count=8)

        self.assertEqual(selection.selected, list(range(8)))
        self.assertEqual(selection.skipped, [])

    def test_comma_and_range_tokens_are_supported(self):
        with patch.dict(os.environ, {}, clear=True):
            selection = resolve_gpu_selection(["1,2", "3-4", "4"], device_count=8)

        self.assertEqual(selection.requested_ids, [1, 2, 3, 4])
        self.assertEqual(selection.selected, [1, 2, 3, 4])

    def test_cuda_visible_devices_filters_by_physical_id(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2,3"}, clear=True):
            selection = resolve_gpu_selection(device_count=3)

        self.assertEqual(selection.visible_to_physical, {0: 1, 1: 2, 2: 3})
        self.assertEqual(selection.selected, [0, 1, 2])
        self.assertEqual(selection.skipped, [])

    def test_exclusion_uses_physical_id_with_cuda_visible_devices(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=True):
            selection = resolve_gpu_selection(
                device_count=4,
                excluded_physical_gpus=[0],
            )

        self.assertEqual(selection.selected, [1, 2, 3])
        self.assertEqual(selection.skipped, [0])

    def test_parse_exclude_gpu_ranges(self):
        self.assertEqual(parse_gpu_id_list(["0,2", "4-5"], device_count=8), [0, 2, 4, 5])

    def test_invalid_visible_id_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_gpu_selection(["0-8"], device_count=8)


if __name__ == "__main__":
    unittest.main()
