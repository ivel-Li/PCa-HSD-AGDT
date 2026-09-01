import importlib
import unittest
from unittest.mock import patch

import models.model_factory as factory


class ModelFactoryTest(unittest.TestCase):
    def test_agdt_and_legacy_name_build_the_masked_model(self):
        expected = {
            "vit_name": "vit_large_patch16_224",
            "vit_ckpt": None,
            "use_mask": True,
            "fusion": "transformer",
        }
        with patch.object(factory, "VITClassifier", return_value=object()) as build:
            factory.build_model("AGDT")
            factory.build_model("LSDT-Large")

        self.assertEqual([call.kwargs for call in build.call_args_list], [expected, expected])

    def test_no_mask_ablation_keeps_its_original_identifier(self):
        with patch.object(factory, "VITClassifier", return_value=object()) as build:
            factory.build_model("VITLargeAttentionClassifier")

        self.assertEqual(
            build.call_args.kwargs,
            {
                "vit_name": "vit_large_patch16_224",
                "vit_ckpt": None,
                "use_mask": False,
                "fusion": "transformer",
            },
        )

    def test_named_configs_only_override_their_ablation(self):
        no_mask = importlib.import_module("configs.no_mask")
        medicalsam3 = importlib.import_module("configs.medicalsam3")

        self.assertEqual(no_mask.model_name, "VITLargeAttentionClassifier")
        self.assertFalse(no_mask.LOAD_MASK)
        self.assertEqual(medicalsam3.model_name, "AGDT")
        self.assertEqual(medicalsam3.MASK_KEY, "medicalsam3")
        self.assertTrue(medicalsam3.LOAD_MASK)


if __name__ == "__main__":
    unittest.main()
