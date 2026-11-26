import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.decontamination_service import check_sample_contamination
from app.services.flywheel_service import _synthesis_leak_text, _validate_synthesized_sample
from app.services.llamafactory_config_service import generate_training_yaml


class DecontaminationTests(unittest.TestCase):
    def test_detects_direct_leak_and_keeps_clean_sample(self) -> None:
        corpus = ["某仓库原有货物 240 箱，第一天运出总数的四分之一。"]
        leaked = check_sample_contamination(corpus[0], corpus)
        clean = check_sample_contamination("计算圆周长，半径为 7。", corpus)
        self.assertTrue(leaked["is_contaminated"])
        self.assertEqual(leaked["method"], "13-gram_exact_overlap")
        self.assertFalse(clean["is_contaminated"])

    def test_code_boilerplate_does_not_create_false_positive(self) -> None:
        benchmark = ["编写 Python 函数 first_unique_char(s) 返回第一个不重复字符索引。"]
        row = {
            "instruction": "编写 Python 函数 merge_intervals(intervals) 合并重叠区间。",
            "output": "def merge_intervals(intervals):\n    return intervals",
        }
        result = check_sample_contamination(
            _synthesis_leak_text(row, "code_sandbox"),
            benchmark,
            ngram_size=13,
            semantic_threshold=0.72,
        )
        self.assertFalse(result["is_contaminated"])


class SynthesisQualityTests(unittest.TestCase):
    def test_json_and_code_use_deterministic_gates(self) -> None:
        valid_json = {
            "instruction": "请提取订单信息并仅输出 JSON 对象。",
            "output": json.dumps({"order_id": "A-1"}, ensure_ascii=False),
        }
        invalid_json = {"instruction": "请提取订单信息并仅输出 JSON 对象。", "output": "{'order_id': 'A-1'}"}
        valid_code = {
            "instruction": "实现一个合并区间函数并返回新区间列表。",
            "output": "def merge_intervals(items):\n    return items",
        }
        invalid_code = {"instruction": "实现一个合并区间函数并返回新区间列表。", "output": "def broken("}
        self.assertTrue(_validate_synthesized_sample(valid_json, "json_format")[0])
        self.assertFalse(_validate_synthesized_sample(invalid_json, "json_format")[0])
        self.assertTrue(_validate_synthesized_sample(valid_code, "code_sandbox")[0])
        self.assertFalse(_validate_synthesized_sample(invalid_code, "code_sandbox")[0])


class QLoRAConfigTests(unittest.TestCase):
    def test_qlora_maps_to_lora_plus_four_bit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = SimpleNamespace(
                id=1,
                model_family="qwen",
                budget_level="low",
                base_model=r"D:\models\qwen3.5-2b",
                stage="sft",
                finetuning_type="qlora",
                template="qwen3_5_nothink",
                extra_config_json=None,
            )
            dataset = SimpleNamespace(
                id=12,
                status="converted",
                converted_file_path=str(root / "dataset.jsonl"),
            )
            paths = (root / "job.yaml", root / "output", root / "job.log")
            with patch("app.services.llamafactory_config_service._training_paths", return_value=paths):
                result = generate_training_yaml(job, dataset)
            config = yaml.safe_load(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["finetuning_type"], "lora")
            self.assertEqual(config["quantization_bit"], 4)
            self.assertEqual(config["quantization_method"], "bitsandbytes")
            self.assertEqual(config["template"], "qwen3_5_nothink")
            self.assertEqual(config["cutoff_len"], 512)


if __name__ == "__main__":
    unittest.main()
