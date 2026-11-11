"""Public-facing project names use KnowGPO; legacy code identifiers stay intact."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PublicNaming(unittest.TestCase):
    def test_readme_uses_historical_version_only_in_compatibility_note(self):
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(readme.count("V7"), 1)
        self.assertIn("historically labeled V7 RGC-IR", readme)
        self.assertIn("## 🚀 Train KnowGPO\n", readme)

    def test_project_description_uses_method_name(self):
        metadata = (ROOT / "pyproject.toml").read_text()
        self.assertNotIn("V7", metadata)
        self.assertIn("KnowGPO", metadata)

    def test_launcher_messages_and_experiment_name_use_method_name(self):
        result = subprocess.run(
            ["bash", "scripts/train_knowgpo.sh", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer.experiment_name=knowgpo", result.stdout)
        self.assertIn("reward_model.reward_manager=apiprimedapotrajectoryv7rgcir", result.stdout)
        rejected = subprocess.run(
            ["bash", "scripts/train_knowgpo.sh", "--dry-run", "reward_model.reward_manager=naive"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("KnowGPO", rejected.stderr)
        self.assertNotIn("V7", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
