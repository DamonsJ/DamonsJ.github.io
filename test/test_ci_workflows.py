#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRETTIER_WORKFLOW = ROOT / ".github/workflows/prettier.yml"


class CiWorkflowTests(unittest.TestCase):
    def test_prettier_workflow_uses_locked_dependencies(self) -> None:
        workflow = PRETTIER_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("run: npm ci", workflow)
        self.assertNotIn(
            "npm install --save-dev --save-exact prettier @shopify/prettier-plugin-liquid",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
