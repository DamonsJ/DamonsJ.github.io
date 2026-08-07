#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRETTIER_WORKFLOW = ROOT / ".github/workflows/prettier.yml"
TAXONOMY_WORKFLOW = ROOT / ".github/workflows/taxonomy.yml"
JEKYLL_CONFIG = ROOT / "_config.yml"
WORKFLOWS_DIR = ROOT / ".github/workflows"
EXPECTED_RUBY_VERSION = 'ruby-version: "3.3.5"'


class CiWorkflowTests(unittest.TestCase):
    def test_prettier_workflow_uses_locked_dependencies(self) -> None:
        workflow = PRETTIER_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("run: npm ci", workflow)
        self.assertNotIn(
            "npm install --save-dev --save-exact prettier @shopify/prettier-plugin-liquid",
            workflow,
        )

    def test_taxonomy_workflow_validates_source_and_generated_site(self) -> None:
        workflow = TAXONOMY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("bundle exec ruby scripts/validate_taxonomy.rb", workflow)
        self.assertIn("bundle exec jekyll build", workflow)
        self.assertIn(
            "bundle exec ruby scripts/validate_taxonomy.rb --site _site", workflow
        )

    def test_programming_permalink_does_not_depend_on_category(self) -> None:
        config = JEKYLL_CONFIG.read_text(encoding="utf-8")
        self.assertIn(
            "permalink: /programming/:year/:month/:day/:title:output_ext", config
        )

    def test_ruby_workflows_use_one_compatible_version(self) -> None:
        ruby_workflows = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = path.read_text(encoding="utf-8")
            if "ruby/setup-ruby@v1" in workflow:
                ruby_workflows.append(path)
                with self.subTest(path=path.name):
                    self.assertIn(EXPECTED_RUBY_VERSION, workflow)
                    self.assertNotIn('ruby-version: "3.2.2"', workflow)

        self.assertTrue(ruby_workflows, "expected at least one Ruby workflow")


if __name__ == "__main__":
    unittest.main()
