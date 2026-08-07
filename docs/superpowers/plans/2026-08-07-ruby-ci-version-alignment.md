# Ruby CI Version Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Ruby-based GitHub Actions workflow install the locked bundle successfully by using Ruby 3.3.5 consistently.

**Architecture:** Treat the deployment workflow's Ruby 3.3.5 as the repository-wide CI baseline. Enforce that baseline with a static workflow test, then update only the three stale workflow version fields; application dependencies and the lockfile remain unchanged.

**Tech Stack:** GitHub Actions, ruby/setup-ruby, Python unittest, Ruby/Bundler, Jekyll, Prettier.

---

### Task 1: Add a failing workflow-version regression test

**Files:**

- Modify: `test/test_ci_workflows.py`

- [ ] **Step 1: Add the repository-wide Ruby version test**

```python
WORKFLOWS_DIR = ROOT / ".github/workflows"
EXPECTED_RUBY_VERSION = 'ruby-version: "3.3.5"'

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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python3 -m unittest test.test_ci_workflows.CiWorkflowTests.test_ruby_workflows_use_one_compatible_version`

Expected: failure naming `axe.yml`, `broken-links-site.yml`, and `taxonomy.yml` because each contains Ruby 3.2.2.

### Task 2: Align every Ruby workflow on 3.3.5

**Files:**

- Modify: `.github/workflows/axe.yml`
- Modify: `.github/workflows/broken-links-site.yml`
- Modify: `.github/workflows/taxonomy.yml`
- Modify: `_data/taxonomy.yml`

- [ ] **Step 1: Change the stale version fields**

In each file, replace:

```yaml
ruby-version: "3.2.2"
```

with:

```yaml
ruby-version: "3.3.5"
```

Do not change `Gemfile`, `Gemfile.lock`, Bundler configuration, workflow triggers, or workflow commands.

- [ ] **Step 2: Remove vocabulary orphaned by the deleted reading entry**

Delete `马里奥·普佐` and `犯罪小说` from the Reading tags in `_data/taxonomy.yml`. These terms were used only by the deleted `The Godfather` entry and otherwise cause the taxonomy workflow to fail after Ruby setup.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run: `python3 -m unittest test.test_ci_workflows.CiWorkflowTests.test_ruby_workflows_use_one_compatible_version`

Expected: one passing test with no failures.

### Task 3: Verify the complete repository and commit

**Files:**

- Modify: `docs/superpowers/plans/2026-08-07-ruby-ci-version-alignment.md`

- [ ] **Step 1: Format changed files**

Run: `npx prettier .github/workflows/axe.yml .github/workflows/broken-links-site.yml .github/workflows/taxonomy.yml _data/taxonomy.yml docs/superpowers/specs/2026-08-07-ruby-ci-version-alignment-design.md docs/superpowers/plans/2026-08-07-ruby-ci-version-alignment.md --write`

Expected: command exits 0.

- [ ] **Step 2: Verify the locked bundle and tests**

Run:

```bash
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle check
PATH=/opt/homebrew/opt/ruby/bin:$PATH python3 -m unittest discover -s test -p 'test_*.py'
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby -Itest test/test_taxonomy_validator.rb
```

Expected: the bundle is satisfied, nine or more Python tests pass, and all taxonomy validator tests pass.

- [ ] **Step 3: Verify formatting and build**

Run:

```bash
npx prettier . --check
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build
git diff --check
```

Expected: formatting passes, Jekyll exits 0, and Git reports no whitespace errors.

- [ ] **Step 4: Commit only the CI alignment files**

```bash
git add .github/workflows/axe.yml .github/workflows/broken-links-site.yml .github/workflows/taxonomy.yml _data/taxonomy.yml test/test_ci_workflows.py docs/superpowers/specs/2026-08-07-ruby-ci-version-alignment-design.md docs/superpowers/plans/2026-08-07-ruby-ci-version-alignment.md
git commit -m "fix: align Ruby version across CI"
```
