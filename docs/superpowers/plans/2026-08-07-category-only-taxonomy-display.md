# Category-Only Taxonomy Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show only secondary categories in the taxonomy strip on Programming, AI, Math, and Reading landing pages while preserving article tags, tag archives, and tag search.

**Architecture:** Keep `_data/taxonomy.yml` as the complete category/tag vocabulary, add each collection's landing path for generated-page validation, and make `_includes/collection_taxonomy.liquid` a category-only renderer. Extend the existing validator so CI rejects tag links that reappear in a landing-page taxonomy strip.

**Tech Stack:** Jekyll/Liquid, Ruby/Minitest, GitHub Actions, Prettier.

---

### Task 1: Specify generated landing-page behavior test-first

**Files:**

- Modify: `test/test_taxonomy_validator.rb`
- Modify: `scripts/validate_taxonomy.rb`
- Modify: `_data/taxonomy.yml`

- [ ] **Step 1: Add a failing generated-page test**

Add `landing_path: /blog/` to the temporary taxonomy fixture. Assert that a missing landing page is reported, a `.tag-category-list` containing `/tag/` is rejected, and a category-only block passes.

```ruby
landing.write('<div class="tag-category-list"><a href="/blog/tag/one/">One</a></div>')
assert_includes validator.validate_generated(site_dir: site_dir).join("\n"), "landing taxonomy must not display tag links"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby -Itest test/test_taxonomy_validator.rb`

Expected: failure because generated landing-page taxonomy is not validated.

- [ ] **Step 3: Implement generated landing validation**

For each collection, load `<site_dir>/<landing_path>/index.html`, isolate `.tag-category-list`, require at least one category link, and reject `/tag/` links or `fa-hashtag` icons inside that block.

- [ ] **Step 4: Record real landing paths**

Add `/programming/`, `/AI/`, `/math/`, and `/reading/` to the four collection entries in `_data/taxonomy.yml`.

- [ ] **Step 5: Run the validator tests and verify GREEN**

Run: `PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby -Itest test/test_taxonomy_validator.rb`

Expected: all tests pass.

### Task 2: Render categories only and verify retained tag behavior

**Files:**

- Modify: `_includes/collection_taxonomy.liquid`

- [ ] **Step 1: Remove the tag rendering loop**

Keep the category loop, tag icon, category archive URL, separators between categories, and surrounding `.tag-category-list`. Remove the separator between categories and tags plus the complete tag loop.

- [ ] **Step 2: Format the changed files**

Run: `npx prettier _data/taxonomy.yml _includes/collection_taxonomy.liquid docs/superpowers/plans/2026-08-07-category-only-taxonomy-display.md --write`

Expected: command exits 0.

- [ ] **Step 3: Build and validate generated pages**

Run: `PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec jekyll build`

Run: `PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby scripts/validate_taxonomy.rb --site _site`

Expected: source metadata, all category/tag archives, search keywords, and all four category-only landing strips pass.

- [ ] **Step 4: Run regression checks**

Run:

```bash
PATH=/opt/homebrew/opt/ruby/bin:$PATH bundle exec ruby -Itest test/test_taxonomy_validator.rb
PATH=/opt/homebrew/opt/ruby/bin:$PATH python3 -m unittest discover -s test -p 'test_*.py'
npx prettier . --check
git diff --check
```

Expected: all commands exit 0. User-owned `_pages/about.md` and staged reading changes remain untouched.
