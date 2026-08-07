# Blog Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all 24 entries to a controlled taxonomy with valid archive links, searchable metadata, and automated local/CI validation while preserving article bodies and URLs.

**Architecture:** `_data/taxonomy.yml` is the canonical collection-scoped vocabulary. A Ruby validator checks source front matter and generated Jekyll output; a shared Liquid include renders the vocabulary, and Ninja Keys receives a dedicated `keywords` field for taxonomy search.

**Tech Stack:** Jekyll/Liquid, jekyll-archives-v2, Ruby/YAML/Minitest, Ninja Keys 1.2.11, GitHub Actions, Prettier.

---

### Task 1: Define validator behavior test-first

**Files:**

- Create: `test/test_taxonomy_validator.rb`
- Create: `scripts/validate_taxonomy.rb`

- [ ] **Step 1: Write failing Minitest cases**

Create fixtures in a temporary repository and assert that validation reports: non-array or wrong-count categories/tags, terms outside the controlled vocabulary, duplicate tags, `转载` as a tag, invalid `source_type`, and unused vocabulary. Add a valid fixture that has one category, two tags, and optional `source_type: repost`.

```ruby
errors = TaxonomyValidator.new(root: fixture_root).validate_source
assert_includes errors.join("\n"), "exactly one category"
assert_includes errors.join("\n"), "between 2 and 4 tags"
assert_includes errors.join("\n"), "not in the controlled vocabulary"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `ruby -Itest test/test_taxonomy_validator.rb`

Expected: failure because `scripts/validate_taxonomy.rb` does not exist.

- [ ] **Step 3: Implement the source validator**

Implement `TaxonomyValidator` using Ruby standard YAML plus `Jekyll::Utils.slugify`. Parse only YAML between the first two `---` delimiters, validate every `*.md` in the configured collection directory, aggregate file-specific errors, and exit nonzero from the CLI when errors exist.

- [ ] **Step 4: Run the validator unit tests and verify GREEN**

Run: `bundle exec ruby -Itest test/test_taxonomy_validator.rb`

Expected: all tests pass.

### Task 2: Install the controlled vocabulary and migrate front matter

**Files:**

- Create: `_data/taxonomy.yml`
- Modify: `_config.yml`
- Modify: `_AI/*.md`
- Modify: `_posts/*.md`
- Modify: `_math/*.md`
- Modify: `_reading/*.md`

- [ ] **Step 1: Run the new validator against current metadata and verify RED**

Run: `bundle exec ruby scripts/validate_taxonomy.rb`

Expected: failure for repeated collection categories, old tags, and missing controlled metadata.

- [ ] **Step 2: Add the canonical taxonomy**

Create `_data/taxonomy.yml` with four entries under `collections`. Each entry contains `directory`, `archive_prefix`, the exact category list, and the exact tag list from `docs/superpowers/specs/2026-08-07-blog-taxonomy-design.md`. Add `source_types: [repost]`.

- [ ] **Step 3: Migrate all entry front matter**

For each of the 24 entries, replace only `categories` and `tags` according to the design table. Use YAML arrays, for example:

```yaml
categories: [GPU 与高性能计算]
tags: [BERT, CUDA, 源码解读]
```

Add `source_type: repost` only to the five entries marked repost. Add a `date` matching `finished` to the four reading entries so their displayed year links match generated year archives. Do not change content after the closing front-matter delimiter.

- [ ] **Step 4: Remove legacy display arrays**

Delete `programming_tags`, `math_tags`, `AI_tags`, and commented `display_categories` examples from `_config.yml`; display data now comes from `_data/taxonomy.yml`.

- [ ] **Step 5: Run source validation and verify GREEN**

Run: `bundle exec ruby scripts/validate_taxonomy.rb`

Expected: `Taxonomy source validation passed (24 entries).`

### Task 3: Render active taxonomy and correct archive links

**Files:**

- Create: `_includes/collection_taxonomy.liquid`
- Modify: `_pages/programming.md`
- Modify: `_pages/AI.md`
- Modify: `_pages/math.md`
- Modify: `_pages/reading.md`
- Modify: `_layouts/post.liquid`
- Modify: `_layouts/book-review.liquid`
- Modify: `_layouts/book-shelf.liquid`
- Modify: `test/test_taxonomy_validator.rb`

- [ ] **Step 1: Add failing generated-archive tests**

Extend the test fixture to create `_site/<prefix>/tag/<slug>/index.html` and category pages. Assert missing pages are reported and complete pages pass.

- [ ] **Step 2: Run the test and verify RED**

Run: `bundle exec ruby -Itest test/test_taxonomy_validator.rb`

Expected: failure because generated archive validation is not implemented.

- [ ] **Step 3: Implement generated archive validation**

For every assigned term, compute the slug with `Jekyll::Utils.slugify` and require `_site/<archive_prefix>/<type>/<slug>/index.html`.

- [ ] **Step 4: Add the shared taxonomy include**

Render categories followed by tags from `site.data.taxonomy.collections[include.collection]`. Build URLs as `<archive_prefix>category/<slug>/` and `<archive_prefix>tag/<slug>/`, and pass them through `relative_url`.

- [ ] **Step 5: Replace landing-page hard-coded lists**

Use exactly one include per landing page:

```liquid
{% include collection_taxonomy.liquid collection="posts" %}
```

Use `AI`, `math`, and `reading` for the other three pages.

- [ ] **Step 6: Fix entry and shelf links**

In `_layouts/post.liquid`, construct `col_prefix` first and then `<prefix>tag/<slug>/` or `<prefix>category/<slug>/`. In reading layouts use `/reading/`, not `/books/`; build reading year links as `/reading/<year>/`.

- [ ] **Step 7: Build and verify archives**

Run: `bundle exec jekyll build && bundle exec ruby scripts/validate_taxonomy.rb --site _site`

Expected: build succeeds and no archive page is missing.

### Task 4: Make categories and tags searchable

**Files:**

- Modify: `_scripts/search.liquid.js`
- Modify: `test/test_taxonomy_validator.rb`

- [ ] **Step 1: Add failing search-output tests**

Require every generated article action to contain a `taxonomySource` matching its source path and a `keywords` value containing its category, all tags, and `转载` for reposts.

- [ ] **Step 2: Run the test and verify RED**

Run: `bundle exec ruby -Itest test/test_taxonomy_validator.rb`

Expected: failure because search output has no taxonomy metadata.

- [ ] **Step 3: Generate search keywords**

For posts and all non-post collection documents, add taxonomy metadata and append it to the generated action description:

```liquid
taxonomySource: "{{ item.path }}", keywords: "{{ item.categories | join: ' ' }}
{{ item.tags | join: ' ' -}}
{%- if item.source_type == 'repost' %} 转载{% endif %}", description: "{{ item.description }}
{{ item.categories | join: ' ' }}
{{ item.tags | join: ' ' -}}
{%- if item.source_type == 'repost' %} 转载{% endif %}",
```

Use `post` rather than `item` in the posts loop.

- [ ] **Step 4: Include taxonomy in Ninja Keys scoring**

Keep the dedicated `keywords` field for validation and append the same terms to each generated action description. Ninja Keys 1.2.11 already scores descriptions but renders only titles, so taxonomy becomes searchable without modifying the vendored library or visible search result.

- [ ] **Step 5: Build and verify search output**

Run: `bundle exec jekyll build && bundle exec ruby scripts/validate_taxonomy.rb --site _site`

Expected: every article passes its generated keyword check.

### Task 5: Enforce the taxonomy in CI and complete verification

**Files:**

- Create: `.github/workflows/taxonomy.yml`
- Modify: `docs/superpowers/specs/2026-08-07-blog-taxonomy-design.md`

- [ ] **Step 1: Add the taxonomy workflow**

On pushes and pull requests affecting content, taxonomy, layouts, scripts, or the validator: check out, set up Ruby 3.2.2 with Bundler cache, run source validation, build Jekyll, then run generated-output validation.

- [ ] **Step 2: Format supported files**

Run: `npx prettier _config.yml _data/taxonomy.yml _includes/collection_taxonomy.liquid _pages/programming.md _pages/AI.md _pages/math.md _pages/reading.md _layouts/post.liquid _layouts/book-review.liquid _layouts/book-shelf.liquid _scripts/search.liquid.js .github/workflows/taxonomy.yml docs/superpowers/specs/2026-08-07-blog-taxonomy-design.md docs/superpowers/plans/2026-08-07-blog-taxonomy-implementation.md --write`

Expected: command exits 0.

- [ ] **Step 3: Verify article bodies are unchanged**

Compare every current content file with its version at commit `a89b4a7`, stripping the first YAML front-matter block from both sides. Expected: 24 matches and no missing file.

- [ ] **Step 4: Run the full verification suite**

Run:

```bash
bundle exec ruby -Itest test/test_taxonomy_validator.rb
python3 -m unittest discover -s test -p 'test_*.py'
bundle exec ruby scripts/validate_taxonomy.rb
bundle exec jekyll build
bundle exec ruby scripts/validate_taxonomy.rb --site _site
npx prettier . --check
git diff --check
```

Expected: every command exits 0, all tests pass, all archives/search terms validate, and Prettier reports all files formatted.
