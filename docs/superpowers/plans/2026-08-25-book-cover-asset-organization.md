# Book Cover Asset Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every `assets/img/book_covers` image into its owning blog directory, move the shared Douban icon into `shared`, remove the unused Godfather cover, and update all source references.

**Architecture:** Article image directories use the corresponding `_reading` Markdown basename as their stable identifier. A repository test enforces two invariants: no image remains directly under `book_covers`, and every source reference to a book-cover asset resolves to an existing file.

**Tech Stack:** Jekyll, Liquid, Markdown front matter, Ruby/Minitest

---

### Task 1: Add the organization regression test

**Files:**
- Create: `test/test_book_cover_organization.rb`

- [ ] **Step 1: Write the failing test**

```ruby
# frozen_string_literal: true

require "minitest/autorun"

class BookCoverOrganizationTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)
  BOOK_COVERS = File.join(ROOT, "assets/img/book_covers")
  IMAGE_PATTERN = "*.{jpg,jpeg,png,webp,svg}"
  REFERENCE_PATTERN = %r{(?:\.\./|/)?(assets/img/book_covers/[A-Za-z0-9_.\-/]+\.(?:jpg|jpeg|png|webp|svg))}

  def test_images_are_grouped_below_blog_or_shared_directories
    root_images = Dir.glob(File.join(BOOK_COVERS, IMAGE_PATTERN), File::FNM_EXTGLOB)

    assert_empty root_images, "Move root-level images into a blog directory or shared/"
  end

  def test_every_book_cover_reference_resolves_to_a_file
    source_files = Dir.glob(File.join(ROOT, "_reading/*.md")) +
      [File.join(ROOT, "_layouts/book-review.liquid")]
    references = source_files.flat_map do |source_file|
      File.read(source_file).scan(REFERENCE_PATTERN).flatten
    end.uniq

    refute_empty references
    references.each do |relative_path|
      assert File.file?(File.join(ROOT, relative_path)), "Missing referenced asset: #{relative_path}"
    end
  end
end
```

- [ ] **Step 2: Run the focused test and verify the organization check fails**

Run:

```bash
/opt/homebrew/opt/ruby/bin/bundle exec /opt/homebrew/opt/ruby/bin/ruby test/test_book_cover_organization.rb
```

Expected: the root-level image test fails and lists the current images under `assets/img/book_covers`.

### Task 2: Move assets and update their owners

**Files:**
- Move: `assets/img/book_covers/before-the-path-to-the-spiders-nests.png` → `assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/spider-nest-italy-1943-timeline.png` → `assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/spider-nest-italy-september-1943.png` → `assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/spider-nest-partisans-liguria.jpg` → `assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/spider-nest-literary-roots.png` → `assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/the-path-to-the-spiders-nests.jpg` → `assets/img/book_covers/the-path-to-the-spiders-nests/`
- Move: `assets/img/book_covers/italo_calvino_history_today.webp` → `assets/img/book_covers/calvino-traveller-uncertainty/`
- Move: `assets/img/book_covers/invisible-cities.jpg` → `assets/img/book_covers/invisible-cites/`
- Move: `assets/img/book_covers/palomar.jpg` → `assets/img/book_covers/palomar/`
- Move: `assets/img/book_covers/douban.svg` → `assets/img/book_covers/shared/`
- Delete: `assets/img/book_covers/the_godfather.jpg`
- Modify: `_reading/before-reading-the-path-to-the-spiders-nests.md`
- Modify: `_reading/the-path-to-the-spiders-nests.md`
- Modify: `_reading/calvino-traveller-uncertainty.md`
- Modify: `_reading/invisible-cites.md`
- Modify: `_reading/palomar.md`
- Modify: `_layouts/book-review.liquid`

- [ ] **Step 1: Create the destination directories**

```bash
mkdir -p \
  assets/img/book_covers/before-reading-the-path-to-the-spiders-nests \
  assets/img/book_covers/the-path-to-the-spiders-nests \
  assets/img/book_covers/calvino-traveller-uncertainty \
  assets/img/book_covers/invisible-cites \
  assets/img/book_covers/palomar \
  assets/img/book_covers/shared
```

- [ ] **Step 2: Move the images without renaming them**

Move each file according to the mapping in the Files section. Do not modify image bytes.

- [ ] **Step 3: Delete the explicitly rejected orphan**

Delete only `assets/img/book_covers/the_godfather.jpg` after confirming it is still unreferenced with:

```bash
rg -n --hidden --glob '!.git/**' 'the_godfather\.jpg' .
```

Expected: no matches.

- [ ] **Step 4: Update the five cover paths**

Use these exact values:

```yaml
cover: assets/img/book_covers/before-reading-the-path-to-the-spiders-nests/before-the-path-to-the-spiders-nests.png
cover: assets/img/book_covers/the-path-to-the-spiders-nests/the-path-to-the-spiders-nests.jpg
cover: assets/img/book_covers/calvino-traveller-uncertainty/italo_calvino_history_today.webp
cover: assets/img/book_covers/invisible-cites/invisible-cities.jpg
cover: assets/img/book_covers/palomar/palomar.jpg
```

- [ ] **Step 5: Update inline and shared references**

Prefix each `spider-nest-*` inline image with `before-reading-the-path-to-the-spiders-nests/`. Change the layout icon path to:

```liquid
src="../assets/img/book_covers/shared/douban.svg"
```

- [ ] **Step 6: Run the focused test and verify it passes**

Run:

```bash
/opt/homebrew/opt/ruby/bin/bundle exec /opt/homebrew/opt/ruby/bin/ruby test/test_book_cover_organization.rb
```

Expected: all organization and reference assertions pass.

### Task 3: Verify the complete site

**Files:**
- Verify all files changed in Tasks 1 and 2.

- [ ] **Step 1: Check for stale root-level references and files**

```bash
find assets/img/book_covers -maxdepth 1 -type f -print
rg -n 'assets/img/book_covers/(before-the-path|spider-nest|the-path|italo_calvino|invisible-cities|palomar\.jpg|douban\.svg|the_godfather)' _reading _layouts
```

Expected: `find` reports only `.DS_Store`; `rg` reports no stale paths.

- [ ] **Step 2: Run all repository tests**

```bash
/opt/homebrew/opt/ruby/bin/bundle exec /opt/homebrew/opt/ruby/bin/ruby -Itest -e 'Dir["test/test_*.rb"].sort.each { |file| require_relative file }'
```

Expected: zero failures and zero errors.

- [ ] **Step 3: Run a production site build**

```bash
/bin/zsh -lc 'export PATH=/Users/honglei.sun/Library/Python/3.9/bin:/opt/homebrew/opt/ruby/bin:$PATH; JEKYLL_ENV=production bundle exec jekyll build'
```

Expected: exit status 0 with the `_reading` entries and their image assets generated successfully.

- [ ] **Step 4: Inspect final status without committing overlapping user work**

```bash
git status --short
git diff --check
```

Do not create an implementation commit because `_reading/the-path-to-the-spiders-nests.md`, `_reading/before-reading-the-path-to-the-spiders-nests.md`, and its cover include pre-existing user changes. Leave the reorganized working tree intact for the user to review or commit with those related article changes.
