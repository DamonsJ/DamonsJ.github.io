# Medium Zoom Layering Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep zoomed images and their opaque overlay above the sidebar table of contents in production builds.

**Architecture:** Preserve runtime Medium Zoom selectors through PurgeCSS and pass the computed theme color to Medium Zoom without unsafe string concatenation. A focused Minitest regression test guards both production-build inputs.

**Tech Stack:** Jekyll, Sass, PurgeCSS, JavaScript, Ruby Minitest, Medium Zoom

---

### Task 1: Add a failing production-input regression test

**Files:**
- Create: `test/test_medium_zoom.rb`

- [ ] **Step 1: Write the failing test**

```ruby
# frozen_string_literal: true

require "minitest/autorun"

class MediumZoomTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)

  def test_runtime_layering_classes_are_safelisted
    config = File.read(File.join(ROOT, "purgecss.config.js"))

    assert_match(/safelist:/, config)
    assert_includes config, '"medium-zoom-overlay"'
    assert_includes config, '"medium-zoom-image--opened"'
  end

  def test_theme_background_is_not_modified_as_a_hex_string
    %w[assets/js/zoom.js assets/js/theme.js].each do |relative_path|
      source = File.read(File.join(ROOT, relative_path))

      refute_match(/getPropertyValue\("--global-bg-color"\)\s*\+\s*"ee"/, source, relative_path)
    end
  end
end
```

- [ ] **Step 2: Run the test to verify it fails for the production regression**

Run: `ruby test/test_medium_zoom.rb`

Expected: FAIL because `purgecss.config.js` has no safelist and both JavaScript files append `"ee"`.

### Task 2: Preserve stacking rules and use a valid opaque overlay color

**Files:**
- Modify: `purgecss.config.js:1-7`
- Modify: `assets/js/zoom.js:1-6`
- Modify: `assets/js/theme.js:78-83`

- [ ] **Step 1: Safelist the runtime classes**

Add the following property to `purgecss.config.js`:

```javascript
safelist: ["medium-zoom-overlay", "medium-zoom-image--opened"],
```

- [ ] **Step 2: Use the computed theme color directly during initialization**

Set the Medium Zoom option in `assets/js/zoom.js` to:

```javascript
background: getComputedStyle(document.documentElement).getPropertyValue("--global-bg-color"),
```

- [ ] **Step 3: Use the computed theme color directly during theme changes**

Set the Medium Zoom update option in `assets/js/theme.js` to:

```javascript
background: getComputedStyle(document.documentElement).getPropertyValue("--global-bg-color"),
```

- [ ] **Step 4: Run the focused and existing tests**

Run: `ruby test/test_medium_zoom.rb && ruby test/test_taxonomy_validator.rb`

Expected: both test files pass with zero failures.

### Task 3: Verify the production artifact and original browser symptom

**Files:**
- Verify generated output: `_site/assets/css/main.css`
- Verify page: `_site/AI/2026-08-08-distribute-aigc.html`

- [ ] **Step 1: Build in production mode**

Run: `JEKYLL_ENV=production bundle exec jekyll build`

Expected: exit status 0.

- [ ] **Step 2: Apply the same PurgeCSS step used by deployment**

Run: `npx purgecss -c purgecss.config.js`

Expected: exit status 0 and `_site/assets/css/main.css` still contains `.medium-zoom-overlay` and `.medium-zoom-image--opened` with `z-index: 999`.

- [ ] **Step 3: Verify the target page in a desktop browser**

Serve `_site` locally, open `/AI/2026-08-08-distribute-aigc.html`, and click a zoomable image.

Expected: the image and overlay compute to `z-index: 999`, the sidebar computes to `z-index: 1`, the overlay is opaque, and `elementsFromPoint` reports the opened image before the sidebar in overlapping coordinates.

- [ ] **Step 4: Run final repository checks**

Run: `ruby test/test_medium_zoom.rb && ruby test/test_taxonomy_validator.rb && git diff --check`

Expected: zero failures and no whitespace errors.

- [ ] **Step 5: Commit the implementation**

```bash
git add purgecss.config.js assets/js/zoom.js assets/js/theme.js test/test_medium_zoom.rb docs/superpowers/plans/2026-08-22-medium-zoom-layering.md
git commit -m "fix: keep zoomed images above sidebar"
```
