# Legacy Sass Deprecation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the legacy site build with Dart Sass 1.99 without deprecation warnings while preserving layout, theme behavior, icons, assets, routes, and content.

**Architecture:** Convert only first-party Sass to the module system. Compile the exact vendored Font Awesome and Tabler sources once into a static local stylesheet, remove those deprecated Sass sources from Jekyll's compilation path, and track the Ruby lockfile for reproducibility.

**Tech Stack:** Jekyll, Dart Sass via `sass-embedded 1.99.0`, SCSS modules, Python `unittest`, local browser validation

---

### Task 1: Capture the Legacy Baseline

**Files:**

- Verify: existing generated site under `_site/`
- Create outside repository: `/tmp/damons-sass-baseline/`

- [ ] **Step 1: Build the unmodified legacy site and save its warning log**

Run:

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
mkdir -p /tmp/damons-sass-baseline
JEKYLL_ENV=production bundle exec jekyll build --trace > /tmp/damons-sass-baseline/build.log 2>&1
```

Expected: exit code 0 and `build.log` contains `Deprecation Warning`.

- [ ] **Step 2: Save the generated CSS and content hashes**

Run:

```bash
cp _site/assets/css/main.css /tmp/damons-sass-baseline/main.css
find _AI _math _posts _reading _pages -type f -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/damons-sass-baseline/content.sha256
```

Expected: both baseline files exist and are non-empty.

### Task 2: Add Failing Modernization Tests

**Files:**

- Create: `test/test_sass_modernization.py`

- [ ] **Step 1: Add source and integration acceptance tests**

Create `test/test_sass_modernization.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "assets/css/main.scss"
VENDOR_CSS = ROOT / "assets/css/vendor-icons.css"


class SassModernizationTests(unittest.TestCase):
    def test_first_party_entrypoint_uses_modules(self) -> None:
        source = MAIN.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*@import\b")
        for module in ("variables", "themes", "layout", "base", "distill", "cv", "tabs", "typograms"):
            self.assertIn(f'@use "{module}"', source)

    def test_first_party_sass_has_no_deprecated_calls(self) -> None:
        offenders: list[str] = []
        patterns = (
            re.compile(r"(?m)^\s*@import\b"),
            re.compile(r"(?<![.\w-])(?:red|green|blue)\("),
        )
        for path in sorted((ROOT / "_sass").glob("*.scss")):
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in patterns):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_vendor_icons_are_static_and_wired_after_main(self) -> None:
        self.assertTrue(VENDOR_CSS.is_file())
        css = VENDOR_CSS.read_text(encoding="utf-8")
        for marker in ("Font Awesome 6 Free", "Font Awesome 6 Brands", "tabler-icons", "tabler-icons-filled"):
            self.assertIn(marker, css)
        head = (ROOT / "_includes/head.liquid").read_text(encoding="utf-8")
        main_link = head.index("/assets/css/main.css")
        vendor_link = head.index("/assets/css/vendor-icons.css")
        self.assertGreater(vendor_link, main_link)
        self.assertFalse((ROOT / "_sass/font-awesome").exists())
        self.assertFalse((ROOT / "_sass/tabler-icons").exists())

    def test_font_assets_exist(self) -> None:
        for relative in (
            "assets/webfonts/fa-brands-400.woff2",
            "assets/webfonts/fa-regular-400.woff2",
            "assets/webfonts/fa-solid-900.woff2",
            "assets/fonts/tabler-icons.woff2",
            "assets/fonts/tabler-icons-outline.woff2",
            "assets/fonts/tabler-icons-filled.woff2",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_lockfile_is_tracked_by_policy(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertNotIn("Gemfile.lock", ignored)
        self.assertTrue((ROOT / "Gemfile.lock").is_file())
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "Gemfile.lock"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(tracked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest test.test_sass_modernization -v
```

Expected: failures for legacy `@import`, deprecated color calls, missing vendor CSS, existing vendor Sass directories, and ignored/untracked `Gemfile.lock`.

### Task 3: Freeze the Existing Icon Libraries as CSS

**Files:**

- Create: `assets/css/vendor-icons.css`
- Modify: `_includes/head.liquid`
- Delete: `_sass/font-awesome/`
- Delete: `_sass/tabler-icons/`

- [ ] **Step 1: Create a temporary compilation wrapper outside the repository**

Use `apply_patch` to create `/tmp/damons-vendor-icons.scss` with:

```scss
@use "sass:meta";

$fa-font-path: "../webfonts";

@import "font-awesome/fontawesome";
@import "font-awesome/brands";
@import "font-awesome/solid";
@import "font-awesome/regular";

@include meta.load-css("tabler-icons/tabler-icons", $with: ("ti-font-path": "../fonts"));
@include meta.load-css("tabler-icons/tabler-icons-filled", $with: ("ti-font-path": "../fonts"));
@include meta.load-css("tabler-icons/tabler-icons-outline", $with: ("ti-font-path": "../fonts"));
```

- [ ] **Step 2: Compile the exact vendored versions once**

Run:

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
bundle exec sass --load-path _sass --style=compressed --no-source-map /tmp/damons-vendor-icons.scss assets/css/vendor-icons.css
```

Expected: exit code 0; this one-time conversion may print the legacy warnings, and the output contains Font Awesome 6.7.1 and Tabler Icons 3.20.0 rules.

- [ ] **Step 3: Link vendor CSS after main CSS**

Add immediately after the existing main stylesheet link in `_includes/head.liquid`:

```liquid
<link rel="stylesheet" href="{{ '/assets/css/main.css' | relative_url | bust_css_cache }}">
<link rel="stylesheet" href="{{ '/assets/css/vendor-icons.css' | relative_url | bust_file_cache }}">
```

- [ ] **Step 4: Remove obsolete icon Sass source directories**

Delete only `_sass/font-awesome/` and `_sass/tabler-icons/` after confirming `vendor-icons.css` contains all four marker strings from the test.

- [ ] **Step 5: Run the vendor and font tests**

Run:

```bash
python3 -m unittest \
  test.test_sass_modernization.SassModernizationTests.test_vendor_icons_are_static_and_wired_after_main \
  test.test_sass_modernization.SassModernizationTests.test_font_assets_exist -v
```

Expected: both tests PASS.

### Task 4: Convert First-Party Sass to Modules

**Files:**

- Modify: `assets/css/main.scss`
- Modify: `_sass/_variables.scss`
- Modify: `_sass/_themes.scss`
- Modify: `_sass/_layout.scss`

- [ ] **Step 1: Define the configurable content width**

Add to `_sass/_variables.scss` before the color variables:

```scss
$max-content-width: 930px !default;
```

Replace the code background declaration with:

```scss
$code-bg-color-light: color.change($google-blue-color, $alpha: 0.05);
```

- [ ] **Step 2: Add explicit variable dependencies**

At the top of `_sass/_themes.scss`, after its header comment, use:

```scss
@use "sass:color";
@use "variables" as *;
```

At the top of `_sass/_layout.scss`, after its header comment, use:

```scss
@use "variables" as *;
```

- [ ] **Step 3: Replace deprecated channel extraction**

In `_sass/_themes.scss`, replace the two channel-based values with:

```scss
--global-back-to-top-bg-color: #{color.change($black-color, $alpha: 0.4)};
```

and:

```scss
--global-back-to-top-bg-color: #{color.change($white-color, $alpha: 0.5)};
```

- [ ] **Step 4: Replace the legacy entrypoint**

Keep the front matter and charset, then use:

```scss
@use "variables" with (
  $max-content-width: {{site.max_width | default: "930px"}}
);
@use "themes";
@use "layout";
@use "base";
@use "distill";
@use "cv";
@use "tabs";
@use "typograms";
```

Do not include icon modules in `main.scss`.

- [ ] **Step 5: Run first-party source tests**

Run:

```bash
python3 -m unittest \
  test.test_sass_modernization.SassModernizationTests.test_first_party_entrypoint_uses_modules \
  test.test_sass_modernization.SassModernizationTests.test_first_party_sass_has_no_deprecated_calls -v
```

Expected: both tests PASS.

### Task 5: Lock the Ruby Dependency Graph

**Files:**

- Modify: `.gitignore`
- Create/track: `Gemfile.lock`

- [ ] **Step 1: Stop ignoring the lockfile**

Remove only the `Gemfile.lock` line from `.gitignore`.

- [ ] **Step 2: Verify the resolved versions**

Run:

```bash
rg -n "jekyll-sass-converter \(3\.1\.0\)|sass-embedded \(1\.99\.0-" Gemfile.lock
```

Expected: matches for the converter and platform-specific Sass packages.

- [ ] **Step 3: Stage the lockfile so the tracking test reflects repository policy**

Run:

```bash
git add Gemfile.lock .gitignore
python3 -m unittest test.test_sass_modernization.SassModernizationTests.test_lockfile_is_tracked_by_policy -v
```

Expected: PASS.

### Task 6: Prove the Build Is Warning-Free and Compatible

**Files:**

- Verify: all modified files
- Verify outside repository: `/tmp/damons-sass-after.log`

- [ ] **Step 1: Build with complete logs**

Run:

```bash
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
JEKYLL_ENV=production bundle exec jekyll build --trace > /tmp/damons-sass-after.log 2>&1
```

Expected: exit code 0.

- [ ] **Step 2: Assert there are no Sass deprecation warnings**

Run:

```bash
if rg -n "Deprecation Warning|repetitive deprecation warnings omitted" /tmp/damons-sass-after.log; then exit 1; fi
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run the complete modernization suite**

Run:

```bash
python3 -m unittest discover -s test -p "test_*.py" -v
```

Expected: all tests PASS.

- [ ] **Step 4: Verify content hashes are unchanged**

Run:

```bash
find _AI _math _posts _reading _pages -type f -print0 | sort -z | xargs -0 shasum -a 256 > /tmp/damons-sass-baseline/content-after.sha256
diff -u /tmp/damons-sass-baseline/content.sha256 /tmp/damons-sass-baseline/content-after.sha256
```

Expected: no diff.

- [ ] **Step 5: Verify generated CSS and local assets**

Confirm `_site/index.html` loads both CSS files, all URLs referenced by `vendor-icons.css` resolve under `_site/assets`, and representative HTML still contains Font Awesome and Tabler icon classes.

### Task 7: Browser Regression, Formatting, and Commit

**Files:**

- Verify: homepage and representative article pages
- Commit: all approved changes

- [ ] **Step 1: Browser-check light mode**

Verify homepage container width, navbar, footer, profile image, theme colors, Font Awesome social icons, Tabler search icon, and Tabler theme icon.

- [ ] **Step 2: Browser-check dark mode**

Toggle dark mode and verify background, text, footer, navigation, and icons remain visible.

- [ ] **Step 3: Browser-check a representative article**

Open an AI article and verify typography, code blocks, figures, sidebar/table-of-contents behavior, and Font Awesome metadata icons.

- [ ] **Step 4: Run formatting and diff checks**

Run:

```bash
npx prettier . --check
git diff --check
```

Expected: both commands pass.

- [ ] **Step 5: Review and commit**

Verify no content file appears in `git status`, then run:

```bash
git add .gitignore Gemfile.lock assets/css/main.scss assets/css/vendor-icons.css _includes/head.liquid _sass test docs/superpowers/plans/2026-08-07-legacy-sass-deprecation-cleanup.md
git commit -m "fix: modernize legacy Sass build"
```

- [ ] **Step 6: Confirm clean branch**

Run:

```bash
git status --short --branch
git log -3 --oneline
```

Expected: `fix/sass-deprecations` is clean and the latest commit is the Sass modernization fix.
