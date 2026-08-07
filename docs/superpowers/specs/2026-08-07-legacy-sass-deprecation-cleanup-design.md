# Legacy Sass Deprecation Cleanup Design

## Goal

Eliminate all Dart Sass deprecation warnings from the legacy `DamonsJ.github.io` build while preserving the site's rendered layout, icons, dark mode, routes, and content.

## Root Causes

The legacy stylesheet entrypoint still uses Sass `@import`, first-party theme code calls deprecated global color functions, and vendored Font Awesome 6.7.1 and Tabler Icons Sass sources use additional deprecated global functions and legacy imports. The repository also ignores `Gemfile.lock`, allowing local installs to resolve newer Sass compilers without a reproducible dependency baseline.

## Approved Architecture

The build will be split into two stable layers:

1. **First-party site Sass** continues to compile through Jekyll into `assets/css/main.css`, but is migrated to the Sass module system with `@use`.
2. **Third-party icon styles** are frozen as precompiled CSS in `assets/css/vendor-icons.css` and loaded immediately after `main.css`. The corresponding legacy Font Awesome and Tabler Sass source directories are removed after the compiled artifact is verified.

This preserves the existing cascade order: site styles first, icon-library rules second. Existing font files and public font URLs remain unchanged.

## First-Party Sass Migration

- Replace the entrypoint's aggregate `@import` block with ordered `@use` statements.
- Configure `variables` once from the entrypoint using the existing Jekyll `site.max_width` value.
- Add explicit `@use "variables" as *` declarations to first-party partials that consume shared variables.
- Use namespaced built-in modules such as `sass:color`, `sass:math`, and `sass:string` where required.
- Replace `red()`, `green()`, and `blue()` calls with modern color APIs while retaining the same RGBA values.
- Preserve emitted selector order and avoid unrelated Sass refactoring.

## Vendor Icon CSS

- Compile the currently vendored Font Awesome 6.7.1 and Tabler Icons sources once into a single CSS artifact.
- Verify that its `@font-face` declarations still reference the existing files below `assets/webfonts`.
- Add the vendor stylesheet link directly after `main.css` in `_includes/head.liquid`.
- Verify representative solid, regular, brand, Tabler outline, and Tabler filled icons before removing the obsolete Sass source directories.
- Do not introduce a runtime CDN dependency.

## Dependency Reproducibility

Remove `Gemfile.lock` from `.gitignore` and commit the resolved lockfile. The fix must work with the currently resolved `jekyll-sass-converter 3.1.0` and `sass-embedded 1.99.0`; it must not rely on downgrading or suppressing deprecations.

## Error and Compatibility Handling

- If a module reports an undefined variable, add an explicit dependency to that partial rather than restoring a global import.
- If compiled icon URLs do not resolve, correct the vendor CSS URL relative to `assets/css/vendor-icons.css`; do not duplicate font files under a second path.
- If the converted stylesheet changes key computed styles, stop and reconcile module order or variable configuration before proceeding.
- No warning-suppression setting may be used to satisfy acceptance.

## Verification

1. Capture the pre-change generated site and key computed styles for the homepage and representative article pages.
2. Add a regression test that fails while the source contains first-party `@import`, deprecated color calls, ignored `Gemfile.lock`, or active vendored icon Sass imports.
3. Build with verbose Sass output and assert that neither `Deprecation Warning` nor `repetitive deprecation warnings omitted` appears.
4. Run existing Jekyll tests and validate all generated local links and assets.
5. Compare key computed styles before and after: theme colors, typography, container width, navbar, footer, profile image, article text, and code blocks.
6. Browser-check light and dark themes plus representative Font Awesome and Tabler icons.
7. Confirm that article and page content files are unchanged.

## Scope

Expected changes are limited to the Sass entrypoint and first-party partials, the head stylesheet include, the precompiled vendor icon CSS, removal of obsolete vendor Sass sources, dependency locking, and regression tests or build-check scripts. Blog posts, collections, pages, navigation structure, JavaScript behavior, images, and personal content remain untouched.
