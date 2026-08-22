# Medium Zoom Layering Fix

## Problem

On production pages that combine a sidebar table of contents with zoomable images, the sidebar text appears above an opened image. The problem is site-wide and can be reproduced on `/AI/2026-08-08-distribute-aigc.html`.

Two production-only transformations cause the behavior:

1. PurgeCSS removes the custom `.medium-zoom-overlay` and `.medium-zoom-image--opened` `z-index` rules because those classes are added at runtime and do not exist in the generated HTML.
2. CSS minification shortens `--global-bg-color` from `#ffffff` to `#fff`. Appending `ee` in `zoom.js` produces the invalid color `#fffee`, leaving the zoom overlay transparent.

The resulting production state has an opened image and overlay with `z-index: auto`, while `#toc-sidebar` retains `z-index: 1`.

## Design

Keep the existing Medium Zoom integration and fix its production inputs:

- Safelist the runtime Medium Zoom classes in `purgecss.config.js` so the site-owned stacking rules survive production pruning.
- Give the overlay the computed theme background color directly instead of constructing an eight-digit hex value by string concatenation. The overlay will be opaque in both light and dark themes.
- Retain the current layer values: the opened image and overlay use `z-index: 999`, while the sidebar remains at `z-index: 1`.

This is preferred over hiding the sidebar during zoom because it fixes the shared modal layering contract and also prevents other page content from showing through the overlay.

## Scope

The change applies to all pages that use `data-zoomable` images. No article-specific markup or image assets will change.

## Verification

Add a regression check that builds the site in production mode, runs PurgeCSS, and asserts that:

- the built stylesheet still contains the Medium Zoom stacking selectors and their `z-index`;
- the zoom initialization no longer appends alpha text to a possibly shortened hex color.

Then serve the production output and verify in a desktop viewport that opening an image on `/AI/2026-08-08-distribute-aigc.html` places the image above the sidebar and renders an opaque theme-colored overlay. Repeat in light and dark themes if the browser state permits.

## Non-goals

- Replacing Medium Zoom with another lightbox library.
- Changing sidebar layout or navigation behavior.
- Editing individual blog posts or SVG files.
