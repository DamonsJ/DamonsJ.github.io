# Book Cover Asset Organization Design

## Goal

Organize `assets/img/book_covers` by the blog entry that owns each image, update every corresponding reference, and keep shared layout assets separate from article-owned images.

## Directory Structure

Each article-owned directory uses the basename of its Markdown file so the asset path maps directly back to the source entry:

- `before-reading-the-path-to-the-spiders-nests/` contains its cover and four inline illustrations.
- `the-path-to-the-spiders-nests/` contains its cover.
- `calvino-traveller-uncertainty/` contains its cover.
- `invisible-cites/` contains its cover.
- `palomar/` contains its cover.
- `shared/` contains `douban.svg`, which is used by the shared book-review layout.

Image filenames remain unchanged. Only their parent directories change.

## Reference Updates

Update `cover` values in the five `_reading` entries. Update the four inline Liquid image URLs in `before-reading-the-path-to-the-spiders-nests.md`. Update `_layouts/book-review.liquid` to load `douban.svg` from `book_covers/shared`.

Delete `the_godfather.jpg` because no current source file references it and no corresponding blog entry exists.

## Safety

The affected Spider's Nest entries already contain uncommitted user edits. File moves and link changes must preserve all unrelated content and stage only the asset-organization changes when committing.

## Verification

- Confirm every referenced `book_covers` asset exists.
- Confirm no source reference uses one of the old root-level paths.
- Confirm `the_godfather.jpg` is absent.
- Run the repository test suite and a production Jekyll build to catch broken Liquid or asset paths.
