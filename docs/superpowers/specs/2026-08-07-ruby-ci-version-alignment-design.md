# Ruby CI Version Alignment Design

## Goal

Restore GitHub Actions dependency installation by making every Ruby-based workflow use a Ruby version compatible with the locked bundle.

## Root cause

`Gemfile.lock` resolves `css_parser` to 2.0.0, which requires Ruby 3.3 or newer. The deployment workflow already uses Ruby 3.3.5, but the taxonomy, accessibility, and deployed-site link-check workflows still use Ruby 3.2.2. `ruby/setup-ruby` therefore fails during its Bundler cache installation step before any project command runs.

## Approaches considered

1. **Align workflows on Ruby 3.3.5 — selected.** This matches the existing deployment workflow and the current al-folio workflow baseline. It preserves the resolved dependency graph and removes environment drift between CI jobs.
2. **Pin `css_parser` below 2.0.** This would retain Ruby 3.2.2 but unnecessarily changes application dependencies and leaves deployment and validation jobs on different Ruby versions.
3. **Move all workflows to a newer Ruby release.** Ruby 3.4 is not required by the current dependency failure and introduces a larger compatibility change than necessary.

## Design

- Change `ruby-version` from 3.2.2 to 3.3.5 in `.github/workflows/taxonomy.yml`, `.github/workflows/axe.yml`, and `.github/workflows/broken-links-site.yml`.
- Leave `.github/workflows/deploy.yml` unchanged because it already uses Ruby 3.3.5.
- Leave `Gemfile`, `Gemfile.lock`, and Bundler 2.6.9 unchanged.
- Extend `test/test_ci_workflows.py` to scan every workflow containing `ruby/setup-ruby@v1` and require `ruby-version: "3.3.5"`. The test reports the offending workflow path if a different version is introduced later.

## Verification

1. Run the new workflow-version test before the fix and confirm it fails on the three Ruby 3.2.2 workflows.
2. Update the workflows and confirm the test passes.
3. Run the complete Python and Ruby test suites.
4. Run Prettier across the repository and `git diff --check`.
5. Run the local Jekyll build with the installed compatible Ruby environment.
6. The next GitHub Actions run is the final remote-environment confirmation that `ruby/setup-ruby` completes its Bundler installation.

## Out of scope

- Changing Ruby dependencies or regenerating `Gemfile.lock`.
- Upgrading Bundler.
- Changing deployment behavior beyond Ruby runtime alignment.
