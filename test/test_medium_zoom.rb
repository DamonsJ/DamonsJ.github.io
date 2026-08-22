# frozen_string_literal: true

require "minitest/autorun"
require "fileutils"
require "tmpdir"
require_relative "../scripts/validate_medium_zoom"

class MediumZoomTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)

  def test_runtime_layering_classes_are_safelisted
    config = File.read(File.join(ROOT, "purgecss.config.js"))

    assert_match(/safelist:/, config)
    assert_includes config, '"medium-zoom-overlay"'
    assert_includes config, '"medium-zoom-image--opened"'
  end

  def test_theme_background_is_passed_directly_to_medium_zoom
    %w[assets/js/zoom.js assets/js/theme.js].each do |relative_path|
      source = File.read(File.join(ROOT, relative_path))

      assert_match(
        /background:\s*getComputedStyle\(document\.documentElement\)\.getPropertyValue\("--global-bg-color"\)/,
        source,
        relative_path
      )
      refute_match(/getPropertyValue\("--global-bg-color"\)\s*\+\s*"ee"/, source, relative_path)
    end
  end

  def test_validator_accepts_valid_production_artifacts
    with_site_artifacts do |site_dir|
      errors = MediumZoomValidator.new(site_dir: site_dir).validate

      assert_empty errors
    end
  end

  def test_validator_rejects_missing_layering_rule
    with_site_artifacts(css: ".medium-zoom-overlay{z-index:1}") do |site_dir|
      errors = MediumZoomValidator.new(site_dir: site_dir).validate

      assert errors.any? { |error| error.include?("z-index: 999") }
    end
  end

  def test_validator_rejects_modified_zoom_background
    invalid_js = 'mediumZoom("img",{background:getComputedStyle(document.documentElement).getPropertyValue("--global-bg-color")+"ee"})'

    with_site_artifacts(theme_js: invalid_js) do |site_dir|
      errors = MediumZoomValidator.new(site_dir: site_dir).validate

      assert errors.any? { |error| error.include?("theme.js") }
    end
  end

  private

  def with_site_artifacts(
    css: ".medium-zoom-overlay,.medium-zoom-image--opened{z-index:999}",
    zoom_js: valid_zoom_js,
    theme_js: valid_zoom_js
  )
    Dir.mktmpdir do |site_dir|
      FileUtils.mkdir_p(File.join(site_dir, "assets/css"))
      FileUtils.mkdir_p(File.join(site_dir, "assets/js"))
      File.write(File.join(site_dir, "assets/css/main.css"), css)
      File.write(File.join(site_dir, "assets/js/zoom.js"), zoom_js)
      File.write(File.join(site_dir, "assets/js/theme.js"), theme_js)
      yield site_dir
    end
  end

  def valid_zoom_js
    'mediumZoom("img",{background:getComputedStyle(document.documentElement).getPropertyValue("--global-bg-color")})'
  end
end
