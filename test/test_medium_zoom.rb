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
