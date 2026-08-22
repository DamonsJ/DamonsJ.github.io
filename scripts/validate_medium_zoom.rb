# frozen_string_literal: true

class MediumZoomValidator
  LAYERING_RULE = /\.medium-zoom-overlay\s*,\s*\.medium-zoom-image--opened\s*\{[^}]*z-index\s*:\s*999(?:;)?[^}]*\}/m
  BACKGROUND_RULE = /background\s*:\s*getComputedStyle\(document\.documentElement\)\.getPropertyValue\((["'])--global-bg-color\1\)(?!\s*\+)/

  def initialize(site_dir: "_site")
    @site_dir = site_dir
  end

  def validate
    errors = []
    validate_layering_rule(errors)
    validate_backgrounds(errors)
    errors
  end

  private

  def validate_layering_rule(errors)
    relative_path = "assets/css/main.css"
    css = read_artifact(relative_path, errors)
    return unless css
    return if css.match?(LAYERING_RULE)

    errors << "#{relative_path} must keep the Medium Zoom overlay and opened image at z-index: 999"
  end

  def validate_backgrounds(errors)
    %w[assets/js/zoom.js assets/js/theme.js].each do |relative_path|
      javascript = read_artifact(relative_path, errors)
      next unless javascript
      next if javascript.match?(BACKGROUND_RULE)

      errors << "#{relative_path} must pass --global-bg-color directly to Medium Zoom"
    end
  end

  def read_artifact(relative_path, errors)
    File.read(File.join(@site_dir, relative_path))
  rescue Errno::ENOENT
    errors << "Missing production artifact: #{relative_path}"
    nil
  end
end

if $PROGRAM_NAME == __FILE__
  errors = MediumZoomValidator.new(site_dir: ARGV.fetch(0, "_site")).validate

  if errors.empty?
    puts "Medium Zoom production artifacts are valid."
  else
    warn errors.join("\n")
    exit 1
  end
end
