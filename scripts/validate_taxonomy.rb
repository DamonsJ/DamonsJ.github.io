#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "pathname"
require "yaml"

begin
  require "jekyll/utils"
rescue LoadError
  warn "Jekyll is required. Run this command with `bundle exec ruby`."
  raise
end

class TaxonomyValidator
  Entry = Struct.new(:collection, :path, :metadata, keyword_init: true)

  attr_reader :entry_count

  def initialize(root: Pathname.pwd)
    @root = Pathname(root).expand_path
    @entry_count = 0
    @entries = []
    @taxonomy = nil
  end

  def validate_source
    @entries = []
    @entry_count = 0
    errors = []
    taxonomy_path = @root / "_data/taxonomy.yml"
    unless taxonomy_path.file?
      return ["#{relative(taxonomy_path)}: taxonomy configuration is missing"]
    end

    @taxonomy = load_yaml(taxonomy_path)
    collections = @taxonomy.fetch("collections", {})
    allowed_source_types = Array(@taxonomy["source_types"])
    used_categories = Hash.new { |hash, key| hash[key] = [] }
    used_tags = Hash.new { |hash, key| hash[key] = [] }

    collections.each do |collection, config|
      directory = @root / config.fetch("directory")
      allowed_categories = Array(config["categories"])
      allowed_tags = Array(config["tags"])

      directory.glob("*.md").sort.each do |path|
        metadata = read_front_matter(path, errors)
        next unless metadata

        @entries << Entry.new(collection: collection, path: path, metadata: metadata)
        validate_entry(
          path: path,
          metadata: metadata,
          allowed_categories: allowed_categories,
          allowed_tags: allowed_tags,
          allowed_source_types: allowed_source_types,
          errors: errors
        )

        Array(metadata["categories"]).each do |category|
          used_categories[collection] << category if allowed_categories.include?(category)
        end
        Array(metadata["tags"]).each do |tag|
          used_tags[collection] << tag if allowed_tags.include?(tag)
        end
      end

      allowed_categories.each do |category|
        errors << "_data/taxonomy.yml: #{collection} has unused category '#{category}'" unless used_categories[collection].include?(category)
      end
      allowed_tags.each do |tag|
        errors << "_data/taxonomy.yml: #{collection} has unused tag '#{tag}'" unless used_tags[collection].include?(tag)
      end
    end

    @entry_count = @entries.length
    errors
  rescue KeyError => error
    ["_data/taxonomy.yml: missing required key #{error.key.inspect}"]
  rescue Psych::Exception => error
    ["_data/taxonomy.yml: invalid YAML: #{error.message}"]
  end

  def validate_generated(site_dir: @root / "_site")
    validate_source if @taxonomy.nil? || @entries.empty?
    site_dir = Pathname(site_dir)
    errors = []

    @entries.each do |entry|
      config = @taxonomy.fetch("collections").fetch(entry.collection)
      prefix = config.fetch("archive_prefix").sub(%r{\A/}, "").sub(%r{/\z}, "")
      {
        "category" => Array(entry.metadata["categories"]),
        "tag" => Array(entry.metadata["tags"])
      }.each do |type, terms|
        terms.each do |term|
          slug = Jekyll::Utils.slugify(term.to_s)
          archive = site_dir / prefix / type / slug / "index.html"
          next if archive.file?

          errors << "#{relative(entry.path)}: missing #{type} archive #{archive.relative_path_from(site_dir)}"
        end
      end
    end

    @taxonomy.fetch("collections").each do |collection, config|
      landing_path = config.fetch("landing_path").sub(%r{\A/}, "").sub(%r{/\z}, "")
      landing_page = site_dir / landing_path / "index.html"
      unless landing_page.file?
        errors << "#{collection}: missing collection landing page #{landing_page.relative_path_from(site_dir)}"
        next
      end

      html = landing_page.read(encoding: "UTF-8")
      taxonomy_block = html[/<div[^>]*class=["'][^"']*tag-category-list[^"']*["'][^>]*>.*?<\/div>/m]
      unless taxonomy_block
        errors << "#{collection}: landing page is missing the taxonomy category list"
        next
      end

      errors << "#{collection}: landing taxonomy must display category links" unless taxonomy_block.include?("/category/")
      if taxonomy_block.include?("/tag/") || taxonomy_block.include?("fa-hashtag")
        errors << "#{collection}: landing taxonomy must not display tag links"
      end
    end

    search_path = site_dir / "assets/js/search-data.js"
    unless search_path.file?
      errors << "#{relative(search_path)}: missing generated search data"
      return errors
    end

    search_data = search_path.read(encoding: "UTF-8")
    @entries.each do |entry|
      source = relative(entry.path)
      block = search_action_block(search_data, source)
      unless block
        errors << "#{source}: missing generated search action"
        next
      end

      keywords = block[/keywords:\s*["']([^"']*)["']/, 1].to_s
      description = block[/description:\s*["']([^"']*)["']/, 1].to_s
      expected_terms = Array(entry.metadata["categories"]) + Array(entry.metadata["tags"])
      expected_terms << "转载" if entry.metadata["source_type"] == "repost"
      expected_terms.each do |term|
        errors << "#{source}: generated search keywords omit '#{term}'" unless keywords.include?(term.to_s)
        errors << "#{source}: generated search description omits '#{term}'" unless description.include?(term.to_s)
      end
    end

    errors
  end

  private

  def validate_entry(path:, metadata:, allowed_categories:, allowed_tags:, allowed_source_types:, errors:)
    categories = metadata["categories"]
    tags = metadata["tags"]

    unless categories.is_a?(Array)
      errors << "#{relative(path)}: categories must be a YAML array"
      categories = []
    end
    unless tags.is_a?(Array)
      errors << "#{relative(path)}: tags must be a YAML array"
      tags = []
    end

    errors << "#{relative(path)}: expected exactly one category, found #{categories.length}" unless categories.length == 1
    errors << "#{relative(path)}: expected between 2 and 4 tags, found #{tags.length}" unless (2..4).cover?(tags.length)

    categories.each do |category|
      errors << "#{relative(path)}: category '#{category}' is not in the controlled vocabulary" unless allowed_categories.include?(category)
    end
    tags.each do |tag|
      errors << "#{relative(path)}: tag '#{tag}' is not in the controlled vocabulary" unless allowed_tags.include?(tag)
    end
    tags.tally.each do |tag, count|
      errors << "#{relative(path)}: duplicate tag '#{tag}'" if count > 1
    end
    errors << "#{relative(path)}: '转载' must use source_type: repost instead of a tag" if tags.include?("转载")

    source_type = metadata["source_type"]
    return if source_type.nil? || allowed_source_types.include?(source_type)

    errors << "#{relative(path)}: source_type '#{source_type}' is invalid"
  end

  def read_front_matter(path, errors)
    content = path.read(encoding: "UTF-8")
    match = content.match(/\A---\s*\n(.*?)\n---\s*(?:\n|\z)/m)
    unless match
      errors << "#{relative(path)}: missing YAML front matter"
      return nil
    end

    YAML.safe_load(match[1], permitted_classes: [Date, Time], aliases: true) || {}
  rescue Psych::Exception => error
    errors << "#{relative(path)}: invalid YAML front matter: #{error.message}"
    nil
  end

  def load_yaml(path)
    YAML.safe_load(path.read(encoding: "UTF-8"), permitted_classes: [Date, Time], aliases: true) || {}
  end

  def search_action_block(search_data, source)
    source_pattern = Regexp.escape(source)
    marker = /taxonomySource:\s*["']#{source_pattern}["']/
    start = search_data.index(marker)
    return nil unless start

    tail = search_data[start..]
    next_action = tail.index(/taxonomySource:/, marker.match(tail).end(0))
    next_action ? tail[0...next_action] : tail
  end

  def relative(path)
    Pathname(path).expand_path.relative_path_from(@root).to_s
  rescue ArgumentError
    path.to_s
  end
end

if $PROGRAM_NAME == __FILE__
  site_argument = ARGV.index("--site")
  site_dir = site_argument ? ARGV.fetch(site_argument + 1) : nil
  validator = TaxonomyValidator.new
  errors = validator.validate_source
  errors.concat(validator.validate_generated(site_dir: Pathname(site_dir))) if site_dir && errors.empty?

  if errors.empty?
    puts "Taxonomy source validation passed (#{validator.entry_count} entries)."
    puts "Generated archives and search data passed." if site_dir
  else
    warn errors.join("\n")
    exit 1
  end
end
