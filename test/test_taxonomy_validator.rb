# frozen_string_literal: true

require "fileutils"
require "minitest/autorun"
require "tmpdir"
require "yaml"

require_relative "../scripts/validate_taxonomy"

class TaxonomyValidatorTest < Minitest::Test
  def setup
    @tmpdir = Dir.mktmpdir("taxonomy-validator")
    @root = Pathname(@tmpdir)
    write_taxonomy(
      "categories" => ["Topic"],
      "tags" => ["One", "Two"]
    )
  end

  def teardown
    FileUtils.remove_entry(@tmpdir)
  end

  def test_accepts_valid_source_metadata
    write_entry(
      "categories" => ["Topic"],
      "tags" => ["One", "Two"],
      "source_type" => "repost"
    )

    validator = TaxonomyValidator.new(root: @root)

    assert_empty validator.validate_source
    assert_equal 1, validator.entry_count
  end

  def test_reports_cardinality_vocabulary_provenance_and_unused_terms
    write_taxonomy(
      "categories" => ["Topic", "Unused category"],
      "tags" => ["One", "Two", "Unused tag"]
    )
    write_entry(
      "categories" => ["Wrong", "Another"],
      "tags" => ["转载", "One", "One", "Extra", "Fifth"],
      "source_type" => "copied"
    )

    errors = TaxonomyValidator.new(root: @root).validate_source.join("\n")

    assert_includes errors, "exactly one category"
    assert_includes errors, "between 2 and 4 tags"
    assert_includes errors, "category 'Wrong' is not in the controlled vocabulary"
    assert_includes errors, "tag 'Extra' is not in the controlled vocabulary"
    assert_includes errors, "duplicate tag 'One'"
    assert_includes errors, "'转载' must use source_type: repost instead of a tag"
    assert_includes errors, "source_type 'copied' is invalid"
    assert_includes errors, "unused category 'Topic'"
    assert_includes errors, "unused tag 'Two'"
  end

  def test_requires_array_metadata
    write_entry("categories" => "Topic", "tags" => "One Two")

    errors = TaxonomyValidator.new(root: @root).validate_source.join("\n")

    assert_includes errors, "categories must be a YAML array"
    assert_includes errors, "tags must be a YAML array"
  end

  def test_validates_generated_archives_and_search_keywords
    write_entry(
      "categories" => ["Topic"],
      "tags" => ["One", "Two"],
      "source_type" => "repost"
    )
    site_dir = @root / "_site"
    validator = TaxonomyValidator.new(root: @root)
    validator.validate_source

    missing_errors = validator.validate_generated(site_dir: site_dir).join("\n")
    assert_includes missing_errors, "missing category archive"
    assert_includes missing_errors, "missing tag archive"
    assert_includes missing_errors, "missing generated search data"
    assert_includes missing_errors, "missing collection landing page"

    %w[category/topic tag/one tag/two].each do |relative_path|
      path = site_dir / "blog" / relative_path / "index.html"
      path.dirname.mkpath
      path.write("archive", mode: "w", encoding: "UTF-8")
    end
    search_data = site_dir / "assets/js/search-data.js"
    search_data.dirname.mkpath
    search_data.write(
      <<~JAVASCRIPT,
        ninja.data = [{
          taxonomySource: "_posts/example.md",
          keywords: "Topic One Two 转载",
        }];
      JAVASCRIPT
      mode: "w",
      encoding: "UTF-8"
    )

    unsearchable_errors = validator.validate_generated(site_dir: site_dir).join("\n")
    assert_includes unsearchable_errors, "generated search description omits 'Topic'"

    search_data.write(
      <<~JAVASCRIPT,
        ninja.data = [{
          taxonomySource: "_posts/example.md",
          keywords: "Topic One Two 转载",
          description: "Example description Topic One Two 转载",
        }];
      JAVASCRIPT
      mode: "w",
      encoding: "UTF-8"
    )

    landing_page = site_dir / "blog/index.html"
    landing_page.dirname.mkpath
    landing_page.write(
      '<div class="tag-category-list"><a href="/blog/tag/one/">One</a></div>',
      mode: "w",
      encoding: "UTF-8"
    )
    landing_errors = validator.validate_generated(site_dir: site_dir).join("\n")
    assert_includes landing_errors, "landing taxonomy must not display tag links"

    landing_page.write(
      '<div class="tag-category-list"><a href="/blog/category/topic/">Topic</a></div>',
      mode: "w",
      encoding: "UTF-8"
    )

    assert_empty validator.validate_generated(site_dir: site_dir)
  end

  private

  def write_taxonomy(collection_values)
    taxonomy = {
      "collections" => {
        "posts" => {
          "directory" => "_posts",
          "archive_prefix" => "/blog/",
          "landing_path" => "/blog/"
        }.merge(collection_values)
      },
      "source_types" => ["repost"]
    }
    path = @root / "_data/taxonomy.yml"
    path.dirname.mkpath
    path.write(YAML.dump(taxonomy), mode: "w", encoding: "UTF-8")
  end

  def write_entry(front_matter)
    path = @root / "_posts/example.md"
    path.dirname.mkpath
    metadata = { "title" => "Example" }.merge(front_matter)
    path.write("---\n#{YAML.dump(metadata).sub(/\A---\s*\n/, "")}---\nBody\n", mode: "w", encoding: "UTF-8")
  end
end
