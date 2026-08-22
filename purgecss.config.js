module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  safelist: ["medium-zoom-overlay", "medium-zoom-image--opened"],
  skippedContentGlobs: ["_site/assets/**/*.html"],
};
