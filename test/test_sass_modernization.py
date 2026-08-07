#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "assets/css/main.scss"
VENDOR_CSS = ROOT / "assets/css/vendor-icons.css"


class SassModernizationTests(unittest.TestCase):
    def test_first_party_entrypoint_uses_modules(self) -> None:
        source = MAIN.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*@import\b")
        for module in (
            "variables",
            "themes",
            "layout",
            "base",
            "distill",
            "cv",
            "tabs",
            "typograms",
        ):
            self.assertIn(f'@use "{module}"', source)

    def test_first_party_sass_has_no_deprecated_calls(self) -> None:
        offenders: list[str] = []
        patterns = (
            re.compile(r"(?m)^\s*@import\b"),
            re.compile(r"(?<![.\w-])(?:red|green|blue)\("),
        )
        for path in sorted((ROOT / "_sass").glob("*.scss")):
            source = path.read_text(encoding="utf-8")
            if any(pattern.search(source) for pattern in patterns):
                offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_variable_consumers_use_variables_module(self) -> None:
        for relative_path in ("_sass/_base.scss", "_sass/_layout.scss", "_sass/_themes.scss"):
            with self.subTest(path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn('@use "variables" as *;', source)
                result = subprocess.run(
                    [
                        "bundle",
                        "exec",
                        "sass",
                        "--load-path",
                        "_sass",
                        "--no-source-map",
                        relative_path,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_vendor_icons_are_static_and_wired_after_main(self) -> None:
        self.assertTrue(VENDOR_CSS.is_file())
        css = VENDOR_CSS.read_text(encoding="utf-8")
        for marker in (
            "Font Awesome 6 Free",
            "Font Awesome 6 Brands",
            "tabler-icons",
            "tabler-icons-filled",
        ):
            self.assertIn(marker, css)

        head = (ROOT / "_includes/head.liquid").read_text(encoding="utf-8")
        main_link = head.index("/assets/css/main.css")
        vendor_link = head.index("/assets/css/vendor-icons.css")
        self.assertGreater(vendor_link, main_link)
        self.assertIn(
            "'/assets/css/vendor-icons.css' | relative_url | bust_file_cache",
            head,
        )
        self.assertFalse((ROOT / "_sass/font-awesome").exists())
        self.assertFalse((ROOT / "_sass/tabler-icons").exists())

    def test_font_assets_exist(self) -> None:
        assets = (
            "assets/webfonts/fa-brands-400.woff2",
            "assets/webfonts/fa-regular-400.woff2",
            "assets/webfonts/fa-solid-900.woff2",
            "assets/fonts/tabler-icons.woff2",
            "assets/fonts/tabler-icons-filled.woff2",
            "assets/fonts/tabler-icons-outline.woff2",
        )
        for asset in assets:
            with self.subTest(asset=asset):
                self.assertTrue((ROOT / asset).is_file())

    def test_lockfile_is_tracked_by_policy(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertNotIn("Gemfile.lock", ignored)
        self.assertTrue((ROOT / "Gemfile.lock").is_file())
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "Gemfile.lock"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
