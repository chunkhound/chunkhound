"""The shipped homepage paints only the icon for the selected theme mode."""

from tests.site.browser_helpers import homepage_probe


def test_theme_toggle_shows_one_icon_per_setting(built_site) -> None:
    result = homepage_probe("""
const result = await homepageProbe(['light', 'dark', 'system'], themeSnapshot);
console.log(JSON.stringify(result));
""")
    expected = {"light": "ph-sun", "dark": "ph-moon", "system": "ph-monitor"}
    for setting, icon in expected.items():
        assert result[setting] == [icon], (
            f"{setting}: expected only {icon} visible, got {result[setting]}"
        )
