"""Tests for VoyageAI endpoint detection utilities."""

from chunkhound.core.utils.voyageai_utils import is_official_voyageai_endpoint


class TestIsOfficialVoyageaiEndpoint:
    """Tests for is_official_voyageai_endpoint()."""

    def test_none_returns_true(self):
        assert is_official_voyageai_endpoint(None) is True

    def test_empty_string_returns_true(self):
        # falsy -> default endpoint
        assert is_official_voyageai_endpoint("") is True

    def test_official_domain(self):
        assert is_official_voyageai_endpoint("https://api.voyageai.com") is True

    def test_official_domain_with_v1(self):
        assert is_official_voyageai_endpoint("https://api.voyageai.com/v1") is True

    def test_official_domain_trailing_slash(self):
        assert is_official_voyageai_endpoint("https://api.voyageai.com/") is True

    def test_subdomain_spoofing(self):
        # A host that merely starts with the official base is NOT official.
        assert (
            is_official_voyageai_endpoint("https://api.voyageai.com.evil.com") is False
        )

    def test_suffix_hyphen_spoofing(self):
        assert (
            is_official_voyageai_endpoint("https://api.voyageai.com-evil.com") is False
        )

    def test_custom_proxy(self):
        assert (
            is_official_voyageai_endpoint("https://my-proxy.internal/voyage") is False
        )
