"""Tests for OpenAI endpoint detection utilities."""

import pytest

from chunkhound.core.config.openai_utils import (
    is_azure_openai_endpoint,
    is_official_openai_endpoint,
)


class TestIsOfficialOpenaiEndpoint:
    """Tests for is_official_openai_endpoint()."""

    def test_none_returns_true(self):
        assert is_official_openai_endpoint(None) is True

    def test_official_domain(self):
        assert is_official_openai_endpoint("https://api.openai.com") is True

    def test_official_domain_with_v1(self):
        assert is_official_openai_endpoint("https://api.openai.com/v1") is True

    def test_http_not_https(self):
        assert is_official_openai_endpoint("http://api.openai.com/v1") is False

    def test_subdomain_spoofing(self):
        assert is_official_openai_endpoint("https://api.openai.com.evil.com") is False

    def test_localhost(self):
        assert is_official_openai_endpoint("http://localhost:8080/v1") is False

    def test_proxy(self):
        assert is_official_openai_endpoint("https://my-proxy.com/openai") is False

    def test_empty_string(self):
        assert is_official_openai_endpoint("") is True  # falsy -> default endpoint


class TestIsAzureOpenaiEndpoint:
    """Tests for is_azure_openai_endpoint()."""

    def test_none_returns_false(self):
        assert is_azure_openai_endpoint(None) is False

    def test_empty_string_returns_false(self):
        assert is_azure_openai_endpoint("") is False

    def test_valid_azure_endpoint(self):
        assert is_azure_openai_endpoint("https://myresource.openai.azure.com") is True

    def test_valid_azure_with_trailing_slash(self):
        assert is_azure_openai_endpoint("https://myresource.openai.azure.com/") is True

    def test_http_rejected(self):
        assert is_azure_openai_endpoint("http://myresource.openai.azure.com") is False

    def test_subdomain_spoofing(self):
        assert is_azure_openai_endpoint("https://evil.com/openai.azure.com") is False

    def test_suffix_spoofing(self):
        assert is_azure_openai_endpoint("https://openai.azure.com.evil.com") is False

    def test_regular_openai_endpoint(self):
        assert is_azure_openai_endpoint("https://api.openai.com/v1") is False

    def test_case_insensitive(self):
        assert is_azure_openai_endpoint("https://MyResource.OpenAI.Azure.COM") is True
