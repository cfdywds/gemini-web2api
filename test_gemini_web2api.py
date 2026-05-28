import unittest

from gemini_web2api import GeminiHandler


class GoogleModelPathParsingTests(unittest.TestCase):
    def parse_model(self, path):
        handler = object.__new__(GeminiHandler)
        handler.path = path
        return handler._parse_google_model_from_path()

    def test_parse_google_v1_generate_content_model_path(self):
        self.assertEqual(
            self.parse_model("/v1/models/gemini-3.5-flash:generateContent"),
            "gemini-3.5-flash",
        )

    def test_parse_google_v1beta_stream_model_path_with_query(self):
        self.assertEqual(
            self.parse_model(
                "/v1beta/models/gemini-3.5-flash-thinking:streamGenerateContent?alt=sse"
            ),
            "gemini-3.5-flash-thinking",
        )

    def test_parse_google_model_path_requires_model_segment(self):
        self.assertIsNone(
            self.parse_model("/v1beta/models:generateContent")
        )


if __name__ == "__main__":
    unittest.main()
