"""Reading web pages politely. Uses a fake HTTP client, so no network is needed."""

from adaptive_tutor.research.content_extraction import PageFetcher, html_to_text

PAGE_HTML = """
<html><head><title>T</title><style>body {color: red}</style><script>var x = 1;</script></head>
<body>
  <nav>Home | About | Login</nav>
  <article><h1>Mitochondria</h1><p>Mitochondria make energy &amp; heat.</p><p>They have DNA.</p></article>
  <footer>Copyright</footer>
</body></html>
"""


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "", content_type: str = "text/html; charset=utf-8") -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}
        self.charset_encoding = "utf-8"

    def iter_bytes(self):
        data = self.text.encode("utf-8")
        for start in range(0, len(data), 100):
            yield data[start : start + 100]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeClient:
    """Answers from a dict of url -> FakeResponse and remembers every URL requested."""

    def __init__(self, pages: dict[str, FakeResponse]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def _answer(self, url: str) -> FakeResponse:
        self.requested.append(url)
        return self.pages.get(url, FakeResponse(404, "not found", "text/plain"))

    def get(self, url: str) -> FakeResponse:
        return self._answer(url)

    def stream(self, method: str, url: str) -> FakeResponse:
        return self._answer(url)


def test_html_to_text_keeps_content_and_drops_menus_and_scripts():
    text = html_to_text(PAGE_HTML)
    assert text == "T\nMitochondria\nMitochondria make energy & heat.\nThey have DNA."


def test_page_is_read_when_there_is_no_robots_file():
    client = FakeClient({"https://site.example/page": FakeResponse(text=PAGE_HTML)})
    text = PageFetcher(client).fetch_text("https://site.example/page")
    assert "They have DNA." in text
    assert client.requested == ["https://site.example/robots.txt", "https://site.example/page"]


def test_robots_txt_disallow_is_respected_and_page_is_never_requested():
    client = FakeClient({
        "https://site.example/robots.txt": FakeResponse(text="User-agent: *\nDisallow: /private/", content_type="text/plain"),
        "https://site.example/private/page": FakeResponse(text=PAGE_HTML),
    })
    fetcher = PageFetcher(client)
    assert fetcher.fetch_text("https://site.example/private/page") is None
    assert "https://site.example/private/page" not in client.requested
    assert fetcher.is_allowed("https://site.example/public/page") is True


def test_robots_txt_that_names_our_app_is_respected():
    client = FakeClient({
        "https://site.example/robots.txt": FakeResponse(text="User-agent: AdaptiveAITutor\nDisallow: /", content_type="text/plain"),
    })
    assert PageFetcher(client).is_allowed("https://site.example/anything") is False


def test_protected_robots_file_means_do_not_fetch():
    client = FakeClient({"https://site.example/robots.txt": FakeResponse(403, "forbidden", "text/plain")})
    assert PageFetcher(client).is_allowed("https://site.example/page") is False


def test_robots_file_is_read_once_per_site():
    client = FakeClient({
        "https://site.example/a": FakeResponse(text=PAGE_HTML),
        "https://site.example/b": FakeResponse(text=PAGE_HTML),
    })
    fetcher = PageFetcher(client)
    fetcher.fetch_text("https://site.example/a")
    fetcher.fetch_text("https://site.example/b")
    assert client.requested.count("https://site.example/robots.txt") == 1


def test_network_failure_while_reading_robots_means_do_not_fetch():
    class BrokenClient(FakeClient):
        def get(self, url):
            raise ConnectionError("no network")

    assert PageFetcher(BrokenClient({})).fetch_text("https://site.example/page") is None


def test_non_html_pages_are_skipped():
    client = FakeClient({"https://site.example/file.pdf": FakeResponse(text="%PDF-1.4", content_type="application/pdf")})
    assert PageFetcher(client).fetch_text("https://site.example/file.pdf") is None


def test_long_pages_are_trimmed():
    long_page = "<p>" + "energy " * 5000 + "</p>"
    client = FakeClient({"https://site.example/long": FakeResponse(text=long_page)})
    text = PageFetcher(client, max_chars=500).fetch_text("https://site.example/long")
    assert len(text) == 500