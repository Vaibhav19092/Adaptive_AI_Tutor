"""Optional: read the text of a web page, but only where the site's robots.txt allows it.

Uses only Python's standard library (html.parser and urllib.robotparser),
so no new packages are needed. Switched on with RESEARCH_FETCH_PAGES=true.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

if TYPE_CHECKING:  # only for type hints; keeps this module importable without httpx
    import httpx

logger = logging.getLogger(__name__)

ROBOTS_AGENT = "AdaptiveAITutor"  # the name we look for in robots.txt rules
MAX_PAGE_BYTES = 2_000_000  # stop downloading after about 2 MB
MAX_PAGE_TEXT_CHARS = 5_000  # keep at most this much text per page

# Parts of a page that are menus, code, or decoration rather than content.
SKIPPED_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe",
    "nav", "header", "footer", "aside", "form", "button",
}
# Tags that start a new line of text.
BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "br", "li", "ul", "ol", "tr", "table",
    "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
}


class _TextCollector(HTMLParser):
    """Walks through the HTML and keeps only readable text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # turns &amp; into &, etc.
        self._skip_depth = 0  # > 0 while we are inside a skipped tag
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Turn an HTML page into plain text lines, without menus and scripts."""
    collector = _TextCollector()
    collector.feed(html)
    collector.close()
    lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(collector.parts).split("\n"))
    return "\n".join(line for line in lines if line)


class PageFetcher:
    """Downloads result pages politely: robots.txt first, size limit, HTML only."""

    def __init__(
        self,
        client: httpx.Client,
        max_bytes: int = MAX_PAGE_BYTES,
        max_chars: int = MAX_PAGE_TEXT_CHARS,
    ) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._robots: dict[str, RobotFileParser | None] = {}  # one robots.txt per website

    def is_allowed(self, url: str) -> bool:
        """True only if the site's robots.txt lets us read this page."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        rules = self._robots[origin]
        return rules is not None and rules.can_fetch(ROBOTS_AGENT, url)

    def _load_robots(self, origin: str) -> RobotFileParser | None:
        """Read robots.txt. When in doubt, the answer is 'don't fetch'."""
        rules = RobotFileParser()
        try:
            response = self._client.get(f"{origin}/robots.txt")
        except Exception:  # noqa: BLE001 - any network failure means "don't fetch"
            logger.info("Could not read robots.txt for %s; skipping its pages", urlsplit(origin).hostname)
            return None

        status = response.status_code
        if status in (401, 403) or status >= 500:
            rules.disallow_all = True  # the site is protected or broken: stay away
        elif status >= 400:
            rules.allow_all = True  # no robots.txt (e.g. 404) means no restrictions
        elif status >= 300:
            return None  # redirected robots.txt: we don't follow redirects, so skip
        else:
            rules.parse(response.text.splitlines())
        return rules

    def fetch_text(self, url: str) -> str | None:
        """Return the page's readable text, or None if not allowed or not possible."""
        host = urlsplit(url).hostname
        if not self.is_allowed(url):
            logger.info("Not reading a page on %s: robots.txt does not allow it", host)
            return None

        try:
            with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    return None
                if "text/html" not in response.headers.get("content-type", "").lower():
                    return None  # PDFs, images, etc. are skipped
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= self._max_bytes:
                        break
                encoding = response.charset_encoding or "utf-8"
        except Exception:  # noqa: BLE001 - a page we can't read just keeps its snippet
            logger.info("Could not read a page on %s; keeping the search snippet", host)
            return None

        raw = b"".join(chunks)
        try:
            html = raw.decode(encoding, errors="replace")
        except LookupError:  # the page named an encoding Python doesn't know
            html = raw.decode("utf-8", errors="replace")

        try:
            text = html_to_text(html)
        except Exception:  # noqa: BLE001 - badly broken HTML
            return None
        return text[: self._max_chars] or None