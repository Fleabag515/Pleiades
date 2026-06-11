"""Tests for the HTML->text extractor that feeds http_fetch / deep_research.

Pure and offline — no network, no SearXNG, no browser. Guards the signal-per-token
behaviour: chrome (script/style/nav/header/footer) is dropped, entities decoded,
and paragraph structure preserved.
"""

from pleiades.harness.builtins.web import _strip_html


SAMPLE = """
<html><head>
  <title>Demo</title>
  <style>.x{color:red}</style>
  <script>var tracker = 1; alert('no');</script>
</head><body>
  <nav><a href="/a">Home</a><a href="/b">About</a></nav>
  <header>SiteName menu junk</header>
  <article>
    <h1>Real Heading</h1>
    <p>First paragraph with an entity: don&#39;t panic &amp; carry on.</p>
    <p>Second paragraph.</p>
  </article>
  <footer>copyright boilerplate 2026</footer>
</body></html>
"""


def test_drops_chrome_blocks():
    text = _strip_html(SAMPLE)
    # scripts/styles never leak
    assert "tracker" not in text
    assert "color:red" not in text
    # nav / header / footer chrome dropped
    assert "About" not in text
    assert "menu junk" not in text
    assert "boilerplate" not in text


def test_keeps_content_and_decodes_entities():
    text = _strip_html(SAMPLE)
    assert "Real Heading" in text
    assert "Second paragraph." in text
    # &#39; -> ' and &amp; -> &
    assert "don't panic & carry on." in text


def test_preserves_paragraph_structure():
    text = _strip_html(SAMPLE)
    # the two paragraphs should not be glued into one line
    assert "\n" in text
    assert "carry on.Second" not in text


def test_blank_input_is_safe():
    assert _strip_html("") == ""
    assert _strip_html("<p></p>") == ""
