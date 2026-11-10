"""Bounded GitHub code-search evidence, using only the Python standard library.

``GitHubResearch`` implements ``ResearchPort``. It calls only the GitHub code
-search GET endpoint, returns a small citable evidence record, and enforces
tight limits so it can never become a general web browser:

* the ``GITHUB_TOKEN`` is read from the environment at composition time, never
  from source, and is never returned to the caller;
* queries must be a single non-empty line of at most 180 characters;
* at most three requests and three results are allowed per repair;
* each result keeps its provenance (repository, path, GitHub URL) and a snippet
  capped at 500 characters; private results are discarded even if a
  misconfigured token could see them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

API_URL = "https://api.github.com/search/code"
MAX_QUERY_LENGTH = 180
MAX_REQUESTS = 3
MAX_RESULTS = 3
MAX_SNIPPET = 500
USER_AGENT = "coding-agent-lab-research"

# An opener matches ``urllib.request.urlopen(request, timeout=...)``: it returns
# a context manager whose ``read()`` yields the response bytes. Injectable so
# tests can supply a mocked HTTP response without contacting GitHub.
Opener = Callable[..., Any]


class GitHubResearch:
    """Read-only, budget-limited GitHub code search over the standard library."""

    def __init__(
        self,
        token: str | None = None,
        opener: Opener | None = None,
        timeout: float = 15.0,
    ) -> None:
        # Token is captured here (composition time), never read inside source.
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._opener = opener
        self._timeout = timeout
        self._requests_made = 0
        self._results_returned = 0

    def search_github(self, query: str) -> dict[str, Any]:
        """Return one bounded evidence record, or raise if a limit is reached."""

        clean = self._validate_query(query)
        if self._requests_made >= MAX_REQUESTS:
            raise ValueError("research request budget exhausted (max 3 per repair)")
        remaining = MAX_RESULTS - self._results_returned
        if remaining <= 0:
            raise ValueError("research result budget exhausted (max 3 per repair)")

        self._requests_made += 1
        payload = self._fetch(clean)
        sources = self._extract_sources(payload, remaining)
        self._results_returned += len(sources)
        return {
            "query": clean,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        }

    @staticmethod
    def _validate_query(query: str) -> str:
        """Accept only a single, non-empty, bounded-length query string."""

        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if "\n" in query or "\r" in query:
            raise ValueError("query must be a single line")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"query must be at most {MAX_QUERY_LENGTH} characters"
            )
        clean = query.strip()
        if not clean:
            raise ValueError("query must not be empty")
        return clean

    def _fetch(self, query: str) -> dict[str, Any]:
        """GET the code-search endpoint and parse its JSON envelope."""

        params = urllib.parse.urlencode({"q": query, "per_page": MAX_RESULTS})
        request = urllib.request.Request(
            f"{API_URL}?{params}",
            method="GET",
            headers=self._headers(),
        )
        opener = self._opener if self._opener is not None else urllib.request.urlopen
        try:
            with opener(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"GitHub search failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GitHub returned a non-JSON body: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        """Build request headers; the token stays here and is never returned."""

        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _extract_sources(
        self, payload: dict[str, Any], remaining: int
    ) -> list[dict[str, str]]:
        """Keep only public results and only the citable, bounded fields."""

        items = payload.get("items", []) if isinstance(payload, dict) else []
        sources: list[dict[str, str]] = []
        for item in items:
            if len(sources) >= remaining:
                break
            if not isinstance(item, dict):
                continue
            repository = item.get("repository") or {}
            if repository.get("private"):
                continue  # Discard anything GitHub marks private.
            sources.append(
                {
                    "repository": str(repository.get("full_name", "")),
                    "path": str(item.get("path", "")),
                    "url": str(item.get("html_url", "")),
                    "snippet": self._snippet(item),
                }
            )
        return sources

    @staticmethod
    def _snippet(item: dict[str, Any]) -> str:
        """Return a short text-match fragment, never a full remote file."""

        matches = item.get("text_matches") or []
        fragment = ""
        if isinstance(matches, list) and matches and isinstance(matches[0], dict):
            fragment = str(matches[0].get("fragment", "") or "")
        return fragment[:MAX_SNIPPET]
