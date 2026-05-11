"""
modules/rag/databanks.py

DataBank abstraction for the RAG module.

A DataBank is a named, indexable source of text content. Two implementations
are provided here:

  FilesDataBank     — a folder of text files (*.md, *.txt, *.rst, etc.)
  LoreBookDataBank  — a SillyTavern lorebook/worldinfo JSON file

Public API
----------
  DataBank (Protocol)        — duck-typed interface
  FilesDataBank(name, root, extensions)
  LoreBookDataBank(name, path)
  discover_databanks(rag_dir, extensions) -> dict[str, DataBank]
      Scan workspace/rag/ and return all valid databanks by name.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DataBank(Protocol):
    """
    A named, indexable source of text content.

    name         — identifier used in tool calls (e.g. "lore", "characters")
    kind         — "files" | "lorebook" | etc.
    iter_files() — yields (path_str, content) pairs for all indexable items
    """

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def iter_files(self) -> Iterator[tuple[str, str]]:
        """
        Yield (path_str, text_content) for each indexable item.
        path_str is a stable, unique string key used by the store (e.g. absolute path).
        text_content is the full text to chunk and index.
        """
        ...


# ---------------------------------------------------------------------------
# FilesDataBank — folder of text files
# ---------------------------------------------------------------------------

class FilesDataBank:
    """
    A databank backed by a folder of text files.

    Recursively walks `root` and yields any file whose suffix is in `extensions`.
    Files that cannot be decoded as UTF-8 are skipped with a warning.

    Args:
        name:       Databank identifier (typically the folder name).
        root:       Absolute path to the databank folder.
        extensions: Set of lowercase file extensions to include (e.g. {".md", ".txt"}).
    """

    def __init__(self, name: str, root: Path, extensions: set[str]) -> None:
        self._name       = name
        self._root       = root
        self._extensions = extensions

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return "files"

    def iter_files(self) -> Iterator[tuple[str, str]]:
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._extensions:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("[rag/databanks] skipping %s: %s", path, exc)
                continue
            yield str(path.resolve()), content

    def __repr__(self) -> str:
        return f"FilesDataBank({self._name!r}, root={self._root})"


# ---------------------------------------------------------------------------
# LoreBookDataBank — SillyTavern lorebook/worldinfo JSON
# ---------------------------------------------------------------------------

# selectiveLogic values from world-info.js:33-38
_AND_ANY = 0  # primary hit OR secondary hit (either alone fires)
_NOT_ALL = 1  # primary hit AND NOT all secondary keys present
_NOT_ANY = 2  # primary hit AND none of secondary keys present
_AND_ALL = 3  # primary hit AND all secondary keys present


class LoreBookDataBank:
    """
    A databank backed by a SillyTavern lorebook JSON file.

    Two access modes:

    iter_files() — for rag_search (embedding / BM25)
        Yields one document per active entry. The text is:
            "{comment}\\n{keys}\\n{content}"
        so the retriever can match on title, keywords, and body alike.
        path_str key is "{json_path}::{uid}".

    keyword_match(text) — for set_auto_rag_databanks injection
        Replicates SillyTavern keyword matching logic:
        - Skips disabled entries.
        - constant=True entries always fire.
        - Non-selective entries: fire if any primary key hits.
        - Selective entries: apply selectiveLogic with secondary keys:
            AND_ANY (0): primary hit OR secondary hit
            NOT_ALL (1): primary hit AND NOT all secondary keys present
            NOT_ANY (2): primary hit AND none of secondary keys present
            AND_ALL (3): primary hit AND all secondary keys present
        - caseSensitive and matchWholeWords are respected when set.
        Returns list of content strings for matched entries.
    """

    def __init__(self, name: str, path: Path) -> None:
        self._name = name
        self._path = path
        self._entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[rag/databanks] failed to load lorebook %s: %s", self._path, exc)
            return

        # Support both the flat {"entries": {"1": {...}}} shape and
        # the originalData {"entries": [...]} shape. Prefer top-level entries.
        entries_raw = raw.get("entries", {})
        if isinstance(entries_raw, dict):
            self._entries = list(entries_raw.values())
        elif isinstance(entries_raw, list):
            self._entries = entries_raw
        else:
            logger.warning("[rag/databanks] unexpected entries format in %s", self._path)

        logger.debug(
            "[rag/databanks] loaded LoreBookDataBank %r: %d entries",
            self._name, len(self._entries),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return "lorebook"

    # -- rag_search path ------------------------------------------------------

    def iter_files(self) -> Iterator[tuple[str, str]]:
        """Yield (key, text) for each active entry, for embedding/BM25 indexing."""
        for entry in self._entries:
            if entry.get("disable", False):
                continue
            uid = entry.get("uid") or entry.get("id", "?")
            keys = entry.get("key") or entry.get("keys", [])
            comment = entry.get("comment", "")
            content = entry.get("content", "")

            # Combine comment + keys + content so all are searchable
            text = "\n".join(filter(None, [
                comment,
                ", ".join(keys),
                content,
            ]))
            path_key = f"{self._path}::{uid}"
            yield path_key, text

    # -- set_auto_rag_databanks path ------------------------------------------

    def keyword_match(self, text: str) -> list[str]:
        """
        Return content strings for all entries whose keywords match `text`.
        Follows SillyTavern selectiveLogic from world-info.js:33-38.
        """
        results = []
        for entry in self._entries:
            if entry.get("disable", False):
                continue
            if entry.get("constant", False):
                results.append(entry.get("content", ""))
                continue

            primary_keys   = entry.get("key") or entry.get("keys", [])
            secondary_keys = entry.get("keysecondary") or entry.get("secondary_keys", [])
            selective      = entry.get("selective", False)
            logic          = entry.get("selectiveLogic", _AND_ANY)
            case_sensitive = entry.get("caseSensitive")   # None -> False
            whole_words    = entry.get("matchWholeWords")  # None -> False

            primary_hit = self._any_key_matches(primary_keys, text, case_sensitive, whole_words)

            if not selective:
                fired = primary_hit
            elif logic == _AND_ANY:
                # Either primary or secondary hit suffices
                secondary_hit = self._any_key_matches(secondary_keys, text, case_sensitive, whole_words)
                fired = primary_hit or secondary_hit
            elif logic == _NOT_ALL:
                # Primary must hit AND not all secondary keys present
                all_secondary = self._all_keys_match(secondary_keys, text, case_sensitive, whole_words)
                fired = primary_hit and not all_secondary
            elif logic == _NOT_ANY:
                # Primary must hit AND none of secondary keys present
                secondary_hit = self._any_key_matches(secondary_keys, text, case_sensitive, whole_words)
                fired = primary_hit and not secondary_hit
            elif logic == _AND_ALL:
                # Primary must hit AND all secondary keys present
                all_secondary = self._all_keys_match(secondary_keys, text, case_sensitive, whole_words)
                fired = primary_hit and all_secondary
            else:
                fired = primary_hit

            if fired:
                results.append(entry.get("content", ""))

        return results

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _any_key_matches(
        keys: list[str],
        text: str,
        case_sensitive: bool | None,
        whole_words: bool | None,
    ) -> bool:
        if not keys:
            return False
        cs = bool(case_sensitive)
        ww = bool(whole_words)
        haystack = text if cs else text.lower()
        for key in keys:
            needle = key if cs else key.lower()
            if ww:
                flags = 0 if cs else re.IGNORECASE
                if re.search(r"\b" + re.escape(needle) + r"\b", haystack if cs else text, flags):
                    return True
            else:
                if needle in haystack:
                    return True
        return False

    @staticmethod
    def _all_keys_match(
        keys: list[str],
        text: str,
        case_sensitive: bool | None,
        whole_words: bool | None,
    ) -> bool:
        """Return True only if every key in `keys` matches `text`."""
        if not keys:
            return False
        cs = bool(case_sensitive)
        ww = bool(whole_words)
        haystack = text if cs else text.lower()
        for key in keys:
            needle = key if cs else key.lower()
            if ww:
                flags = 0 if cs else re.IGNORECASE
                if not re.search(r"\b" + re.escape(needle) + r"\b", haystack if cs else text, flags):
                    return False
            else:
                if needle not in haystack:
                    return False
        return True

    def __repr__(self) -> str:
        return f"LoreBookDataBank({self._name!r}, path={self._path})"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_databanks(rag_dir: Path, extensions: set[str]) -> dict[str, "DataBank"]:
    """
    Scan `rag_dir` and return a dict of {name: DataBank} for all valid sources.

    Rules:
      - A subdirectory of rag_dir   -> FilesDataBank named after the folder.
      - A *.json file in rag_dir    -> LoreBookDataBank named after the stem.
      - Other files at root level   -> ignored.
      - rag_dir doesn't exist yet   -> returns empty dict (no error).

    The .cache directory is always excluded.
    """
    if not rag_dir.exists():
        return {}

    result: dict[str, DataBank] = {}

    for entry in sorted(rag_dir.iterdir()):
        if entry.name.startswith(".") or entry.name == ".cache":
            continue

        if entry.is_dir():
            bank: DataBank = FilesDataBank(name=entry.name, root=entry, extensions=extensions)
            result[entry.name] = bank
            logger.debug("[rag/databanks] discovered FilesDataBank: %s", entry.name)

        elif entry.is_file() and entry.suffix.lower() == ".json":
            bank = LoreBookDataBank(name=entry.stem, path=entry)
            result[entry.stem] = bank
            logger.debug("[rag/databanks] discovered LoreBookDataBank: %s", entry.stem)

    return result
