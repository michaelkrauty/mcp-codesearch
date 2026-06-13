"""Query preprocessing: synonym expansion, syntax parsing, normalization."""

from __future__ import annotations

import functools
import re

from pydantic import BaseModel, Field


@functools.lru_cache(maxsize=64)
def _compile_exclude_regex(exclude_tuple: tuple[str, ...]) -> re.Pattern[str]:
    """Compile exclude paths regex with caching for performance."""
    return re.compile("|".join(re.escape(e) for e in exclude_tuple))

# Code-aware synonym mappings
CODE_SYNONYMS: dict[str, list[str]] = {
    # Common abbreviations
    "fn": ["function"],
    "func": ["function"],
    "cls": ["class"],
    "err": ["error", "exception"],
    "msg": ["message"],
    "req": ["request"],
    "res": ["response"],
    "resp": ["response"],
    "params": ["parameters"],
    "args": ["arguments"],
    "kwargs": ["keyword arguments"],
    "ret": ["return"],
    "val": ["value"],
    "var": ["variable"],
    "impl": ["implementation"],
    "init": ["initialize", "constructor"],
    "ctor": ["constructor"],
    "dtor": ["destructor"],
    "cfg": ["config", "configuration"],
    "conf": ["config", "configuration"],
    "db": ["database"],
    "auth": ["authentication", "authorization"],
    "ctx": ["context"],
    "env": ["environment"],
    "pkg": ["package"],
    "dep": ["dependency"],
    "deps": ["dependencies"],
    "util": ["utility"],
    "utils": ["utilities"],
    "str": ["string"],
    "int": ["integer"],
    "bool": ["boolean"],
    "dict": ["dictionary"],
    "arr": ["array"],
    "idx": ["index"],
    "len": ["length"],
    "num": ["number"],
    "max": ["maximum"],
    "min": ["minimum"],
    "avg": ["average"],
    "src": ["source"],
    "dst": ["destination"],
    "tmp": ["temporary"],
    "async": ["asynchronous"],
    "sync": ["synchronous"],
    # Network & Protocol
    "ws": ["websocket"],
    "http": ["protocol", "request"],
    "https": ["protocol", "secure"],
    "tcp": ["protocol", "socket"],
    "udp": ["protocol", "datagram"],
    "rpc": ["remote procedure call"],
    "grpc": ["remote procedure call"],
    "api": ["interface", "endpoint"],
    "rest": ["representational state transfer"],
    "url": ["uniform resource locator", "address"],
    "uri": ["uniform resource identifier"],
    # I/O & Streams
    "io": ["input output"],
    "stdin": ["standard input"],
    "stdout": ["standard output"],
    "stderr": ["standard error"],
    "stream": ["input output", "data flow"],
    "buf": ["buffer"],
    "buffer": ["data storage", "temporary"],
    # Data Structures
    "obj": ["object"],
    "tuple": ["collection", "pair"],
    "set": ["collection", "unique"],
    "map": ["dictionary", "key value"],
    "hashmap": ["dictionary", "key value"],
    "heap": ["tree", "priority"],
    "queue": ["list", "fifo"],
    "stack": ["list", "lifo"],
    "vec": ["vector", "array"],
    "ptr": ["pointer", "reference"],
    "ref": ["reference"],
    # Concurrency & Threading
    "mutex": ["lock", "synchronization"],
    "semaphore": ["synchronization", "signal"],
    "thread": ["concurrent", "parallel"],
    "task": ["unit of work", "job"],
    "worker": ["thread", "background"],
    "callback": ["handler", "listener"],
    "promise": ["future", "async"],
    "future": ["promise", "async"],
    "spawn": ["create", "launch"],
    "chan": ["channel", "communication"],
    # Testing
    "test": ["verify", "check"],
    "mock": ["fake", "stub"],
    "stub": ["fake", "mock"],
    "spy": ["monitor", "track"],
    "assert": ["verify", "check"],
    "expect": ["verify", "should"],
    "spec": ["specification", "test"],
    "e2e": ["end to end", "integration"],
    "unit": ["single", "isolated"],
    # Error Handling
    "panic": ["crash", "fatal"],
    "throw": ["raise", "error"],
    "catch": ["handle", "trap"],
    "try": ["attempt", "handle"],
    "except": ["catch", "handle"],
    "finally": ["cleanup", "always"],
    # Memory & Performance
    "alloc": ["allocate", "memory"],
    "dealloc": ["deallocate", "free"],
    "gc": ["garbage collection"],
    "cache": ["memory", "store"],
    "mem": ["memory"],
    "perf": ["performance"],
    "opt": ["optimize", "optimization"],
    "bench": ["benchmark", "performance"],
    # Common Patterns
    "singleton": ["single instance", "pattern"],
    "factory": ["create", "pattern"],
    "builder": ["construct", "pattern"],
    "handler": ["process", "callback"],
    "listener": ["event", "callback"],
    "observer": ["watch", "pattern"],
    "middleware": ["interceptor", "handler"],
    # Misc
    "regex": ["regular expression", "pattern"],
    "json": ["javascript object notation"],
    "xml": ["extensible markup language"],
    "yaml": ["configuration", "markup"],
    "toml": ["configuration"],
    "sql": ["structured query language", "database"],
    "orm": ["object relational mapping"],
    "crud": ["create read update delete"],
}


class ParsedQuery(BaseModel):
    """Parsed query with extracted structured components."""

    text: str  # Remaining natural language text
    function_name: str | None = None
    class_name: str | None = None
    file_pattern: str | None = None
    path_prefix: str | None = None
    chunk_type: str | None = None  # function, class, etc.
    exclude_paths: list[str] = Field(default_factory=list)
    inferred_language: str | None = None  # Auto-detected from query hints
    scope: str | None = None  # Scope filter: function, class, test, impl (non-test)

    # Cached compiled regex for exclude_paths (set lazily)
    _exclude_paths_regex: re.Pattern[str] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get_exclude_paths_regex(self) -> re.Pattern[str] | None:
        """Get compiled regex for exclude_paths filtering (cached globally)."""
        if not self.exclude_paths:
            return None
        if self._exclude_paths_regex is None:
            # Use module-level LRU cache for cross-instance reuse
            self._exclude_paths_regex = _compile_exclude_regex(
                tuple(self.exclude_paths)
            )
        return self._exclude_paths_regex


# Language inference patterns: (regex pattern, language)
# These detect language-specific constructs in the query
LANGUAGE_HINTS: list[tuple[str, str]] = [
    # TypeScript/JavaScript React hooks and patterns
    (r'\buse(Effect|State|Context|Memo|Callback|Ref|Reducer|LayoutEffect)\b', 'typescript'),
    (r'\buseSelector\b|\buseDispatch\b', 'typescript'),  # Redux
    (r'\bReact\.(Component|useState|useEffect)', 'typescript'),
    (r'\bJSX\b|\bTSX\b', 'typescript'),
    (r'\.tsx?\b', 'typescript'),
    # Python-specific
    (r'\b__init__\b|\b__main__\b|\b__name__\b', 'python'),
    (r'\bself\.\w+', 'python'),
    (r'\bdef\s+\w+\s*\(self', 'python'),
    (r'\basyncio\b|\bawait\s+\w+\(', 'python'),
    (r'\.py\b', 'python'),
    # Rust-specific
    (r'\bimpl\s+\w+', 'rust'),
    (r'\b(Option|Result|Vec|String)::', 'rust'),
    (r'\bunwrap\(\)|\bexpect\(', 'rust'),
    (r'\bfn\s+\w+\s*<', 'rust'),
    (r'\.rs\b', 'rust'),
    # Go-specific
    (r'\bfunc\s+\(\w+\s+\*?\w+\)', 'go'),
    (r'\bgoroutine\b|\bchannel\b', 'go'),
    (r'\bgo\s+func\b', 'go'),
    (r'\.go\b', 'go'),
    # Java-specific
    (r'\bpublic\s+class\b|\bprivate\s+void\b', 'java'),
    (r'\bextends\s+\w+\s+implements\b', 'java'),
    (r'\.java\b', 'java'),
    # C/C++-specific
    (r'\b(std::|vector<|unique_ptr<|shared_ptr<)', 'cpp'),
    (r'\b#include\s*<\w+>', 'cpp'),
    (r'\.(cpp|hpp|cc|hh)\b', 'cpp'),
]


def infer_language(query: str) -> str | None:
    """
    Infer programming language from query text patterns.

    Returns the detected language or None if no strong signal found.
    """
    for pattern, language in LANGUAGE_HINTS:
        if re.search(pattern, query, re.IGNORECASE):
            return language
    return None


def expand_camelcase(query: str) -> str:
    """
    Expand camelCase and PascalCase to space-separated words.

    Example: "getUserData" -> "get User Data"
    Example: "XMLParser" -> "XML Parser"
    Example: "parseHTTPResponse" -> "parse HTTP Response"

    Preserves original query and appends expanded version.
    """
    # Split on lowercase->uppercase boundary: getUserData -> get User Data
    expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
    # Split on uppercase->uppercase+lowercase: XMLParser -> XML Parser
    expanded = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', expanded)

    if expanded != query:
        # Return both original and expanded for better matching
        return f"{query} {expanded.lower()}"
    return query


def expand_synonyms(
    query: str,
    max_expansions: int = 5,
    max_synonyms_per_word: int = 2,
) -> str:
    """
    Expand query with code-relevant synonyms.

    Example: "fn error handling" -> "fn function error exception handling"
    Example: "fn(err) => msg" -> "fn function (err error exception ) => msg message"

    Handles words with attached punctuation by extracting word tokens.

    Args:
        query: The search query to expand
        max_expansions: Maximum number of words to expand (performance guard)
        max_synonyms_per_word: Maximum synonyms to add per word (limits query explosion)

    Returns:
        Query string with synonyms appended to expandable words
    """
    # Find all word tokens and their positions
    tokens = list(re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', query.lower()))

    if not tokens:
        return query

    # Build result by processing each token and preserving non-token characters
    result_parts = []
    last_end = 0
    expansions_made = 0

    for match in tokens:
        # Add any characters between last token and this one
        if match.start() > last_end:
            result_parts.append(query[last_end:match.start()])

        word = match.group(1)
        # Add the word itself
        result_parts.append(word)
        # Add synonyms if any (respecting limits)
        if word in CODE_SYNONYMS and expansions_made < max_expansions:
            synonyms = CODE_SYNONYMS[word][:max_synonyms_per_word]
            result_parts.append(" " + " ".join(synonyms))
            expansions_made += 1

        last_end = match.end()

    # Add any trailing characters
    if last_end < len(query):
        result_parts.append(query[last_end:])

    return "".join(result_parts)


def parse_query(query: str) -> ParsedQuery:
    """
    Parse structured search syntax from query.

    Supported syntax:
      - function:name, fn:name, def:name, method:name
      - class:name, cls:name, type:name
      - file:pattern
      - path:prefix
      - -path:exclude (negative)

    Example: "function:processData error handling"
           -> ParsedQuery(text="error handling", function_name="processData")
    """
    result = ParsedQuery(text=query)
    remaining = query

    # function:name or fn:name or def:name or method:name
    match = re.search(r'\b(?:function|fn|def|method):(\w+)', remaining, re.I)
    if match:
        result.function_name = match.group(1)
        result.chunk_type = "function"
        remaining = remaining[:match.start()] + remaining[match.end():]

    # class:name or cls:name or type:name or struct:name (for Rust)
    match = re.search(r'\b(?:class|cls|type|struct):(\w+)', remaining, re.I)
    if match:
        result.class_name = match.group(1)
        result.chunk_type = "class"  # Treat struct as class for filtering purposes
        remaining = remaining[:match.start()] + remaining[match.end():]

    # file:pattern
    match = re.search(r'\bfile:(\S+)', remaining, re.I)
    if match:
        result.file_pattern = match.group(1)
        remaining = remaining[:match.start()] + remaining[match.end():]

    # -path:exclude (can have multiple) - MUST be processed before path: to avoid conflicts
    for match in re.finditer(r'-path:(\S+)', remaining, re.I):
        result.exclude_paths.append(match.group(1))
    remaining = re.sub(r'-path:\S+', '', remaining, flags=re.I)

    # path:prefix (after -path: to avoid matching inside -path:xxx)
    match = re.search(r'\bpath:(\S+)', remaining, re.I)
    if match:
        result.path_prefix = match.group(1)
        remaining = remaining[:match.start()] + remaining[match.end():]

    # scope:type - filter by chunk type (function, class, test, impl)
    match = re.search(r'\bscope:(function|class|method|test|impl)\b', remaining, re.I)
    if match:
        scope_val = match.group(1).lower()
        # Normalize: method -> function
        if scope_val == "method":
            scope_val = "function"
        result.scope = scope_val
        remaining = remaining[:match.start()] + remaining[match.end():]

    result.text = remaining.strip()
    return result


# Mirrors QdrantStorage._TEXT_INDEX_MAX_TOKEN_LEN (storage/qdrant.py): the
# full-text payload indexes never index tokens longer than this, so a
# MatchText query containing a longer token matches NOTHING at all. Any
# pushdown token above this length must be dropped (dropping a token only
# widens the candidate superset, which is always safe).
_MAX_INDEXED_TOKEN_LEN = 64


def file_pattern_pushdown_tokens(pattern: str) -> list[str] | None:  # noqa: PLR0912
    """Extract path tokens guaranteed to appear in any ``file:`` match.

    Given an fnmatch pattern applied to a FILENAME, return lowercased
    alphanumeric tokens that are guaranteed to appear as whole tokens
    (word-tokenizer semantics: maximal alphanumeric runs) in the full
    PATH of every file whose filename matches the pattern. The result
    can therefore be used as a Qdrant ``MatchText`` pre-filter on the
    indexed ``path`` field: it may admit extra candidates, but it can
    never exclude a true match (a superset prefilter).

    Why filename tokens are valid against the full path: the filename is
    a complete path component, so a token boundary at pattern start or
    pattern end becomes ``/`` or string-end in the path — still a token
    boundary under the word tokenizer.

    Wildcards are ``*``, ``?``, and any ``[...]`` character class (an
    unmatched ``[`` is treated conservatively as a wildcard). An
    alphanumeric run in the literal parts qualifies only when both of
    its neighbors are literal non-alphanumeric characters or the pattern
    boundary; a run adjacent to a wildcard is not guaranteed to be a
    whole token in the matched filename and is dropped. Tokens longer
    than the indexed token limit are dropped too (they are never
    indexed; see _MAX_INDEXED_TOKEN_LEN).

    Returns None when no tokens survive, meaning no pushdown is possible.
    """
    # Flatten the pattern into elements: a literal character (str) or a
    # wildcard (None). Each [...] class collapses into one wildcard.
    elements: list[str | None] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c in "*?":
            elements.append(None)
            i += 1
        elif c == "[":
            # Find the closing bracket using fnmatch rules: a ']' that
            # appears first (optionally after '!') belongs to the set.
            j = i + 1
            if j < n and pattern[j] == "!":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j < n:
                elements.append(None)
                i = j + 1
            else:
                # Unmatched '[': fnmatch treats it as a literal, but a
                # wildcard is the conservative (superset-safe) reading.
                elements.append(None)
                i += 1
        else:
            elements.append(c)
            i += 1

    tokens: list[str] = []
    k = 0
    m = len(elements)
    while k < m:
        e = elements[k]
        if isinstance(e, str) and e.isalnum():
            start = k
            run_chars: list[str] = []
            while k < m:
                ch = elements[k]
                if isinstance(ch, str) and ch.isalnum():
                    run_chars.append(ch)
                    k += 1
                else:
                    break
            # The run is maximal, so a literal neighbor is necessarily
            # non-alphanumeric; only a wildcard neighbor (None) is unsafe.
            left_ok = start == 0 or isinstance(elements[start - 1], str)
            right_ok = k == m or isinstance(elements[k], str)
            if left_ok and right_ok:
                token = "".join(run_chars).lower()
                if len(token) <= _MAX_INDEXED_TOKEN_LEN:
                    tokens.append(token)
        else:
            k += 1

    return tokens or None


def preprocess_query(query: str, expand: bool = True) -> tuple[str, ParsedQuery]:
    """
    Full query preprocessing pipeline.

    Args:
        query: Raw query string
        expand: Whether to expand synonyms

    Returns:
        Tuple of (processed_text_for_embedding, parsed_query)
    """
    # Parse structured syntax first
    parsed = parse_query(query)

    # Infer language from query patterns (e.g., useEffect → TypeScript)
    parsed.inferred_language = infer_language(query)

    # Build the text to embed
    text_parts = [parsed.text]

    # Include structured terms in embedding text for semantic matching
    if parsed.function_name:
        text_parts.append(f"function {parsed.function_name}")
    if parsed.class_name:
        text_parts.append(f"class {parsed.class_name}")

    text_for_embedding = " ".join(text_parts)

    # Expand camelCase/PascalCase first (before synonym expansion)
    if expand:
        text_for_embedding = expand_camelcase(text_for_embedding)
        text_for_embedding = expand_synonyms(text_for_embedding)

    return text_for_embedding, parsed
