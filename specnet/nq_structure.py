"""Gold-independent, source-mapped rendering of a native NQ HTML page.

Native NQ document tokens omit many inline tags.  We therefore recover layout
from the original HTML, but spell every visible word using the official token
string.  Answer annotations and the question are deliberately not inputs.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
import re


STRUCTURE_VERSION = "nq-safe-html-source-mapped-v1"
SAFE_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li",
             "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
             "b", "strong", "i", "em", "sup", "sub", "small", "s", "del", "u",
             "code", "blockquote", "dl", "dt", "dd", "br"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd",
              "caption", "blockquote"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}


@dataclass(eq=False)
class Word:
    index: int
    text: str
    parent: object = None


@dataclass(eq=False)
class Element:
    tag: str
    attrs: dict = field(default_factory=dict)
    parent: object = None
    children: list = field(default_factory=list)
    words: list = field(default_factory=list)
    column: int | None = None


class _Tree(HTMLParser):
    """Attach official words to the original DOM using byte coordinates."""
    def __init__(self, source, tokens, visible):
        super().__init__(convert_charrefs=False)
        self.source, self.tokens, self.visible = source, tokens, visible
        self.lines = [0] + [m.end() for m in re.finditer("\n", source)]
        self.root = Element("root")
        self.stack = [self.root]
        self.elements, self.words = [], []
        self.cursor = self.previous_char = self.previous_byte = 0

    def _position(self):
        line, column = self.getpos()
        position = self.lines[line - 1] + column
        self.previous_byte += len(self.source[self.previous_char:position].encode("utf-8"))
        self.previous_char = position
        return self.previous_byte

    def _flush(self, position):
        while self.cursor < len(self.visible):
            index, text = self.visible[self.cursor]
            if self.tokens[index]["start_byte"] >= position:
                break
            word = Word(index, text, self.stack[-1])
            self.stack[-1].children.append(word)
            self.words.append(word)
            for element in self.stack:
                element.words.append(word)
            self.cursor += 1

    def handle_starttag(self, tag, attrs):
        self._flush(self._position())
        element = Element(tag, dict(attrs), self.stack[-1])
        self.stack[-1].children.append(element)
        self.elements.append(element)
        if tag not in VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self._flush(self._position())
        element = Element(tag, dict(attrs), self.stack[-1])
        self.stack[-1].children.append(element)
        self.elements.append(element)

    def handle_endtag(self, tag):
        self._flush(self._position())
        match = next((i for i in range(len(self.stack) - 1, 0, -1)
                      if self.stack[i].tag == tag), None)
        if match is not None:
            del self.stack[match:]

    def finish(self):
        self._flush(float("inf"))
        return self


def _ancestors(node):
    while node is not None:
        yield node
        node = node.parent


def _nearest(node, tags):
    return next((p for p in _ancestors(node) if isinstance(p, Element) and p.tag in tags), None)


def _integer(value, default=1):
    try:
        result = int(value)
        return result if 1 <= result <= 10000 else default
    except (TypeError, ValueError):
        return default


def _opening(element):
    attrs = []
    if element.tag in {"td", "th"}:
        for name in ("rowspan", "colspan"):
            value = _integer(element.attrs.get(name))
            if value > 1:
                attrs.append(f'{name}="{value}"')
        if element.column is not None:
            attrs.append(f'data-column="{element.column + 1}"')
    # A highlighted row can carry meaning (e.g. an award winner). Preserve the
    # source color without interpreting it as an answer or importing CSS/URLs.
    for declaration in (element.attrs.get("style") or "").split(";"):
        key, separator, value = declaration.partition(":")
        if (separator and key.strip().lower() in {"background", "background-color", "color"}
                and re.fullmatch(r"#[0-9a-fA-F]{3,8}|[A-Za-z]{1,24}", value.strip())):
            attrs.append(f'data-{key.strip().lower()}="{value.strip()}"')
    return "<" + element.tag + (" " + " ".join(attrs) if attrs else "") + ">"


class _Writer:
    def __init__(self):
        self.parts, self.offsets = [], []
        self.length = 0

    def add(self, text):
        self.parts.append(text)
        self.length += len(text)

    def word(self, word):
        # Native token spelling is also used by the answer reference renderer.
        self.add(" ")
        start = self.length
        self.add(word.text)
        self.offsets.append(dict(token_index=word.index, start=start, end=self.length))

    def render(self, node, selected, bounds=None):
        if bounds is None:
            bounds = (min(selected), max(selected))
        if isinstance(node, Word):
            if node.index in selected:
                self.word(node)
            return
        if node.words and (node.words[-1].index < bounds[0] or node.words[0].index > bounds[1]):
            return
        if not any(word.index in selected for word in node.words):
            # Empty cells still occupy columns; br preserves a source line break.
            if not node.words and node.tag in {"td", "th", "br"}:
                self.add(_opening(node))
                if node.tag != "br":
                    self.add("</" + node.tag + ">")
            return
        if node.tag in SAFE_TAGS:
            self.add(_opening(node))
        for child in node.children:
            self.render(child, selected, bounds)
        if node.tag in SAFE_TAGS and node.tag not in VOID_TAGS:
            self.add("</" + node.tag + ">")

    def fragment(self, node, words):
        selected = {word.index for word in words}
        # Reopen structural ancestors, but never unrelated sibling content.
        parents = [p for p in _ancestors(node.parent) if p.tag in SAFE_TAGS]
        for parent in reversed(parents):
            self.add(_opening(parent))
        self.render(node, selected)
        for parent in parents:
            if parent.tag not in VOID_TAGS:
                self.add("</" + parent.tag + ">")
        self.add("\n")


def _table_context(tree):
    """Find explicit table headers and inherited rowspan cells, without labels."""
    contexts, rows_by_table, cells_by_row = {}, defaultdict(list), defaultdict(list)
    for element in tree.elements:
        if element.tag == "tr":
            rows_by_table[_nearest(element.parent, {"table"})].append(element)
        elif element.tag in {"td", "th"}:
            cells_by_row[_nearest(element.parent, {"tr"})].append(element)
    for table in [e for e in tree.elements if e.tag == "table" and e.words]:
        rows = rows_by_table[table]
        headers, active, header_phase = [], [], True
        for row in rows:
            cells = cells_by_row[row]
            inherited = [cell for _, _, cell, _ in active]
            occupied = {column for start, width, _, _ in active
                        for column in range(start, start + width)}
            next_active = [(start, width, cell, remaining - 1)
                           for start, width, cell, remaining in active if remaining > 1]
            column = 0
            for cell in cells:
                while column in occupied:
                    column += 1
                width = _integer(cell.attrs.get("colspan"))
                cell.column = column
                span = _integer(cell.attrs.get("rowspan"))
                if span > 1:
                    next_active.append((column, width, cell, span - 1))
                column += width
            is_header = bool(cells) and all(cell.tag == "th" for cell in cells)
            if header_phase and is_header:
                headers.append(row)
            elif cells:
                header_phase = False
            contexts[row] = dict(headers=list(headers), inherited=inherited, table=table)
            active = next_active
    return contexts


def _units(tree):
    """Use table rows and text blocks as preferred, not mandatory, boundaries."""
    units = []
    for word in tree.words:
        # A paragraph/list inside a cell belongs to its row, preserving columns.
        boundary = (_nearest(word.parent, {"tr"})
                    or _nearest(word.parent, BLOCK_TAGS) or word.parent)
        if not units or units[-1]["node"] is not boundary:
            units.append(dict(node=boundary, words=[]))
        units[-1]["words"].append(word)
    headings = []
    for unit in units:
        node = unit["node"]
        if node.tag in HEADINGS:
            level = int(node.tag[1])
            headings = [h for h in headings if int(h.tag[1]) < level] + [node]
        unit["headings"] = list(headings)
    return units


def structured_passages(row, visible, *, passage_words):
    """Return bounded, complete page chunks plus exact source-to-text offsets.

    The bound counts visible official-token occurrences, including repeated
    context. Layout markers are extra characters, not silently counted as source
    evidence. Context copies may be shortened; every primary source token still
    appears in full somewhere in the candidate pool.
    """
    tree = _Tree(row["document_html"], row["document_tokens"], visible)
    tree.feed(row["document_html"])
    tree.close()
    tree.finish()
    contexts = _table_context(tree)
    context_limit = min(96, passage_words // 4)
    primary_limit = passage_words - context_limit
    units = _units(tree)
    pieces = []
    for unit in units:
        for start in range(0, len(unit["words"]), primary_limit):
            pieces.append({**unit, "words": unit["words"][start:start + primary_limit],
                           "continuation": start > 0,
                           "preceding": unit["words"][:start]})
    groups, current, count = [], [], 0
    for piece in pieces:
        if current and count + len(piece["words"]) > primary_limit:
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += len(piece["words"])
    if current:
        groups.append(current)
    features = sorted({e.tag for e in tree.elements if e.words and e.tag in SAFE_TAGS})
    if any(_integer(e.attrs.get("rowspan")) > 1 for e in tree.elements if e.words):
        features.append("rowspan")
    if any("background" in (e.attrs.get("style") or "").lower() for e in tree.elements if e.words):
        features.append("source_background_color")
    result = []
    for passage_index, group in enumerate(groups):
        first = group[0]
        primary_words = [word for piece in group for word in piece["words"]]
        primary_indices = {word.index for word in primary_words}
        context_nodes = []
        table = contexts.get(first["node"], {})
        # Carry the year/group cell before the table header when a row starts
        # after a rowspan's original cell. This is literal source text only.
        context_nodes.extend((cell, cell.words, "inherited table cell")
                             for cell in table.get("inherited", []))
        context_nodes.extend((header, header.words, "table header")
                             for header in table.get("headers", []))
        if first["continuation"] and first["node"].tag == "tr":
            context_nodes.append((first["node"], first["preceding"], "earlier cells in continued row"))
        context_nodes.extend((heading, heading.words, "section heading")
                             for heading in first["headings"][-2:])
        writer = _Writer()
        writer.add(f"[Page passage {passage_index + 1}; source tokens "
                   f"{primary_words[0].index}:{primary_words[-1].index + 1}]\n")
        used, context_count, context_truncated = set(primary_indices), 0, False
        for node, words, label in context_nodes:
            available = [word for word in words if word.index not in used]
            selected = available[:max(0, context_limit - context_count)]
            if len(selected) < len(available):
                context_truncated = True
            if selected:
                writer.add("[Repeated source context: " + label + "]\n")
                writer.fragment(node, selected)
                used.update(word.index for word in selected)
                context_count += len(selected)
        writer.add("[Page content" + ("; continued structural block" if first["continuation"] else "") + "]\n")
        # Render a single balanced subtree per chunk. Opening a new table for
        # each row would erase the relationship expressed by source rowspans.
        writer.render(tree.root, primary_indices)
        writer.add("\n")
        if context_truncated:
            writer.add("[Repeated context shortened; its full source remains in the page candidate pool.]\n")
        text = "".join(writer.parts)
        heading_text = [" ".join(word.text for word in heading.words) for heading in first["headings"]]
        section_kind = ("reference" if any(re.fullmatch(
            r"references?|notes?|sources?|bibliography|external links|further reading",
            re.sub(r"\s*[\[(]\s*edit\s*[\])]\s*$", "", title.strip(), flags=re.I), re.I)
            for title in heading_text) else "body")
        result.append(dict(text=text, token_indices=[offset["token_index"] for offset in writer.offsets],
                           primary_token_indices=[word.index for word in primary_words],
                           rendered_token_offsets=writer.offsets,
                           structure_version=STRUCTURE_VERSION, structure_features=features,
                           section_path=heading_text, section_kind=section_kind,
                           primary_word_count=len(primary_words), visible_word_count=len(writer.offsets),
                           repeated_context_word_count=context_count,
                           context_truncated=context_truncated,
                           continued_structural_block=first["continuation"]))
    expected = [index for index, _ in visible]
    actual = [index for passage in result for index in passage["primary_token_indices"]]
    if actual != expected:
        raise ValueError("Structured NQ rendering did not preserve the complete source token order")
    if any(passage["visible_word_count"] > passage_words for passage in result):
        raise ValueError("Structured NQ rendering exceeded the configured visible-token bound")
    return result
