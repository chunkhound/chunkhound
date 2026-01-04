"""Tree formatting utilities for hierarchical displays.

Provides reusable functions for rendering tree structures with box-drawing characters.
Used by both terminal progress displays (TreeProgressDisplay) and markdown output
formatters (sources footer in synthesis).

Key Functions:
    build_tree_prefix(depth, is_last): Generate tree prefix with │ ├ └ characters
    format_tree_item(depth, label, is_last): Format single tree item with prefix
    build_file_hierarchy_tree(paths): Convert flat file list to nested tree structure

Design:
    All functions are pure/stateless for easy testing and reuse. No class dependencies
    or instance methods - these are utility functions that can be called directly.

Usage Examples:
    >>> # Build tree prefix for depth 2, last child
    >>> prefix = build_tree_prefix(2, is_last=True)
    '│   └── '

    >>> # Format a tree item
    >>> item = format_tree_item(1, "file.py", is_last=False)
    '├── file.py'

    >>> # Build hierarchy from file paths
    >>> paths = ['src/api/routes.py', 'src/api/models.py', 'tests/test_api.py']
    >>> tree = build_file_hierarchy_tree(paths)
    >>> for name, depth, is_last in tree:
    ...     print(format_tree_item(depth, name, is_last))
    src/
    ├── api/
    │   ├── routes.py
    │   └── models.py
    └── tests/
        └── test_api.py
"""


def build_tree_prefix(depth: int, is_last: bool = False) -> str:
    """Build tree prefix with box-drawing characters.

    Args:
        depth: Node depth level (0 = root, 1 = first level, etc.)
        is_last: Whether this is the last child at this level

    Returns:
        Tree prefix string with proper indentation using box-drawing chars

    Examples:
        >>> build_tree_prefix(0)
        ''
        >>> build_tree_prefix(1, is_last=False)
        '├── '
        >>> build_tree_prefix(1, is_last=True)
        '└── '
        >>> build_tree_prefix(2, is_last=False)
        '│   ├── '
        >>> build_tree_prefix(2, is_last=True)
        '│   └── '
    """
    if depth == 0:
        return ""

    # Build prefix based on depth
    prefix_parts = []
    for _ in range(depth - 1):
        prefix_parts.append("│   ")

    # Add connector for current level
    if is_last:
        prefix_parts.append("└── ")
    else:
        prefix_parts.append("├── ")

    return "".join(prefix_parts)


def format_tree_item(depth: int, label: str, is_last: bool = False) -> str:
    """Format a single tree item with prefix and label.

    Args:
        depth: Node depth level
        label: Item label text
        is_last: Whether this is the last child at this level

    Returns:
        Formatted tree line with prefix and label

    Examples:
        >>> format_tree_item(0, "root")
        'root'
        >>> format_tree_item(1, "child1", is_last=False)
        '├── child1'
        >>> format_tree_item(1, "child2", is_last=True)
        '└── child2'
    """
    prefix = build_tree_prefix(depth, is_last)
    return f"{prefix}{label}"


def build_file_hierarchy_tree(file_paths: list[str]) -> list[tuple[str, int, bool]]:
    """Build hierarchical tree structure from flat file paths.

    Groups files by directory and determines depth/is_last status for each entry.

    Args:
        file_paths: List of file paths (e.g., ['src/a.py', 'src/b.py', 'tests/c.py'])

    Returns:
        List of tuples: (display_name, depth, is_last)
            - display_name: Directory or file name to display
            - depth: Tree depth level
            - is_last: Whether this is the last item at its level

    Examples:
        >>> build_file_hierarchy_tree(['src/services/a.py', 'src/api/b.py'])
        [
            ('src/', 0, False),
            ('├── services/', 1, False),
            ('│   └── a.py', 2, True),
            ('└── api/', 1, True),
            ('    └── b.py', 2, True),
        ]
    """
    if not file_paths:
        return []

    # Build directory tree structure
    tree: dict[str, dict] = {}
    for path in sorted(file_paths):
        parts = path.split("/")
        current = tree
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {}
            current = current[part]

    # Flatten tree to list with depth and is_last markers
    result: list[tuple[str, int, bool]] = []

    def traverse(
        node: dict, depth: int, name: str = "", is_last_child: bool = False
    ) -> None:
        """Recursively traverse tree and build flat list with depth markers."""
        if not node:
            # Leaf node (file)
            return

        items = list(node.items())
        for idx, (key, children) in enumerate(items):
            is_last = idx == len(items) - 1
            is_directory = bool(children)

            # Format display name
            if is_directory:
                display_name = f"{key}/"
            else:
                display_name = key

            # Add to result
            result.append((display_name, depth, is_last))

            # Recurse for children
            if children:
                traverse(children, depth + 1, key, is_last)

    # Start traversal from root
    traverse(tree, 0)

    return result
