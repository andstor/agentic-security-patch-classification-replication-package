import os
import re
from tree_sitter import Language, Parser



# Path to the compiled shared library
SO_PATH = 'build/my-languages.so'

# Tree-sitter language sources
LANGUAGE_SOURCES = [
    'vendor/tree-sitter-c',
    'vendor/tree-sitter-cpp',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-php/php',
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-go',
]

# Build the library only if it doesn't exist
if not os.path.exists(SO_PATH):
    os.makedirs(os.path.dirname(SO_PATH), exist_ok=True)
    Language.build_library(
        SO_PATH,
        LANGUAGE_SOURCES
    )

LANGUAGES = {
    "c": Language('build/my-languages.so', 'cpp'),  # Using C++ grammar for C as well
    "cpp": Language('build/my-languages.so', 'cpp'),
    "python": Language('build/my-languages.so', 'python'),
    "php": Language('build/my-languages.so', 'php'),
    "java": Language('build/my-languages.so', 'java'),
    "go": Language('build/my-languages.so', 'go'),
}

# Per-language function declaration rules
FUNC_RULES = {
    "python": {
        "nodes": ["function_definition"],
        "identifier": ["identifier"],
    },
    "c": {
        "nodes": ["function_definition"],
        "identifier": ["function_declarator > identifier"],
    },
    "cpp": {
        "nodes": ["function_definition"],
        "identifier": ["function_declarator > identifier"],
    },
    "php": {
        "nodes": ["function_definition", "function_declaration"],
        "identifier": ["name", "identifier"],
    },
    "java": {
        "nodes": ["method_declaration"],
        "identifier": ["identifier"],
    },
    "go": {
        "nodes": ["function_declaration"],
        "identifier": ["identifier"],
    },
}
# Map extensions without dot to language keys
EXT_TO_LANG = {
    "c": "c",
    "h": "c",       # header files treated as C
    "cpp": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "hpp": "cpp",
    "py": "python",
    "php": "php",
    "java": "java",
    "go": "go",
}

def detect_language(file_path):
    ext = os.path.splitext(file_path)[1]  # includes the dot
    ext = ext[1:].lower() if ext.startswith('.') else ext.lower()  # remove dot
    return EXT_TO_LANG.get(ext)

def get_func(file_path, line_number):
    lang = detect_language(file_path)
    if not lang:
        raise ValueError(f"Cannot detect language for file: {file_path}")

    parser = Parser()
    parser.set_language(LANGUAGES[lang])

    with open(file_path, 'r', errors='ignore') as f:
        code = f.read()
    tree = parser.parse(bytes(code, "utf-8"))
    code_lines = code.split("\n")

    return find_function_define(tree.root_node, code_lines, line_number, lang)


def find_function_define(node, code_lines, line_number, lang):
    rules = FUNC_RULES[lang]

    # If this node is a function/method definition for this language
    if node.type in rules["nodes"]:
        start, end = node.start_point[0], node.end_point[0]

        if start <= line_number <= end:
            function_code = "\n".join(code_lines[start:end + 1])

            # Extract identifier (only direct descendants or specified path)
            function_name = extract_identifier(node, code_lines, rules)

            return function_code, function_name

    # Recurse children
    for child in node.children:
        function_code, function_name = find_function_define(child, code_lines, line_number, lang)
        if function_name != "NULL":
            return function_code, function_name

    return "NULL", "NULL"


def extract_identifier(node, code_lines, rules):
    """
    Extract function name based on language-specific identifier rules.
    """
    for id_type in rules["identifier"]:
        # Direct children only
        for child in node.children:
            if child.type == id_type:
                return get_node_text(child, code_lines)

            # Handle one-level nested like C++ (function_declarator > identifier)
            if ">" in id_type:
                parent, child_type = id_type.split(">")
                parent = parent.strip()
                child_type = child_type.strip()
                if child.type == parent:
                    for grandchild in child.children:
                        if grandchild.type == child_type:
                            return get_node_text(grandchild, code_lines)

    return "NULL"


def get_node_text(node, code_lines):
    """Extract the exact source text for a node from code_lines."""
    start_line, start_col = node.start_point
    end_line, end_col = node.end_point

    if start_line == end_line:
        return code_lines[start_line][start_col:end_col]
    else:
        # multiline identifier (rare but possible)
        text = [code_lines[start_line][start_col:]]
        for i in range(start_line + 1, end_line):
            text.append(code_lines[i])
        text.append(code_lines[end_line][:end_col])
        return "".join(text)