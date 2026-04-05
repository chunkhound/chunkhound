"""Extract ST code from TcPOU XML files."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .exceptions import XMLExtractionError


@dataclass
class SourceLocation:
    """Position of extracted content in the original XML file."""

    line: int  # 1-indexed line where content starts
    column: int  # 1-indexed column where content starts
    pos: int  # 0-indexed character offset where content starts


@dataclass
class ActionContent:
    """Content extracted from an Action element."""

    name: str
    id: str
    declaration: str | None
    implementation: str
    declaration_location: SourceLocation | None = None
    implementation_location: SourceLocation | None = None


@dataclass
class MethodContent:
    """Content extracted from a Method element."""

    name: str
    id: str
    declaration: str
    implementation: str
    declaration_location: SourceLocation | None = None
    implementation_location: SourceLocation | None = None


@dataclass
class PropertyContent:
    """Content extracted from a Property element."""

    name: str
    id: str
    declaration: str
    get: MethodContent | None = None
    set: MethodContent | None = None
    declaration_location: SourceLocation | None = None


@dataclass
class POUContent:
    """Content extracted from a TcPOU file."""

    name: str
    id: str
    pou_type: str  # PROGRAM, FUNCTION_BLOCK, FUNCTION
    declaration: str
    implementation: str
    actions: list[ActionContent]
    methods: list[MethodContent]
    properties: list[PropertyContent]
    declaration_location: SourceLocation | None = None
    implementation_location: SourceLocation | None = None


class TcPOUExtractor:
    """Extract ST code sections from TcPOU XML files."""

    def extract_file(self, path: str | Path) -> POUContent:
        """Extract all ST content from a TcPOU file.

        Args:
            path: Path to the TcPOU XML file

        Returns:
            POUContent with declaration, implementation, and actions

        Raises:
            XMLExtractionError: If the file cannot be parsed or is invalid
        """
        path = Path(path)
        if not path.exists():
            raise XMLExtractionError(f"File not found: {path}")

        # Read file as text first for position tracking
        xml_text = path.read_text()

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise XMLExtractionError(f"Invalid XML in {path}: {e}")

        return self._extract_from_element(root, str(path), xml_text)

    def extract_string(self, xml_content: str, source: str = "<string>") -> POUContent:
        """Extract all ST content from XML string.

        Args:
            xml_content: TcPOU XML as string
            source: Source identifier for error messages

        Returns:
            POUContent with declaration, implementation, and actions
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise XMLExtractionError(f"Invalid XML in {source}: {e}")

        return self._extract_from_element(root, source, xml_content)

    def _find_cdata_content_start(
        self, xml_text: str, search_start: int, element_path: list[str]
    ) -> SourceLocation | None:
        """Find the position where CDATA content starts for a given element path.

        Args:
            xml_text: The full XML text
            search_start: Position to start searching from
            element_path: List of element names to find
                (e.g., ["Declaration"] or ["Implementation", "ST"])

        Returns:
            SourceLocation with line, column, and pos of content start,
            or None if not found
        """
        pos = search_start

        # Navigate through each element in the path
        for element_name in element_path:
            # Find the opening tag for this element
            # Pattern matches <ElementName or <ElementName> or <ElementName ...>
            tag_pattern = rf"<{element_name}(?:\s[^>]*)?>|<{element_name}>"
            match = re.search(tag_pattern, xml_text[pos:])
            if not match:
                return None
            pos += match.end()

        # Now find the CDATA marker after the last element tag
        cdata_marker = "<![CDATA["
        cdata_pos = xml_text.find(cdata_marker, pos)
        if cdata_pos == -1:
            return None

        # Content starts right after the CDATA marker
        content_start = cdata_pos + len(cdata_marker)

        # Calculate line and column
        # Count newlines before content_start
        text_before = xml_text[:content_start]
        line = text_before.count("\n") + 1

        # Find position of last newline before content_start
        last_newline = text_before.rfind("\n")
        if last_newline == -1:
            column = content_start + 1  # No newline, column is pos + 1 (1-indexed)
        else:
            column = content_start - last_newline  # Distance from last newline

        return SourceLocation(line=line, column=column, pos=content_start)

    def _extract_from_element(
        self, root: ET.Element, source: str, xml_text: str | None = None
    ) -> POUContent:
        """Extract content from parsed XML element."""
        # Find POU element
        pou = root.find("POU")
        if pou is None:
            raise XMLExtractionError(f"No POU element found in {source}")

        name = pou.get("Name")
        if not name:
            raise XMLExtractionError(f"POU missing Name attribute in {source}")

        pou_id = pou.get("Id", "")

        # Find POU start position in XML for location tracking
        pou_search_start = 0
        if xml_text:
            # Find the POU element in the XML text
            pou_match = re.search(rf'<POU\s+Name="{re.escape(name)}"', xml_text)
            if pou_match:
                pou_search_start = pou_match.start()

        # Extract declaration
        decl_elem = pou.find("Declaration")
        if decl_elem is None:
            raise XMLExtractionError(f"No Declaration element found in {source}")
        declaration = decl_elem.text if decl_elem.text is not None else ""

        # Find declaration location
        declaration_location = None
        if xml_text and declaration:
            declaration_location = self._find_cdata_content_start(
                xml_text, pou_search_start, ["Declaration"]
            )

        # Determine POU type from declaration
        pou_type = self._detect_pou_type(declaration)

        # Extract implementation
        impl_elem = pou.find("Implementation")
        if impl_elem is None:
            raise XMLExtractionError(f"No Implementation element found in {source}")

        st_elem = impl_elem.find("ST")
        if st_elem is None or st_elem.text is None:
            implementation = ""
        else:
            implementation = st_elem.text

        # Find implementation location
        implementation_location = None
        if xml_text and implementation:
            implementation_location = self._find_cdata_content_start(
                xml_text, pou_search_start, ["Implementation", "ST"]
            )

        # Extract actions
        actions = []
        # Track position for finding action elements
        action_search_start = pou_search_start
        for action_elem in pou.findall("Action"):
            action = self._extract_action(
                action_elem, source, xml_text, action_search_start
            )
            if action:
                actions.append(action)
                # Move search start past this action for the next one
                if xml_text and action.name:
                    action_match = re.search(
                        rf'<Action\s+Name="{re.escape(action.name)}"',
                        xml_text[action_search_start:],
                    )
                    if action_match:
                        action_search_start += action_match.end()

        # Extract methods
        methods = []
        method_search_start = pou_search_start
        for method_elem in pou.findall("Method"):
            method = self._extract_method(
                method_elem, source, xml_text, method_search_start
            )
            if method:
                methods.append(method)
                # Move search start past this method for the next one
                if xml_text and method.name:
                    method_match = re.search(
                        rf'<Method\s+Name="{re.escape(method.name)}"',
                        xml_text[method_search_start:],
                    )
                    if method_match:
                        method_search_start += method_match.end()

        # Extract properties
        properties = []
        property_search_start = pou_search_start
        for property_elem in pou.findall("Property"):
            prop = self._extract_property(
                property_elem, source, xml_text, property_search_start
            )
            if prop:
                properties.append(prop)
                # Move search start past this property for the next one
                if xml_text and prop.name:
                    property_match = re.search(
                        rf'<Property\s+Name="{re.escape(prop.name)}"',
                        xml_text[property_search_start:],
                    )
                    if property_match:
                        property_search_start += property_match.end()

        return POUContent(
            name=name,
            id=pou_id,
            pou_type=pou_type,
            declaration=declaration,
            implementation=implementation,
            actions=actions,
            methods=methods,
            properties=properties,
            declaration_location=declaration_location,
            implementation_location=implementation_location,
        )

    def _extract_action(
        self,
        action_elem: ET.Element,
        source: str,
        xml_text: str | None = None,
        search_start: int = 0,
    ) -> ActionContent | None:
        """Extract content from an Action element."""
        name = action_elem.get("Name")
        if not name:
            return None

        action_id = action_elem.get("Id", "")

        # Find action start position in XML for location tracking
        action_search_start = search_start
        if xml_text:
            action_match = re.search(
                rf'<Action\s+Name="{re.escape(name)}"', xml_text[search_start:]
            )
            if action_match:
                action_search_start = search_start + action_match.start()

        # Actions may have their own Declaration
        decl_elem = action_elem.find("Declaration")
        declaration = (
            decl_elem.text if decl_elem is not None and decl_elem.text else None
        )

        # Find action declaration location
        declaration_location = None
        if xml_text and declaration:
            declaration_location = self._find_cdata_content_start(
                xml_text, action_search_start, ["Declaration"]
            )

        # Extract implementation
        impl_elem = action_elem.find("Implementation")
        if impl_elem is None:
            return None

        st_elem = impl_elem.find("ST")
        if st_elem is None or st_elem.text is None:
            implementation = ""
        else:
            implementation = st_elem.text

        # Find action implementation location
        implementation_location = None
        if xml_text and implementation:
            implementation_location = self._find_cdata_content_start(
                xml_text, action_search_start, ["Implementation", "ST"]
            )

        return ActionContent(
            name=name,
            id=action_id,
            declaration=declaration,
            implementation=implementation,
            declaration_location=declaration_location,
            implementation_location=implementation_location,
        )

    def _extract_method(
        self,
        method_elem: ET.Element,
        source: str,
        xml_text: str | None = None,
        search_start: int = 0,
    ) -> MethodContent | None:
        """Extract content from a Method element."""
        name = method_elem.get("Name")
        if not name:
            return None

        method_id = method_elem.get("Id", "")

        # Find method start position in XML for location tracking
        method_search_start = search_start
        if xml_text:
            method_match = re.search(
                rf'<Method\s+Name="{re.escape(name)}"', xml_text[search_start:]
            )
            if method_match:
                method_search_start = search_start + method_match.start()

        # Extract declaration
        decl_elem = method_elem.find("Declaration")
        declaration = decl_elem.text if decl_elem is not None and decl_elem.text else ""

        # Find method declaration location
        declaration_location = None
        if xml_text and declaration:
            declaration_location = self._find_cdata_content_start(
                xml_text, method_search_start, ["Declaration"]
            )

        # Extract implementation
        impl_elem = method_elem.find("Implementation")
        if impl_elem is None:
            implementation = ""
        else:
            st_elem = impl_elem.find("ST")
            implementation = ""
            if st_elem is not None and st_elem.text:
                implementation = st_elem.text

        # Find method implementation location
        implementation_location = None
        if xml_text and implementation:
            implementation_location = self._find_cdata_content_start(
                xml_text, method_search_start, ["Implementation", "ST"]
            )

        return MethodContent(
            name=name,
            id=method_id,
            declaration=declaration,
            implementation=implementation,
            declaration_location=declaration_location,
            implementation_location=implementation_location,
        )

    def _extract_property(
        self,
        property_elem: ET.Element,
        source: str,
        xml_text: str | None = None,
        search_start: int = 0,
    ) -> PropertyContent | None:
        """Extract content from a Property element."""
        name = property_elem.get("Name")
        if not name:
            return None

        property_id = property_elem.get("Id", "")

        # Find property start position in XML for location tracking
        property_search_start = search_start
        if xml_text:
            property_match = re.search(
                rf'<Property\s+Name="{re.escape(name)}"', xml_text[search_start:]
            )
            if property_match:
                property_search_start = search_start + property_match.start()

        # Extract declaration
        decl_elem = property_elem.find("Declaration")
        declaration = decl_elem.text if decl_elem is not None and decl_elem.text else ""

        # Find declaration location
        declaration_location = None
        if xml_text and declaration:
            declaration_location = self._find_cdata_content_start(
                xml_text, property_search_start, ["Declaration"]
            )

        # Extract Get accessor
        get_accessor = None
        get_elem = property_elem.find("Get")
        if get_elem is not None:
            get_accessor = self._extract_property_accessor(
                get_elem, "Get", source, xml_text, property_search_start
            )

        # Extract Set accessor
        set_accessor = None
        set_elem = property_elem.find("Set")
        if set_elem is not None:
            set_accessor = self._extract_property_accessor(
                set_elem, "Set", source, xml_text, property_search_start
            )

        return PropertyContent(
            name=name,
            id=property_id,
            declaration=declaration,
            get=get_accessor,
            set=set_accessor,
            declaration_location=declaration_location,
        )

    def _extract_property_accessor(
        self,
        accessor_elem: ET.Element,
        accessor_name: str,
        source: str,
        xml_text: str | None = None,
        search_start: int = 0,
    ) -> MethodContent:
        """Extract content from a Get or Set accessor element."""
        accessor_id = accessor_elem.get("Id", "")

        # Find accessor start position
        accessor_search_start = search_start
        if xml_text:
            accessor_match = re.search(
                rf"<{accessor_name}(?:\s|>)", xml_text[search_start:]
            )
            if accessor_match:
                accessor_search_start = search_start + accessor_match.start()

        # Extract declaration
        decl_elem = accessor_elem.find("Declaration")
        declaration = decl_elem.text if decl_elem is not None and decl_elem.text else ""

        # Find declaration location
        declaration_location = None
        if xml_text and declaration:
            declaration_location = self._find_cdata_content_start(
                xml_text, accessor_search_start, ["Declaration"]
            )

        # Extract implementation
        impl_elem = accessor_elem.find("Implementation")
        if impl_elem is None:
            implementation = ""
        else:
            st_elem = impl_elem.find("ST")
            implementation = ""
            if st_elem is not None and st_elem.text:
                implementation = st_elem.text

        # Find implementation location
        implementation_location = None
        if xml_text and implementation:
            implementation_location = self._find_cdata_content_start(
                xml_text, accessor_search_start, ["Implementation", "ST"]
            )

        return MethodContent(
            name=accessor_name,
            id=accessor_id,
            declaration=declaration,
            implementation=implementation,
            declaration_location=declaration_location,
            implementation_location=implementation_location,
        )

    def _detect_pou_type(self, declaration: str) -> str:
        """Detect POU type from declaration header.

        Skips comments (// and (* ... *)) to find the actual POU keyword.
        """
        # Remove line comments and block comments to find the POU keyword
        cleaned = declaration
        # Remove block comments (* ... *)
        while "(*" in cleaned:
            start = cleaned.find("(*")
            end = cleaned.find("*)", start)
            if end == -1:
                # Unclosed block comment, just remove to end
                cleaned = cleaned[:start]
            else:
                cleaned = cleaned[:start] + cleaned[end + 2 :]
        # Remove line comments
        lines = cleaned.split("\n")
        non_comment_lines = []
        for line in lines:
            # Remove // comments
            comment_pos = line.find("//")
            if comment_pos != -1:
                line = line[:comment_pos]
            if line.strip():
                non_comment_lines.append(line.strip())

        # Find the first non-empty line after removing comments
        cleaned_text = " ".join(non_comment_lines).upper().lstrip()

        if cleaned_text.startswith("PROGRAM"):
            return "PROGRAM"
        elif cleaned_text.startswith("FUNCTION_BLOCK"):
            return "FUNCTION_BLOCK"
        elif cleaned_text.startswith("FUNCTION"):
            return "FUNCTION"
        else:
            # Default to UNKNOWN if not detected
            return "UNKNOWN"
