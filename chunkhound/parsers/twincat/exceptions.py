"""Custom exceptions for the ST parser."""


class STParserError(Exception):
    """Base exception for ST parser errors."""

    def __init__(
        self, message: str, line: int | None = None, column: int | None = None
    ):
        self.line = line
        self.column = column
        location = ""
        if line is not None:
            location = f" at line {line}"
            if column is not None:
                location += f", column {column}"
        super().__init__(f"{message}{location}")


class XMLExtractionError(STParserError):
    """Error extracting ST code from TcPOU XML."""

    pass


class DeclarationParseError(STParserError):
    """Error parsing variable declarations."""

    pass


class ImplementationParseError(STParserError):
    """Error parsing ST implementation code."""

    pass
