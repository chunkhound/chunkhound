"""Spawn-safe PDF parser worker fixtures."""

import time
from multiprocessing.connection import Connection


def blocking_pdf_worker(content: bytes, connection: Connection) -> None:
    """Block until the parent enforces its PDF parse timeout."""
    del content, connection
    time.sleep(60)


def malformed_pdf_worker(content: bytes, connection: Connection) -> None:
    """Send a syntactically valid but contract-invalid payload."""
    del content
    connection.send_bytes(b'["ok",[["symbol",1,1]],false]')
    connection.close()


def invalid_chunk_type_worker(content: bytes, connection: Connection) -> None:
    """Send a structurally valid chunk with an invalid enum value."""
    del content
    connection.send_bytes(
        b'["ok",[["symbol",1,1,"code","not-a-real-type",null]],false]'
    )
    connection.close()
