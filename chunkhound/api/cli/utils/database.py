"""Database utility functions for CLI commands."""

from chunkhound.core.config.config import Config


def verify_database_exists(config: Config) -> None:
    """Verify database exists, raising if not found.

    Args:
        config: Configuration with database settings

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If database path not configured
    """
    if not config.database.path:
        raise ValueError("Database path not configured")

    # Check existence using transformed path (includes provider-specific suffix)
    actual_db_path = config.database.get_db_path()
    if not actual_db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {actual_db_path}. "
            f"Run 'chunkhound index <directory>' to create the database first."
        )
