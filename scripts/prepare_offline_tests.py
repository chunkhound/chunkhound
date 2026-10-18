"""Install declared native test artifacts before CI removes network access."""

import duckdb

from chunkhound.watchman_runtime.loader import (
    is_packaged_watchman_runtime_supported,
    materialize_watchman_binary,
)


def main() -> None:
    with duckdb.connect() as connection:
        connection.execute("INSTALL vss")
        connection.execute("LOAD vss")
    if is_packaged_watchman_runtime_supported():
        materialize_watchman_binary()
    else:
        print(
            "Watchman runtime not supported on this platform; skipping materialization."
        )


if __name__ == "__main__":
    main()
