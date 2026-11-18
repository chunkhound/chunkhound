//! Lets `cargo test` load the DuckDB shared library.
//!
//! Why this exists: since Cargo 1.100 the new build-dir layout is always on,
//! and the runtime library path Cargo sets for `cargo test` no longer includes
//! the `<profile>/deps` directory that libduckdb-sys copies the downloaded
//! library into, so the unit-test binary failed to start on every OS.
//! libduckdb-sys's own rpath reaches only its own targets, and an rpath
//! relative to the test binary would depend on Cargo's build-dir layout,
//! which is not a stable interface. Instead, this script puts the library in
//! its own `OUT_DIR` (a hard link, or a copy where the filesystem refuses
//! links) and declares that directory with `rustc-link-search`. Cargo adds
//! such a directory to `PATH`, `LD_LIBRARY_PATH` or `DYLD_FALLBACK_LIBRARY_PATH`
//! for every binary it runs, so the test binary carries no build-machine path.
//! Cargo documents this as discouraged ("Depending on this behavior is
//! discouraged since this makes it difficult to use the resulting binary"):
//! acceptable here, because only `cargo test` runs this binary. A test binary
//! run directly gets no library path, and a `build.build-dir` moved outside
//! the target directory puts `OUT_DIR` outside the paths Cargo forwards. The
//! shipped extension does not use this; see the duckdb note in Cargo.toml.
//!
//! When it acts (the variables are read the way libduckdb-sys reads them):
//! the download mode (`DUCKDB_DOWNLOAD_LIB`, without
//! `DUCKDB_LIB_DIR`, linked dynamically) always publishes the download
//! directory as `DEP_DUCKDB_INCLUDE` with the library in it, so a missing
//! directory or library there fails the build. A static build
//! (`DUCKDB_STATIC`, including the air-gapped `DUCKDB_LIB_DIR` path) loads
//! nothing at run time and is left alone. Otherwise the library is staged if
//! `DEP_DUCKDB_INCLUDE` holds it; a system, pkg-config or vcpkg DuckDB, whose
//! include directory holds no library, is left to the system loader.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in ["DUCKDB_DOWNLOAD_LIB", "DUCKDB_LIB_DIR", "DUCKDB_STATIC"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let is_static = env::var("DUCKDB_STATIC").is_ok_and(|v| v != "0");
    let download = env::var("DUCKDB_DOWNLOAD_LIB")
        .is_ok_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true"));
    let download_active = download && env::var_os("DUCKDB_LIB_DIR").is_none() && !is_static;

    let lib_name = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "duckdb.dll",
        Ok("macos") => "libduckdb.dylib",
        _ => "libduckdb.so",
    };
    let source = env::var_os("DEP_DUCKDB_INCLUDE").map(|dir| Path::new(&dir).join(lib_name));

    match source {
        Some(source) if !is_static && source.is_file() => stage(&source, lib_name),
        Some(source) if download_active => panic!(
            "DUCKDB_DOWNLOAD_LIB is set but {} does not exist; libduckdb-sys did not leave the \
             DuckDB library where it published it, so the test binary could not load it",
            source.display()
        ),
        None if download_active => panic!(
            "DUCKDB_DOWNLOAD_LIB is set but libduckdb-sys published no DEP_DUCKDB_INCLUDE; \
             the test binary could not find the DuckDB library"
        ),
        _ => {}
    }
}

fn stage(source: &Path, lib_name: &str) {
    let out_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR for build scripts"));
    let dest = out_dir.join(lib_name);
    match fs::remove_file(&dest) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => panic!("could not remove stale {}: {err}", dest.display()),
    }
    if fs::hard_link(source, &dest).is_err() {
        fs::copy(source, &dest).unwrap_or_else(|err| {
            panic!(
                "could not copy {} to {}: {err}",
                source.display(),
                dest.display()
            )
        });
    }
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
