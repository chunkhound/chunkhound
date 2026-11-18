from __future__ import annotations

import chunkhound.services.realtime_path_filter as rpf
import chunkhound.utils.service_temp_names as stn

PROBE = stn.SERVICE_TEMP_PROBE_FILENAME
HYDRATION = stn.SERVICE_TEMP_HYDRATION_MARKER_FILENAME


def test_user_chunkhound_prefixed_files_are_admitted() -> None:
    assert stn.is_service_temp_name(".chunkhound-notes.py") is False
    assert stn.is_service_temp_name(".chunkhound-backup.txt") is False
    assert stn.is_service_temp_name(".chunkhound-probe.tmp.bak") is False


def test_exact_service_temp_filenames_are_rejected() -> None:
    assert stn.is_service_temp_name(PROBE) is True
    assert stn.is_service_temp_name(HYDRATION) is True


def test_service_temp_set_is_exactly_the_two_known_files() -> None:
    assert stn.SERVICE_TEMP_FILE_NAMES == frozenset({PROBE, HYDRATION})


def test_realtime_filter_uses_shared_service_temp_predicate() -> None:
    assert rpf.is_service_temp_name is stn.is_service_temp_name
