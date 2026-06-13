import pytest

from cage.artifacts.run_storage import validate_run_id


def test_validate_run_id_allows_48_character_ids() -> None:
    run_id = "r" + "a" * 47

    assert validate_run_id(run_id) == run_id


def test_validate_run_id_rejects_ids_longer_than_48_characters() -> None:
    run_id = "r" + "a" * 48

    with pytest.raises(ValueError, match="run_id too long"):
        validate_run_id(run_id)
