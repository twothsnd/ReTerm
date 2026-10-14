"""Task integration is selected by the training application."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reterm_rl.task_adapter import create_task_dataset


def test_factory_receives_task_settings() -> None:
    factory = lambda settings: settings
    with patch("reterm_rl.task_adapter.import_module", return_value=SimpleNamespace(make=factory)):
        dataset = create_task_dataset({"factory": "tasks:make", "root": "/tasks"})
    assert dataset == {"root": "/tasks"}


@pytest.mark.parametrize("reference", ["", "tasks", ":make", "tasks:"])
def test_factory_reference_is_complete(reference: str) -> None:
    with pytest.raises(ValueError, match="module:factory"):
        create_task_dataset({"factory": reference})
