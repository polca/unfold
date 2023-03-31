import pytest
from datapackage.exceptions import DataPackageException

from unfold import Unfold


def test_unfold():
    with pytest.raises(TypeError) as wrapped_error:
        u = Unfold()
    assert wrapped_error.type == TypeError

    with pytest.raises(DataPackageException) as wrapped_error:
        u = Unfold(".")
    assert wrapped_error.type == DataPackageException
