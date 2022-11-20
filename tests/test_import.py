import pytest
from datapackage.exceptions import DataPackageException

from unfold import Fold, Unfold


def test_fold():
    f = Fold()
    assert isinstance(f.bio_dict, dict)
    assert isinstance(f.outdated_flows, dict)


def test_unfold():
    with pytest.raises(TypeError) as wrapped_error:
        u = Unfold()
    assert wrapped_error.type == TypeError

    with pytest.raises(DataPackageException) as wrapped_error:
        u = Unfold(".")
    assert wrapped_error.type == DataPackageException
