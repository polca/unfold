import bw2calc
import bw2data
import bw2io
import numpy as np
import pytest
import yaml

from unfold import Fold, Unfold

bw2data.projects.set_current("test")
bw2io.bw2setup()


def test_db_reproduction():
    fp = "./tests/fixture/reference_database.yaml"
    with open(fp, "r") as stream:
        db = yaml.load(stream, Loader=yaml.FullLoader)

    fp = "./tests/fixture/db A.yaml"

    with open(fp, "r") as stream:
        db_a = yaml.load(stream, Loader=yaml.FullLoader)

    fp = "./tests/fixture/db B.yaml"

    with open(fp, "r") as stream:
        db_b = yaml.load(stream, Loader=yaml.FullLoader)

    bw2data.Database("reference_database").write(db)
    bw2data.Database("db A").write(db_a)
    bw2data.Database("db B").write(db_b)

    lca = bw2calc.LCA({bw2data.get_activity(("db A", "activity A")): 1})
    lca.lci()
    original_supply_A = lca.supply_array

    lca = bw2calc.LCA({bw2data.get_activity(("db B", "activity A")): 1})
    lca.lci()
    original_supply_B = lca.supply_array

    f = Fold()

    f.fold(
        package_name="test",
        package_description="description of test",
        source="reference_database",
        system_model="cutoff",
        version="2.0",
        databases_to_fold=["db A", "db B"],
        descriptions=["this is db A", "this is db B"],
    )

    Unfold("test.zip").unfold(dependencies={"reference_database": "reference_database"})

    lca = bw2calc.LCA({bw2data.get_activity(("db A", "activity A")): 1})
    lca.lci()
    new_supply_A = lca.supply_array
    assert np.allclose(original_supply_A, new_supply_A) == True

    lca = bw2calc.LCA({bw2data.get_activity(("db B", "activity A")): 1})
    lca.lci()
    new_supply_B = lca.supply_array
    assert np.allclose(original_supply_B, new_supply_B) == True
