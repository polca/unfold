from unfold import Fold
import wurst
import bw2data
bw2data.projects.set_current("democarmaup")

f = Fold()
f.fold(
    package_name="a package name",
    package_description="some description",
    source="cutoff38",
    system_model="cut off",
    version="3.8",
    databases_to_fold=["ecoinvent_image_SSP2-Base_2025", "ecoinvent_image_SSP2-Base_2050"],
    descriptions=["some db", "some other db"],
)
