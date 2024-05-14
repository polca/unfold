from unfold import Unfold
import bw2io, bw2data, bw2calc
bw2data.projects.set_current("ei39")
fp = "/Users/romain/Documents/datapackage_IMAGE_SSP2_Ammonia.zip"
u = Unfold(fp)
u.unfold(
    scenarios=[0, 1],
    superstructure=True,
    dependencies={
        "biosphere3": "biosphere3",
        "ecoinvent": "ecoinvent 3.9.1 cutoff",
    }
)