import bw2calc
import bw2data
import bw2io

from unfold import Unfold

bw2data.projects.set_current("ei39")
fp = "/Users/romain/Documents/datapackage_IMAGE_SSP2_Ammonia.zip"
u = Unfold(fp)
u.unfold(
    scenarios=[0, 1],
    superstructure=True,
    dependencies={
        "biosphere3": "biosphere3",
        "ecoinvent": "ecoinvent 3.9.1 cutoff",
    },
)
