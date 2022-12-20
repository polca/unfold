from bw2data import databases
from bw2io.importers.base_lci import LCIImporter


class UnfoldExporter(LCIImporter):
    """
    Inherits from `LCIImporter` to
    allow existing databases to be
    written to disk.

    """

    def __init__(self, db_name, data):
        super().__init__(db_name)
        self.db_name = db_name
        self.data = data
        for act in self.data:
            act["database"] = self.db_name

    # we override `write_database`
    # to allow existing databases
    # to be overwritten
    def write_database(self):
        if self.db_name in databases:
            print(
                f"Database {self.db_name} " f"already exists: it will be overwritten."
            )
        super().write_database()
