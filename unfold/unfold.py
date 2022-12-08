"""
Contains the Unfold class, to extract datapackage files.

"""

from ast import literal_eval
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import bw2data
import bw2io
import numpy as np
import pandas as pd
import pyprind
from datapackage import Package
from prettytable import PrettyTable
from wurst import extract_brightway2_databases
from wurst.linking import (
    change_db_name,
    check_duplicate_codes,
    check_internal_linking,
    link_internal,
)

from .data_cleaning import (
    add_biosphere_links,
    add_product_field_to_exchanges,
    check_exchanges_input,
    check_for_duplicates,
    correct_fields_format,
    remove_categories_for_technosphere_flows,
    remove_missing_fields,
)
from .export import UnfoldExporter
from .fold import get_outdated_flows
from .utils import HiddenPrints


def _c(value):
    """Converts zero to one."""
    if value == 0:
        return 1
    return value


def del_all(mapping, to_remove):
    """Remove list of elements from mapping."""
    for key in to_remove:
        del mapping[key]

    return mapping


class Unfold:
    """Extracts datapackage files."""

    def __init__(self, path: Union[str, Path]):
        self.path = path
        self.package = Package(self.path)
        self.dependencies = self.package.descriptor["dependencies"]
        self.scenarios = self.package.descriptor["scenarios"]
        self.scenario_df = None
        self.show_scenarios()
        self.database = []
        self.databases_to_export = {}
        self.dependency_mapping = {}
        self.factors = {}

    def show_scenarios(self):
        """Shows the scenarios."""
        print("The data package contains the following scenarios:")
        table = PrettyTable()
        table.field_names = ["No.", "Scenario", "Description"]
        for i, scenario in enumerate(self.package.descriptor["scenarios"]):
            table.add_row([i, scenario["name"], scenario.get("description")])
        print(table)
        print("")
        print("To unfold all scenarios, `unfold()`.")
        print("To unfold a specific scenario, `unfold(scenarios=[1,])`.")

    def check_dependencies(self, dependencies: dict):
        """Checks the dependencies."""
        # we need to check that the source database exists
        # and that the scenarios are compatible with the source database

        available_databases = list(bw2data.databases.keys())

        dependencies = dependencies or []

        if dependencies and all(
            dependency in available_databases for dependency in dependencies.values()
        ):
            for database in self.dependencies:
                database["source"] = dependencies[database["name"]]
        else:
            # ask the user to give names to the databases
            print("The following databases are necessary to unfold the scenarios:")

            table = PrettyTable()
            table.field_names = ["No.", "Dependency", "System model", "Version"]

            for _db, database in enumerate(self.dependencies):
                table.add_row(
                    [
                        _db + 1,
                        database["name"],
                        database.get("system model", ""),
                        database.get("version", ""),
                    ]
                )
            print(table)
            print("")

            print("The following databases are available in your project:")

            table = PrettyTable()
            table.field_names = [
                "No.",
                "Database",
            ]

            for _db, database in enumerate(available_databases):
                table.add_row([_db + 1, database])
            print(table)
            print("")

            for _db, database in enumerate(self.dependencies):
                db_number = input(
                    f"Indicate the database number for dependency {_db + 1}: "
                )
                name = available_databases[int(db_number) - 1]
                database["source"] = name

    def build_mapping_for_dependencies(self, database):
        """Builds a mapping for dependencies."""
        self.dependency_mapping.update(
            {
                (
                    a["name"],
                    a.get("reference product"),
                    a.get("location"),
                    a.get("categories"),
                ): (a["database"], a["code"])
                for a in database
            }
        )

    def extract_source_database(self):
        """Extracts the source database."""
        for dependency in self.dependencies:
            database = extract_brightway2_databases(dependency["source"])

            self.build_mapping_for_dependencies(database)
            if dependency.get("type") == "source":
                self.database.extend(database)

    def clean_imported_inventory(self, data):
        """Cleans the imported inventory."""
        print("Cleaning imported inventory...")
        data = remove_missing_fields(data)
        add_biosphere_links(data)
        check_for_duplicates(self.database, data)
        add_product_field_to_exchanges(data, self.database)
        remove_categories_for_technosphere_flows(data)
        return data

    def extract_additional_inventories(self):
        """Extracts additional inventories."""
        print("Extracting additional inventories...")
        with HiddenPrints():
            i = bw2io.CSVImporter(self.package.get_resource("inventories").source)
            i.apply_strategies()

        i.data = self.clean_imported_inventory(i.data)

        self.database.extend(i.data)
        self.database = change_db_name(self.database, self.package.descriptor["name"])
        self.build_mapping_for_dependencies(self.database)

    def adjust_exchanges(self):
        """Adjusts the exchanges."""

        self.factors = (
            self.scenario_df.groupby("flow id").sum(numeric_only=True).to_dict("index")
        )

        for scenario, database in self.databases_to_export.items():
            print(f"Creating database for scenario {scenario}...")
            for dataset in database:
                for exc in dataset["exchanges"]:
                    if exc["type"] != "production":

                        flow_id = (
                            dataset["name"],
                            dataset["reference product"],
                            dataset["location"],
                            exc["name"],
                            exc.get("product"),
                            exc.get("location"),
                            exc.get("categories"),
                            exc["unit"],
                            exc["type"],
                        )
                        if flow_id in self.factors:
                            if scenario in self.factors[flow_id]:
                                if self.factors[flow_id][scenario] is not None:
                                    exc["amount"] = _c(float(exc["amount"])) * float(
                                        self.factors[flow_id][scenario]
                                    )
                                    self.factors[flow_id][scenario] = None

                        if not exc.get("input"):
                            exc["input"] = self.dependency_mapping[
                                (
                                    exc["name"],
                                    exc.get("product"),
                                    exc.get("location"),
                                    exc.get("categories"),
                                )
                            ]

            # check if there are still exchange to add
            self.databases_to_export[scenario] = self.add_exchanges_to_database(
                database, scenario
            )

    def add_exchanges_to_database(self, database: List[dict], scenario: str):
        """
        Add an exchange to `database`.
        :param database: database to add an exchange to.
        :param flow_id: id of the exchanges
        :param factor: multiplication factor
        :return: database with exchange added
        """

        list_ds_to_modify = {(a[0], a[1], a[2]) for a in self.factors}

        datasets = [
            dataset
            for dataset in database
            if (dataset["name"], dataset["reference product"], dataset["location"])
            in list_ds_to_modify
        ]

        for dataset in pyprind.prog_percent(datasets):

            flows = {
                k: v
                for k, v in self.factors.items()
                if k[0] == dataset["name"]
                and k[1] == dataset["reference product"]
                and k[2] == dataset["location"]
                and v[scenario] is not None
            }

            excs = []

            for flow, factor in flows.items():

                exc = {
                    "amount": float(factor[scenario]),
                    "type": flow[-1],
                    "name": flow[3],
                    "product": flow[4],
                    "location": flow[5],
                    "categories": flow[6],
                    "unit": flow[7],
                    "input": self.dependency_mapping.get(
                        (flow[3], flow[4], flow[5], flow[6])
                    ),
                }

                excs.append(exc)

            dataset["exchanges"].extend(excs)

        return database

    def format_dataframe(
        self, scenarios: List[int] = None, superstructure: bool = False
    ):
        """Formats the dataframe."""
        scenarios = scenarios or list(range(len(self.scenarios)))
        scenarios_to_keep = [self.scenarios[i]["name"] for i in scenarios]

        if not superstructure:
            self.databases_to_export = {
                s: deepcopy(self.database) for s in scenarios_to_keep
            }

        scenarios_to_leave_out = list(
            set(s["name"] for s in self.scenarios) - set(scenarios_to_keep)
        )
        self.scenario_df = pd.DataFrame(
            self.package.get_resource("scenario_data").read(keyed=True)
        )

        self.scenario_df = self.scenario_df.loc[
            (self.scenario_df["flow type"] != "production")
        ]
        self.scenario_df = self.scenario_df.drop(scenarios_to_leave_out, axis=1)
        self.scenario_df = self.scenario_df.replace("None", None)
        self.scenario_df = self.scenario_df.replace({np.nan: None})
        self.scenario_df["from categories"] = self.scenario_df["from categories"].apply(
            lambda x: literal_eval(str(x))
        )
        self.scenario_df["to categories"] = self.scenario_df["to categories"].apply(
            lambda x: literal_eval(str(x))
        )
        self.scenario_df["from key"] = self.scenario_df["from key"].apply(
            lambda x: literal_eval(str(x))
        )
        self.scenario_df["to key"] = self.scenario_df["to key"].apply(
            lambda x: literal_eval(str(x))
        )
        self.scenario_df[scenarios_to_keep] = self.scenario_df[
            scenarios_to_keep
        ].astype(float)

        self.scenario_df["flow id"] = list(
            zip(
                self.scenario_df["to activity name"],
                self.scenario_df["to reference product"],
                self.scenario_df["to location"],
                self.scenario_df["from activity name"],
                self.scenario_df["from reference product"],
                self.scenario_df["from location"],
                self.scenario_df["from categories"],
                self.scenario_df["unit"],
                self.scenario_df["flow type"],
            )
        )

    def format_superstructure_dataframe(self, scenarios: List[int]):
        """Formats the superstructure dataframe."""

        self.format_dataframe(scenarios=scenarios, superstructure=True)
        scenarios = scenarios or list(range(len(self.scenarios)))
        scenarios = [self.scenarios[i]["name"] for i in scenarios]

        self.factors = (
            self.scenario_df.groupby("flow id").sum(numeric_only=True).to_dict("index")
        )
        existing_exchanges = []

        for dataset in self.database:
            for exc in dataset["exchanges"]:
                if exc["type"] != "production":
                    flow_id = (
                        dataset["name"],
                        dataset["reference product"],
                        dataset["location"],
                        exc["name"],
                        exc.get("product"),
                        exc.get("location"),
                        exc.get("categories"),
                        exc["unit"],
                        exc["type"],
                    )

                    if flow_id in self.factors:
                        existing_exchanges.append(flow_id)
                        for key, value in self.factors[flow_id].items():
                            if value != 0.0:
                                self.factors[flow_id][key] = float(value) * _c(
                                    float(exc["amount"])
                                )
                                self.factors[flow_id][key] = None

                    if not exc.get("input"):
                        exc["input"] = self.dependency_mapping[
                            (
                                exc["name"],
                                exc.get("product"),
                                exc.get("location"),
                                exc.get("categories"),
                            )
                        ]

        self.scenario_df = pd.DataFrame.from_dict(self.factors).T.reset_index()
        self.scenario_df.columns = [
            "to activity name",
            "to reference product",
            "to location",
            "from activity name",
            "from reference product",
            "from location",
            "from categories",
            "unit",
            "flow type",
        ] + scenarios

        self.scenario_df["to database"] = self.package.descriptor["name"]
        self.scenario_df["to categories"] = None
        self.scenario_df["to key"] = None
        self.scenario_df["from key"] = None
        self.scenario_df = self.scenario_df.replace({np.nan: None})

        self.scenario_df.loc[:, "from key"] = self.scenario_df.apply(
            lambda x: self.dependency_mapping.get(
                (
                    x["from activity name"],
                    x["from reference product"],
                    x["from location"],
                    x["from categories"],
                )
            ),
            axis=1,
        )

        self.scenario_df.loc[:, "to key"] = self.scenario_df.apply(
            lambda x: self.dependency_mapping.get(
                (
                    x["to activity name"],
                    x["to reference product"],
                    x["to location"],
                    x["to categories"],
                )
            ),
            axis=1,
        )

        self.scenario_df.loc[
            (self.scenario_df["flow type"] == "technosphere"), "from database"
        ] = self.package.descriptor["name"]
        self.scenario_df.loc[
            (self.scenario_df["flow type"] == "biosphere"), "from database"
        ] = "biosphere3"
        self.scenario_df = self.scenario_df[
            [
                "from activity name",
                "from reference product",
                "from location",
                "from categories",
                "from database",
                "from key",
                "to activity name",
                "to reference product",
                "to location",
                "to categories",
                "to database",
                "to key",
                "flow type",
            ]
            + scenarios
        ]

        print("Building superstructure database...")

        self.database = self.add_exchanges_to_database(self.database, scenarios[0])

    def unfold(
        self,
        scenarios: List[int] = None,
        dependencies: dict = None,
        superstructure: bool = False,
    ):
        """Extracts specific scenarios."""

        if not self.database:
            self.check_dependencies(dependencies)
            self.extract_source_database()
            self.extract_additional_inventories()

        if not superstructure:
            self.format_dataframe(scenarios)
            self.adjust_exchanges()
        else:
            print("Writing scenario difference file...")
            self.format_superstructure_dataframe(scenarios)

        self.write(superstructure=superstructure)

    def write(self, superstructure: bool = False):
        """
        Write the databases.
        If superstructure is True, write the scenario difference file,
        along with a database.

        :param superstructure: bool, default False

        """

        if not superstructure:
            for scenario, database in self.databases_to_export.items():

                change_db_name(database, scenario)
                check_exchanges_input(database, self.dependency_mapping)
                link_internal(database)
                check_internal_linking(database)
                check_duplicate_codes(database)
                correct_fields_format(database, scenario)
                print(f"Writing database for scenario {scenario}...")
                UnfoldExporter(scenario, database).write_database()

        else:
            try:
                self.scenario_df.to_excel(
                    f"{self.package.descriptor['name']}.xlsx", index=False
                )
            except ValueError:
                # from https://stackoverflow.com/questions/66356152/splitting-a-dataframe-into-multiple-sheets
                GROUP_LENGTH = 1000000  # set nr of rows to slice df
                with pd.ExcelWriter(
                    f"{self.package.descriptor['name']}.xlsx"
                ) as writer:
                    for i in range(0, len(self.scenario_df), GROUP_LENGTH):
                        self.scenario_df[i : i + GROUP_LENGTH].to_excel(
                            writer, sheet_name=f"Row {i}", index=False, header=True
                        )

            print(
                f"Scenario difference file exported to {self.package.descriptor['name']}.xlsx!"
            )
            print("")
            print("Writing superstructure database...")
            change_db_name(self.database, self.package.descriptor["name"])
            self.database = check_exchanges_input(
                self.database, self.dependency_mapping
            )
            link_internal(self.database)
            check_internal_linking(self.database)
            check_duplicate_codes(self.database)
            correct_fields_format(self.database, self.package.descriptor["name"])
            UnfoldExporter(
                self.package.descriptor["name"], self.database
            ).write_database()
