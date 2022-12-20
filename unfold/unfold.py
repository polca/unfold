"""
Contains the Unfold class, to extract datapackage files.

"""
import copy
import uuid
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import bw2data
import bw2io
import numpy as np
import pandas as pd
import sparse
from datapackage import Package
from prettytable import PrettyTable
from scipy import sparse as nsp
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
        self.dict_meta = None
        self.acts_indices = None
        self.reversed_acts_indices = None
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
            i.strategies = i.strategies[:4]
            i.apply_strategies()
            i.data = self.clean_imported_inventory(i.data)

        self.database.extend(i.data)
        self.database = change_db_name(self.database, self.package.descriptor["name"])
        self.build_mapping_for_dependencies(self.database)
        self.store_datasets_metadata()

    def generate_factors(self):
        """Generates the factors."""
        self.factors = (
            self.scenario_df.groupby("flow id").sum(numeric_only=True).to_dict("index")
        )

    def get_list_unique_exchanges(self, databases):

        # get all unique exchanges
        # for each dataset in database
        # for each database in databases

        return list(
            set(
                [
                    (
                        exchange["name"],
                        exchange.get("product"),
                        exchange.get("categories"),
                        exchange.get("location"),
                        exchange.get("unit"),
                        exchange.get("type"),
                    )
                    for database in databases
                    for dataset in database
                    for exchange in dataset["exchanges"]
                ]
            )
        )

    def store_datasets_metadata(self):

        # store the metadata in a dictionary
        self.dict_meta = {
            (
                dataset["name"],
                dataset["reference product"],
                None,
                dataset["location"],
                dataset["unit"],
                "production",
            ): {
                key: values
                for key, values in dataset.items()
                if key
                not in [
                    "exchanges",
                    "code",
                    "name",
                    "reference product",
                    "location",
                    "unit",
                    "database",
                ]
            }
            for dataset in self.database
        }

    def generate_activities_indices(self):
        """Generates the activities indices."""
        list_unique_acts = self.get_list_unique_exchanges(databases=[self.database])

        # add additional exchanges
        for act in list_unique_acts:
            if act[-1] == "production":
                new_id = list(act)
                new_id[-1] = "technosphere"
                new_id = tuple(new_id)
                if new_id not in list_unique_acts:
                    list_unique_acts.append(new_id)

        self.acts_indices = dict(enumerate(list_unique_acts))
        self.reversed_acts_indices = {act: i for i, act in enumerate(list_unique_acts)}

    def fetch_exchange_code(self, name, ref, loc):

        if (name, ref, loc, None) in self.dependency_mapping:
            return self.dependency_mapping[(name, ref, loc, None)][1]
        else:
            return str(uuid.uuid4().hex)

    def get_exchange(
        self,
        ind: int,
        amount: float = 1.0,
        scenario_name: str = None,
    ):
        """
        Return an exchange in teh form of a dictionary.
        If it has a value for "categories", we inder it is a biosphere flow.
        If not, it is either a technosphere or production flow.

        :param ind: index of the exchange
        :param acts_ind: dictionary of activities
        :param amount: amount of the exchange
        :param production: boolean indicating if it is a production flow
        :return: dictionary of the exchange
        """
        name, ref, cat, loc, unit, flow_type = self.acts_indices[ind]
        _ = lambda x: x if x != 0 else 1.0
        return {
            "name": name,
            "product": ref,
            "unit": unit,
            "location": loc,
            "categories": cat,
            "type": flow_type,
            "amount": amount if flow_type != "production" else _(amount),
            "input": self.dependency_mapping[(name, ref, loc, cat)]
            if flow_type == "biosphere"
            else (
                scenario_name,
                self.fetch_exchange_code(name, ref, loc),
            ),
        }

    def populate_sparse_matrix(self):

        self.generate_activities_indices()

        m = nsp.lil_matrix((len(self.acts_indices), len(self.acts_indices)))

        for ds in self.database:
            for exc in ds["exchanges"]:
                s = (
                    exc["name"],
                    exc.get("product"),
                    exc.get("categories"),
                    exc.get("location"),
                    exc["unit"],
                    exc["type"],
                )

                c = (
                    ds["name"],
                    ds.get("reference product"),
                    ds.get("categories"),
                    ds.get("location"),
                    ds["unit"],
                    "production",
                )

                m[self.reversed_acts_indices[s], self.reversed_acts_indices[c]] += exc[
                    "amount"
                ]

        return m

    def write_scaling_factors_in_matrix(self, matrix, scenario_name):

        _ = lambda x: x if x != 0 else 1.0

        for flow_id, factor in self.factors.items():
            c_name, c_prod, c_loc, c_unit = list(flow_id)[:4]
            s_name, s_prod, s_loc, s_cat, s_unit, s_type = list(flow_id)[4:]

            consumer_idx = self.reversed_acts_indices[
                (
                    c_name,
                    c_prod,
                    None,
                    c_loc,
                    c_unit,
                    "production",
                )
            ]

            supplier_id = (
                s_name,
                s_prod,
                s_cat,
                s_loc,
                s_unit,
                s_type,
            )
            supplier_idx = self.reversed_acts_indices[supplier_id]

            matrix[supplier_idx, consumer_idx] = factor[scenario_name] * _(
                matrix[supplier_idx, consumer_idx]
            )

        return matrix

    def get_act_dict_structure(self, ind: int, scenario_name: str) -> dict:
        """
        Get the structure of the activity/dataset dictionary.
        :param ind: index of the activity
        :param acts_ind: dictionary of activities
        :return: dictionary of the activity

        """
        name, ref, _, loc, unit, _ = self.acts_indices[ind]
        code = self.fetch_exchange_code(name, ref, loc)

        return {
            "name": name,
            "reference product": ref,
            "unit": unit,
            "location": loc,
            "database": scenario_name,
            "code": code,
            "parameters": [],
            "exchanges": [],
        }

    def build_superstructure_database(self, matrix):

        print(f"Generating superstructure database...")

        # fetch non-zero indices of matrix 0
        non_zero_indices = sparse.argwhere(matrix[..., 0].T != 0)

        # add non-zero indices of other matrices
        for i in range(1, matrix.shape[-1]):
            non_zero_indices = np.concatenate(
                (non_zero_indices, sparse.argwhere(matrix[..., i].T != 0))
            )

        non_zero_indices = list(map(tuple, non_zero_indices))

        inds_d = defaultdict(list)
        for ind in non_zero_indices:
            if ind[1] not in inds_d[ind[0]]:
                inds_d[ind[0]].append(ind[1])

        new_db = []

        for k, v in inds_d.items():
            act = self.get_act_dict_structure(
                ind=k,
                scenario_name=self.package.descriptor["name"],
            )
            act.update(self.dict_meta[self.acts_indices[k]])

            act["exchanges"].extend(
                self.get_exchange(
                    ind=j,
                    amount=matrix[j, k, 0],
                    scenario_name=self.package.descriptor["name"],
                )
                for j in v
            )
            new_db.append(act)

        return new_db

    def build_single_databases(
        self, matrix, databases_to_build: List[dict]
    ) -> list[list[dict]]:

        databases_to_return = []

        for ix, i in enumerate(databases_to_build):

            print(f"Generating database for scenario {i['name']}...")

            non_zero_indices = sparse.argwhere(matrix[..., ix].T != 0)
            non_zero_indices = list(map(tuple, non_zero_indices))

            inds_d = defaultdict(list)
            for ind in non_zero_indices:
                inds_d[ind[0]].append(ind[1])

            new_db = []

            for k, v in inds_d.items():
                act = self.get_act_dict_structure(
                    ind=k,
                    scenario_name=i["name"],
                )
                act.update(self.dict_meta[self.acts_indices[k]])

                act["exchanges"].extend(
                    self.get_exchange(
                        ind=j, amount=matrix[j, k, ix], scenario_name=i["name"]
                    )
                    for j in v
                )
                new_db.append(act)
            databases_to_return.append(new_db)

        return databases_to_return

    def generate_superstructure_database(self) -> List[dict]:
        """
        Generate the superstructure database.
        """

        m = self.populate_sparse_matrix()

        matrix = sparse.stack(
            [sparse.COO(m)]
            + [
                sparse.COO(
                    self.write_scaling_factors_in_matrix(copy.deepcopy(m), s["name"])
                )
                for _, s in enumerate(self.scenarios)
            ],
            axis=-1,
        )

        return self.build_superstructure_database(matrix=matrix)

    def generate_single_databases(self) -> List[List[dict]]:

        m = self.populate_sparse_matrix()

        matrix = sparse.stack(
            [
                sparse.COO(
                    self.write_scaling_factors_in_matrix(copy.deepcopy(m), s["name"])
                )
                for _, s in enumerate(self.scenarios)
            ],
            axis=-1,
        )

        return self.build_single_databases(
            matrix=matrix, databases_to_build=self.scenarios
        )

    def format_dataframe(
        self, scenarios: List[int] = None, superstructure: bool = False
    ):
        """Formats the dataframe."""
        scenarios = scenarios or list(range(len(self.scenarios)))
        scenarios_to_keep = [self.scenarios[i]["name"] for i in scenarios]
        # remove unused scenarios from self.scenarios
        self.scenarios = [s for s in self.scenarios if s["name"] in scenarios_to_keep]

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
                self.scenario_df["to unit"],
                self.scenario_df["from activity name"],
                self.scenario_df["from reference product"],
                self.scenario_df["from location"],
                self.scenario_df["from categories"],
                self.scenario_df["from unit"],
                self.scenario_df["flow type"],
            )
        )

    def format_superstructure_dataframe(self):
        """Formats the superstructure dataframe."""

        matrix = self.populate_sparse_matrix()

        _ = lambda x: x if x != 0 else 1.0

        for flow_id, factor in self.factors.items():
            c_name, c_prod, c_loc, c_unit = list(flow_id)[:4]
            s_name, s_prod, s_loc, s_cat, s_unit, s_type = list(flow_id)[4:]

            consumer_idx = self.reversed_acts_indices[
                (
                    c_name,
                    c_prod,
                    None,
                    c_loc,
                    c_unit,
                    "production",
                )
            ]

            supplier_id = (
                s_name,
                s_prod,
                s_cat,
                s_loc,
                s_unit,
                s_type,
            )
            supplier_idx = self.reversed_acts_indices[supplier_id]

            for scenario, val in factor.items():
                factor[scenario] = val * _(matrix[consumer_idx, supplier_idx])

        self.scenario_df = pd.DataFrame.from_dict(self.factors).T.reset_index()
        self.scenario_df.columns = [
            "to activity name",
            "to reference product",
            "to location",
            "to unit",
            "from activity name",
            "from reference product",
            "from location",
            "from categories",
            "from unit",
            "flow type",
        ] + [s["name"] for s in self.scenarios]

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
            + [s["name"] for s in self.scenarios]
        ]

    def unfold(
        self,
        scenarios: List[int] = None,
        dependencies: dict = None,
        superstructure: bool = False,
    ):
        """Extracts specific scenarios."""

        self.check_dependencies(dependencies)
        self.extract_source_database()
        self.extract_additional_inventories()

        self.format_dataframe(scenarios=scenarios, superstructure=superstructure)
        self.generate_factors()

        if not superstructure:
            self.databases_to_export = {
                k: v
                for k, v in zip(
                    [s["name"] for s in self.scenarios],
                    self.generate_single_databases(),
                )
            }
        else:
            print("Writing scenario difference file...")
            self.format_superstructure_dataframe()
            self.database = self.generate_superstructure_database()

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
                change_db_name(data=database, name=scenario)
                # check_exchanges_input(database, self.dependency_mapping)
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
