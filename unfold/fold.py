"""
Contains the Fold class, which is used to fold one or several databases
into a data package.

"""

from typing import List, Union
from collections import defaultdict
import uuid
import csv
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
from datapackage import Package
from scipy import sparse as nsp
import sparse
from prettytable import PrettyTable
from wurst import extract_brightway2_databases
import bw2data
from .data_cleaning import get_outdated_flows, get_biosphere_code, DATA_DIR
from . import __version__

DIR_DATAPACKAGE_TEMP = DATA_DIR / "temp"


def get_list_unique_acts(scenarios: List[List[dict]]) -> list:
    """
    Get a list of unique activities from a list of databases
    :param scenarios: list of databases
    :return: list of unique activities
    """

    list_unique_acts = []
    for database in scenarios:
        for dataset in database:
            list_unique_acts.extend(
                [
                    (a["name"], None, a.get("categories"), None, a["unit"])
                    if a["type"] == "biosphere"
                    else (
                        a["name"],
                        a.get("product"),
                        None,
                        a.get("location"),
                        a["unit"],
                    )
                    for a in dataset["exchanges"]
                ]
            )
    return list(set(list_unique_acts))


def write_formatted_data(name, data, filepath):
    """
    Adapted from bw2io.export.csv
    :param name: name of the database
    :param data: data to write
    :param filepath: path to the file
    """

    sections = [
        "project parameters",
        "database",
        "database parameters",
        "activities",
        "activity parameters",
        "exchanges",
    ]

    result = []

    if "database" in sections:
        result.append(["Database", name])
        result.append([])

    if "activities" not in sections:
        return result
    for act in data:
        result.append(["Activity", act["name"]])
        result.append(["reference product", act["reference product"]])
        result.append(["unit", act["unit"]])
        result.append(["location", act["location"]])
        result.append(["comment", act.get("comment", "")])
        result.append(["source", act.get("source", "")])
        result.append([""])

        if "exchanges" in sections:
            result.append(["Exchanges"])
            if act.get("exchanges"):
                result.append(
                    [
                        "name",
                        "amount",
                        "unit",
                        "location",
                        "categories",
                        "type",
                        "product",
                    ]
                )
                for exc in act["exchanges"]:
                    result.append(
                        [
                            exc["name"],
                            exc["amount"],
                            exc["unit"],
                            exc.get("location"),
                            "::".join(list(exc.get("categories", [])))
                            if exc["type"] == "biosphere"
                            else None,
                            exc["type"],
                            exc.get("product"),
                        ]
                    )
        result.append([])

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for line in result:
            writer.writerow(line)

    return filepath


class Fold:
    """
    Folds a list of databases into a data package.
    """

    def __init__(self):

        self.bio_dict = get_biosphere_code()
        self.outdated_flows = get_outdated_flows()
        self.exc_codes = {}
        self.source = None
        self.databases_to_fold = []
        self.dependencies = set()
        self.dependency_mapping = {}
        self.datapackage_name = None
        self.datapackage_description = None

    @staticmethod
    def extract_source_database(db_name):
        """Extracts the source database."""
        return extract_brightway2_databases(db_name)

    def identify_databases_to_fold(
        self,
        source_database: str = None,
        source_database_system_model: str = None,
        source_database_version: Union[float, str] = None,
        databases_to_fold: List[str] = None,
        descriptions: List[str] = None,
    ):

        """
        Identify the source database
        :return: name of the source database
        """

        if not self.datapackage_name:
            self.datapackage_name = str(input("Give a name for this datapackage: "))

        if not self.datapackage_description:
            self.datapackage_description = str(
                input("Give a short description for this datapackage: ")
            )

        available_databases = list(bw2data.databases.keys())

        if not source_database:
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

            source_database = int(input("Indicate the no. of the reference database: "))
            source_database = available_databases[source_database - 1]
        assert source_database in available_databases, "Source database not found"

        self.identify_dependencies(source_database)

        if not source_database_system_model:
            source_database_system_model = str(
                input(f"Indicate the system model of {source_database}: ")
            )
        if not source_database_version:
            source_database_version = str(
                input(f"Indicate the version of {source_database}: ")
            )

        if not databases_to_fold:
            databases_to_fold = str(
                input(
                    "Indicate the no. of the databases "
                    "you would like to fold, "
                    "separated by a comma: "
                )
            )
            databases_to_fold = [
                available_databases[int(db.strip()) - 1]
                for db in databases_to_fold.split(",")
            ]

        if not descriptions:
            databases_descriptions = {
                db: str(input(f"Give a short description for {db}: "))
                for db in databases_to_fold
            }
        else:
            databases_descriptions = dict(zip(databases_to_fold, descriptions))

        assert all(
            database in available_databases for database in databases_to_fold
        ), "Database not found"

        self.identify_dependencies(databases_to_fold)

        self.source = {
            "name": source_database,
            "database": self.extract_source_database(source_database),
            "system model": source_database_system_model,
            "version": source_database_version,
        }

        self.databases_to_fold = [
            {
                "name": database,
                "database": self.extract_source_database(database),
                "description": databases_descriptions[database],
            }
            for database in databases_to_fold
        ]

        for dependency in self.dependencies:
            if dependency not in [source_database] + databases_to_fold:
                if dependency in bw2data.databases:
                    print("")
                    print(
                        f"The database {dependency} is an external dependency. "
                        f"It will be extracted automatically."
                    )
                    self.build_mapping_for_dependencies(
                        self.extract_source_database(dependency)
                    )
                else:
                    raise ValueError(
                        f"Database {dependency} also needed but not found."
                    )

        self.dependencies = {
            dependency
            for dependency in self.dependencies
            if dependency not in [source_database] + databases_to_fold
        }

    def build_mapping_for_dependencies(self, database):
        """Builds a mapping for dependencies."""
        self.dependency_mapping.update(
            {
                (
                    dataset["name"],
                    dataset.get("reference product"),
                    dataset.get("location"),
                    dataset.get("unit"),
                    dataset.get("categories"),
                ): (dataset["database"], dataset["code"])
                for dataset in database
            }
        )

    def identify_dependencies(self, database_names):
        """
        Identify the dependencies between the databases
        :return: dictionary of dependencies
        """

        if isinstance(database_names, str):
            database_names = [database_names]

        for name in database_names:
            self.dependencies.update(bw2data.Database(name).find_graph_dependents())

    def fold(
        self,
        package_name: str = None,
        package_description: str = None,
        source: str = None,
        system_model: str = None,
        version: Union[float, str] = None,
        databases_to_fold: List[str] = None,
        descriptions: List[str] = None,
    ):
        self.datapackage_name = package_name
        self.datapackage_description = package_description

        self.identify_databases_to_fold(
            source_database=source,
            source_database_system_model=system_model,
            source_database_version=version,
            databases_to_fold=databases_to_fold,
            descriptions=descriptions,
        )
        dataframe, extra_inventories = self.generate_scenario_factor_file(
            origin_db=self.source, scenarios=self.databases_to_fold
        )
        extra_inventories = self.zero_out_exchanges(extra_inventories)
        self.build_datapackage(dataframe, extra_inventories)

    def zero_out_exchanges(self, extra_inventories: List[dict]) -> List[dict]:
        """
        Zero out exchanges (except production exchanges) that are not in the source database.

        :param extra_inventories: list of inventories that are not in the source database
        :return: list of inventories with zeroed out exchanges
        """
        for dataset in extra_inventories:
            for exchange in dataset["exchanges"]:
                if exchange["type"] != "production":
                    exchange["amount"] = 0

        return extra_inventories

    def get_exchange(
        self, ind: int, acts_ind: dict, amount: float = 1.0, production: bool = False
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
        name, ref, cat, loc, unit = acts_ind[ind]
        if cat:
            try:
                return {
                    "name": name,
                    "unit": unit,
                    "categories": cat,
                    "type": "biosphere",
                    "amount": amount,
                    "input": (
                        "biosphere3",
                        self.bio_dict[
                            name,
                            cat[0],
                            cat[1] if len(cat) > 1 else "unspecified",
                            unit,
                        ],
                    ),
                }
            except KeyError:
                if name in self.outdated_flows:
                    return {
                        "name": name,
                        "unit": unit,
                        "categories": cat,
                        "type": "biosphere",
                        "amount": amount,
                        "input": (
                            "biosphere3",
                            self.bio_dict[
                                self.outdated_flows[name],
                                cat[0],
                                cat[1] if len(cat) > 1 else "unspecified",
                                unit,
                            ],
                        ),
                    }
        return {
            "name": name,
            "product": ref,
            "unit": unit,
            "location": loc,
            "type": "production" if production else "technosphere",
            "amount": amount,
            "input": (
                self.datapackage_name,
                self.fetch_exchange_code(name, ref, loc, unit),
            ),
        }

    @lru_cache()
    def fetch_exchange_code(self, name: str, ref: str, loc: str, unit: str) -> str:
        """
        Fetch the code of an activity or create one.
        name: name of the activity
        ref: reference product
        loc: location
        unit: unit

        :return: code of the activity
        """

        if (name, ref, loc, unit) not in self.exc_codes:
            code = str(uuid.uuid4().hex)
            self.exc_codes[(name, ref, loc, unit)] = code
        else:
            code = self.exc_codes[(name, ref, loc, unit)]

        return code

    def get_act_dict_structure(self, ind: int, acts_ind: dict) -> dict:
        """
        Get the structure of the activity/dataset dictionary.
        :param ind: index of the activity
        :param acts_ind: dictionary of activities
        :return: dictionary of the activity

        """
        name, ref, _, loc, unit = acts_ind[ind]
        code = self.fetch_exchange_code(name, ref, loc, unit)

        return {
            "name": name,
            "reference product": ref,
            "unit": unit,
            "location": loc,
            "database": self.datapackage_name,
            "code": code,
            "exchanges": [],
        }

    def check_for_outdated_flows(self, database):
        """
        Check for outdated flows in the database.
        The list of outdated flows is stored in the attribute outdated_flows,
        which is a dictionary with the old flow name as key and the new flow name as value.
        See outdated_flows.yaml for the list of outdated flows.

        :param database: the database to check
        :return: a database with the outdated flows replaced
        """

        for dataset in database:
            for exchange in dataset["exchanges"]:
                if exchange["name"] in self.outdated_flows:
                    exchange["name"] = self.outdated_flows[exchange["name"]]
        return database

    def build_datapackage(self, dataframe, inventories):
        """
        Create and export a scenario datapackage.
        """

        # check that directory exists, otherwise create it
        Path(DIR_DATAPACKAGE_TEMP).mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(DIR_DATAPACKAGE_TEMP / "scenario_data.csv", index=False)
        write_formatted_data(
            name=self.datapackage_name,
            data=inventories,
            filepath=DIR_DATAPACKAGE_TEMP / "inventories.csv",
        )
        package = Package(base_path=str(DIR_DATAPACKAGE_TEMP))
        package.infer("**/*.csv")
        package.descriptor["name"] = self.datapackage_name
        package.descriptor["title"] = self.datapackage_name.capitalize()
        package.descriptor["description"] = self.datapackage_description
        package.descriptor["unfold version"] = str(__version__)
        package.descriptor["dependencies"] = [
            {
                "name": self.source["name"],
                "system model": self.source["system model"],
                "version": self.source["version"],
                "type": "source",
            }
        ]
        for dependency in self.dependencies:
            package.descriptor["dependencies"].append(
                {
                    "name": dependency,
                    "type": "dependency",
                }
            )
        package.descriptor["scenarios"] = [
            {"name": s["name"], "description": s["description"]}
            for s in self.databases_to_fold
        ]
        package.descriptor["licenses"] = [
            {
                "id": "CC0-1.0",
                "title": "CC0 1.0",
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        ]
        package.commit()
        package.save(f"{self.datapackage_name}.zip")

        print(f"Data package saved at {f'{self.datapackage_name}.zip'}")

    def generate_scenario_factor_file(self, origin_db, scenarios):
        """
        Generate a scenario factor file from a list of databases
        :param origin_db: the original database
        :param scenarios: a list of databases
        """

        print("Building scenario factor file...")

        # create the dataframe
        dataframe, new_db, list_unique_acts = self.generate_scenario_difference_file(
            origin_db=origin_db, scenarios=scenarios
        )

        original = dataframe["original"]
        original = original.replace(0, 1)
        dataframe.loc[:, "original":] = dataframe.loc[:, "original":].div(
            original, axis=0
        )

        # remove the column `original`
        dataframe = dataframe.drop(columns=["original"])
        # fetch a list of activities not present in original_db
        list_original_acts = get_list_unique_acts([origin_db["database"]])
        new_acts_list = list(set(list_unique_acts) - set(list_original_acts))

        # fetch the additional activities from new_db
        extra_acts = [
            dataset
            for dataset in new_db
            if (
                dataset["name"],
                dataset.get("reference product"),
                None,
                dataset.get("location"),
                dataset["unit"],
            )
            in new_acts_list
        ]

        return dataframe, extra_acts

    def generate_scenario_difference_file(
        self, origin_db: dict, scenarios: List[dict]
    ) -> tuple[pd.DataFrame, list[dict], list[tuple]]:
        """
        Generate a scenario difference file for a given list of databases
        :param origin_db: the original database
        :param scenarios: list of databases
        """

        self.exc_codes.update(
            {
                (
                    dataset["name"],
                    dataset["reference product"],
                    dataset["location"],
                    dataset["unit"],
                ): dataset["code"]
                for dataset in origin_db["database"]
            }
        )

        list_acts = get_list_unique_acts(
            [origin_db["database"]] + [s["database"] for s in scenarios]
        )
        acts_ind = dict(enumerate(list_acts))
        acts_ind_rev = {value: key for key, value in acts_ind.items()}
        list_scenarios = ["original"] + [s["name"] for s in scenarios]
        list_of_databases = [origin_db["database"]] + [a["database"] for a in scenarios]

        matrices = {
            a: nsp.lil_matrix((len(list_acts), len(list_acts)))
            for a, _ in enumerate(list_scenarios)
        }

        # store the metadata in a dictionary
        dict_meta = {
            (
                dataset["name"],
                dataset["reference product"],
                None,
                dataset["location"],
                dataset["unit"],
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
            for database in list_of_databases
            for dataset in database
        }

        for _db_index, database in enumerate(list_of_databases):
            for dataset in database:
                for exc in dataset["exchanges"]:
                    if exc["type"] == "biosphere":
                        exc_id = (
                            exc["name"],
                            None,
                            exc.get("categories"),
                            None,
                            exc["unit"],
                        )
                    else:
                        exc_id = (
                            exc["name"],
                            exc["product"],
                            None,
                            exc["location"],
                            exc["unit"],
                        )
                    dataset_id = (
                        dataset["name"],
                        dataset.get("reference product"),
                        dataset.get("categories"),
                        dataset.get("location"),
                        dataset["unit"],
                    )
                    matrices[_db_index][
                        acts_ind_rev[exc_id], acts_ind_rev[dataset_id]
                    ] += exc["amount"]

        sparse_matrix = sparse.stack(
            [sparse.COO(x) for x in matrices.values()], axis=-1
        )
        indices = sparse.argwhere(sparse_matrix.sum(-1).T != 0)
        indices = list(map(tuple, indices))

        dataframe_rows = []

        inds_d = defaultdict(list)
        for index in indices:
            inds_d[index[0]].append(index[1])

        new_db = []

        for key, value in inds_d.items():
            act = self.get_act_dict_structure(
                key,
                acts_ind,
            )
            act.update(dict_meta[acts_ind[key]])

            act["exchanges"].extend(
                self.get_exchange(i, acts_ind, amount=sparse_matrix[i, key, 0])
                if i != key
                else self.get_exchange(key, acts_ind, production=True)
                for i in value
            )

            new_db.append(act)

        inds_std = sparse.argwhere(sparse_matrix.std(-1).T > 1e-12)

        for _db_index in inds_std:

            c_name, c_ref, c_cat, c_loc, c_unit = acts_ind[_db_index[0]]
            s_name, s_ref, s_cat, s_loc, s_unit = acts_ind[_db_index[1]]

            if s_cat:
                flow_type = "biosphere"
                database_name = "biosphere3"
                try:
                    exc_key_supplier = (
                        database_name,
                        self.bio_dict[
                            s_name,
                            s_cat[0],
                            s_cat[1] if len(s_cat) > 1 else "unspecified",
                            s_unit,
                        ],
                    )
                except KeyError:
                    exc_key_supplier = (
                        database_name,
                        self.bio_dict[
                            self.outdated_flows[s_name],
                            s_cat[0],
                            s_cat[1] if len(s_cat) > 1 else "unspecified",
                            s_unit,
                        ],
                    )

            elif _db_index[0] == _db_index[1]:
                flow_type = "production"
                database_name = self.datapackage_name
                exc_key_supplier = (
                    self.datapackage_name,
                    self.fetch_exchange_code(s_name, s_ref, s_loc, s_unit),
                )
            else:
                flow_type = "technosphere"
                database_name = self.datapackage_name
                exc_key_supplier = (
                    self.datapackage_name,
                    self.fetch_exchange_code(s_name, s_ref, s_loc, s_unit),
                )

            exc_key_consumer = (
                self.datapackage_name,
                self.fetch_exchange_code(c_name, c_ref, c_loc, c_unit),
            )

            row = [
                s_name,
                s_ref,
                s_loc,
                s_cat,
                database_name,
                exc_key_supplier,
                s_unit,
                c_name,
                c_ref,
                c_loc,
                c_cat,
                self.datapackage_name,
                exc_key_consumer,
                flow_type,
            ]

            if flow_type == "production":
                row.extend(np.ones_like(sparse_matrix[_db_index[1], _db_index[0], :]))
            else:
                row.extend(sparse_matrix[_db_index[1], _db_index[0], :])

            dataframe_rows.append(row)

        columns = [
            "from activity name",
            "from reference product",
            "from location",
            "from categories",
            "from database",
            "from key",
            "unit",
            "to activity name",
            "to reference product",
            "to location",
            "to categories",
            "to database",
            "to key",
            "flow type",
        ] + list_scenarios

        # return the dataframe and the new db
        return pd.DataFrame(dataframe_rows, columns=columns), new_db, list_acts
