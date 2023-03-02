"""
Contains the Fold class, which is used to fold one
or several databases into a data package.

"""

import csv
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Union

import bw2data
import pandas as pd
import sparse
from datapackage import Package
from prettytable import PrettyTable
from scipy import sparse as nsp

from . import __version__
from .data_cleaning import (
    DATA_DIR,
    check_commonality_between_databases,
    check_mandatory_fields,
    get_biosphere_code,
    get_outdated_flows,
)

DIR_DATAPACKAGE_TEMP = DATA_DIR / "temp"


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
        self.rev_dependency_mapping = {}
        self.datapackage_name = None
        self.datapackage_description = None

    @staticmethod
    def extract_database(db_name):
        """Extracts the source database."""
        return [v for v in bw2data.Database(db_name).load().values()]

    def identify_databases_to_fold(
        self,
        source_database: str = None,
        source_database_system_model: str = None,
        source_database_version: Union[float, str] = None,
        databases_to_fold: List[str] = None,
        descriptions: List[str] = None,
    ):
        """
        The identify_databases_to_fold function identifies the source database and the databases to be folded into the source database and extracts them.

        :param source_database: Name of the source database to be used. If not specified, the user is prompted to choose from the available databases.
        :param source_database_system_model: System model of the source database to be used. If not specified, the user is prompted to input.
        :param source_database_version: Version of the source database to be used. If not specified, the user is prompted to input.
        :param databases_to_fold: List of databases to be folded into the source database. If not specified, the user is prompted to input.
        :param descriptions: Short descriptions of each database to be folded. If not specified, the user is prompted to input.
        :return: The source dictionary containing information about the source database, such as its name, database, system model, and version.
        :return: The databases_to_fold list containing dictionaries of information about each database to be folded, including its name, database, and description.

        Functionality:

        Checks whether the user has already specified a datapackage name and description. If not, prompts the user to input these details.
        Lists the available databases and prompts the user to input the number of the reference database if the source_database input is not specified.
        Identifies the dependencies of the source and folded databases.
        Prompts the user to input the system model and version of the source database if not specified.
        Prompts the user to input the list of databases to be folded and their descriptions if not specified.
        Extracts the source database and ensures that mandatory fields are included.
        Builds the mapping of dependencies for the source database.
        Extracts each database to be folded, ensures that mandatory fields are included, and builds the mapping of dependencies.
        Identifies whether any dependencies are external and, if so, extracts them.
        Returns a set of the dependencies excluding the source and databases to be folded.

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

        print("Extracting source database...")
        source_database_extracted = self.extract_database(source_database)
        self.source = {
            "name": source_database,
            "database": source_database_extracted,
            "system model": source_database_system_model,
            "version": source_database_version,
        }

        source_database_extracted = check_mandatory_fields(source_database_extracted)

        self.build_mapping_for_dependencies(source_database_extracted)
        print("Done!")

        print("Extracting databases to fold...")
        for database in databases_to_fold:
            extracted_database = self.extract_database(database)
            extracted_database = check_mandatory_fields(extracted_database)
            check_commonality_between_databases(
                source_database_extracted, extracted_database, database
            )

            self.databases_to_fold.append(
                {
                    "name": database,
                    "database": extracted_database,
                    "description": databases_descriptions[database],
                }
            )
            self.build_mapping_for_dependencies(extracted_database)

        print("Done!")

        for dependency in self.dependencies:
            if dependency not in [source_database] + databases_to_fold:
                if dependency in bw2data.databases:
                    print("")
                    print(
                        f"The database {dependency} is an external dependency. "
                        f"It will be extracted automatically."
                    )
                    self.build_mapping_for_dependencies(
                        self.extract_database(dependency)
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

        self.rev_dependency_mapping.update(
            {
                (dataset["database"], dataset["code"]): (
                    dataset["name"],
                    dataset.get("reference product"),
                    dataset.get("location"),
                    dataset.get("unit"),
                    dataset.get("categories"),
                )
                for dataset in database
            }
        )

    def get_list_unique_acts(self, scenarios: List[List[dict]]) -> list:
        """
        Returns a list of unique activities from a list of databases, where each database is represented by a list of
        datasets containing their respective exchanges.

        :param scenarios: A list of databases, where each database is a list of datasets, with each dataset containing the
                          exchanges.
        :type scenarios: list

        :return: A list of tuples representing the unique activities in the provided databases, where each tuple contains
                 the activity name, reference product, location, categories, unit and type.
        :rtype: list
        """

        list_unique_acts = []
        for database in scenarios:
            for dataset in database:
                list_unique_acts.extend(
                    [
                        (
                            self.rev_dependency_mapping.get(
                                exchange.get("input"), (None,)
                            )[0],
                            self.rev_dependency_mapping.get(
                                exchange.get("input"), (None, None)
                            )[1],
                            self.rev_dependency_mapping.get(
                                exchange.get("input"), (None,)
                            )[-1],
                            self.rev_dependency_mapping.get(
                                exchange.get("input"), (None, None, None)
                            )[2],
                            exchange["unit"],
                            exchange["type"],
                        )
                        for exchange in dataset["exchanges"]
                    ]
                )
        return list(set(list_unique_acts))

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
        """
        Folds one or more databases into a new package.

        :param package_name: Name for the new datapackage.
        :type package_name: str, optional
        :param package_description: Short description for the new datapackage.
        :type package_description: str, optional
        :param source: Name of the source database.
        :type source: str, optional
        :param system_model: System model of the source database.
        :type system_model: str, optional
        :param version: Version of the source database.
        :type version: float or str, optional
        :param databases_to_fold: List of names of the databases to fold.
        :type databases_to_fold: List[str], optional
        :param descriptions: Short description for each database to fold.
        :type descriptions: List[str], optional
        :raises AssertionError: When one or more databases to fold are not found.
        """
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
        Zero out exchanges (except production exchanges)
        that are not in the source database.

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
        name, ref, cat, loc, unit, flow_type = acts_ind[ind]
        _ = lambda x: x if x != 0 else 1.0
        return {
            "name": name,
            "product": ref,
            "unit": unit,
            "location": loc,
            "categories": cat,
            "type": flow_type,
            "amount": amount if flow_type != "production" else _(amount),
            "input": self.dependency_mapping[(name, ref, loc, unit, cat)]
            if flow_type == "biosphere"
            else (
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
        name, ref, _, loc, unit, _ = acts_ind[ind]
        code = self.fetch_exchange_code(name, ref, loc, unit)

        return {
            "name": name,
            "reference product": ref,
            "unit": unit,
            "location": loc,
            "database": self.datapackage_name,
            "code": code,
            "parameters": [],
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
        list_original_acts = self.get_list_unique_acts([origin_db["database"]])
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
                "production",
            )
            in new_acts_list
        ]

        return dataframe, extra_acts

    def generate_scenario_difference_file(
        self, origin_db: dict, scenarios: List[dict]
    ) -> tuple[pd.DataFrame, list[dict], list[tuple]]:
        """
        Generate a scenario difference file for a given list of databases.
        The function generate_scenario_difference_file calculates the scenario difference file for a given list of databases.
        This function takes in two parameters, origin_db, and scenarios. origin_db is a dictionary representing the
        original database, and scenarios is a list of databases.

        The function first creates a dictionary self.exc_codes to store the codes of exchanges using their attributes
        like name, reference product, location, and unit. It then fetches the unique activities by calling
        the get_list_unique_acts function using the list of the original database and scenarios as the argument.

        It then creates a dictionary acts_ind to map indices to each activity, and another dictionary acts_ind_rev
        that maps activities to their corresponding indices. It also creates a list of scenarios and a list of databases.

        The function then initializes matrices using the lil_matrix function from the scipy.sparse module to store
        the matrices for each scenario. It then extracts metadata from the databases, and stores them in a dictionary dict_meta.

        Next, for each dataset in each database, the function retrieves the dataset_id and exc_id by using their
        attributes like name, reference product, location, unit, and type. It then updates the corresponding matrix entry with the exchange amount for the current dataset and exchange.

        The function then stacks the matrices into a sparse matrix and retrieves the indices of nonzero elements.
        It then creates a new database by combining the metadata with exchanges corresponding to the indices retrieved.

        Finally, the function creates a dataframe containing the differences between scenarios and returns it along
        with the new database and a list of activities.

        :param origin_db: the original database
        :param scenarios: list of databases
        :return: a tuple containing a dataframe, a list of dictionaries, and a list of tuples
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

        # fetch a list of unique datasets
        list_acts = self.get_list_unique_acts(
            [origin_db["database"]] + [a["database"] for a in scenarios]
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
                dataset_id = (
                    dataset["name"],
                    dataset.get("reference product"),
                    None,
                    dataset.get("location"),
                    dataset["unit"],
                    "production",
                )

                for exc in dataset["exchanges"]:
                    exc_id = (
                        self.rev_dependency_mapping.get(exc.get("input"), (None,))[0],
                        self.rev_dependency_mapping.get(exc.get("input"), (None, None))[
                            1
                        ],
                        self.rev_dependency_mapping.get(exc.get("input"), (None,))[-1],
                        self.rev_dependency_mapping.get(
                            exc.get("input"), (None, None, None)
                        )[2],
                        exc["unit"],
                        exc["type"],
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
            meta_id = tuple(list(acts_ind[key])[:-1])
            act.update(dict_meta[meta_id])

            act["exchanges"].extend(
                self.get_exchange(i, acts_ind, amount=sparse_matrix[i, key, 0])
                for i in value
            )

            new_db.append(act)

        inds_std = sparse.argwhere(
            (sparse_matrix[..., 1:] == sparse_matrix[..., 0, None]).all(axis=-1).T
            == False
        )

        for _db_index in inds_std:
            c_name, c_ref, c_cat, c_loc, c_unit, c_type = acts_ind[_db_index[0]]
            s_name, s_ref, s_cat, s_loc, s_unit, s_type = acts_ind[_db_index[1]]

            if s_type == "biosphere":
                database_name = "biosphere3"

                exc_key_supplier = self.dependency_mapping[
                    (s_name, s_ref, s_loc, s_unit, s_cat)
                ]

            else:
                database_name = self.datapackage_name
                exc_key_supplier = (
                    database_name,
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
                c_unit,
                s_type,
            ]

            row.extend(sparse_matrix[_db_index[1], _db_index[0], :])

            dataframe_rows.append(row)

        columns = [
            "from activity name",
            "from reference product",
            "from location",
            "from categories",
            "from database",
            "from key",
            "from unit",
            "to activity name",
            "to reference product",
            "to location",
            "to categories",
            "to database",
            "to key",
            "to unit",
            "flow type",
        ] + list_scenarios

        df = pd.DataFrame(dataframe_rows, columns=columns)
        df["to categories"] = None
        df.loc[df["flow type"] == "biosphere", "from reference product"] = None
        df.loc[df["flow type"] == "biosphere", "from location"] = None
        df.loc[df["flow type"] == "technosphere", "from categories"] = None
        df.loc[df["flow type"] == "production", "from categories"] = None
        # remove production exchanges
        df = df[df["flow type"] != "production"]

        # return the dataframe and the new db
        return df, new_db, list_acts
