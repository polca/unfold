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

    def check_dependencies(self, dependencies: dict) -> None:
        """
        Checks the dependencies required to unfold the scenarios.

        Parameters:
        -----------
        dependencies : dict
            A dictionary containing the names of the databases required as dependencies.

        Returns:
        --------
        None

        Notes:
        ------
        This method checks that the required source database and scenarios are available and compatible with each other. If the dependencies are not provided, the method assumes that no dependencies are required.

        If the required dependencies are not found in the available databases, the method prompts the user to input the database number for each dependency.

        Once the user has provided the necessary information, the method updates the "source" attribute for each dependency with the name of the corresponding database.

        """
        # Get a list of available databases in the project
        available_databases = list(bw2data.databases.keys())

        # If no dependencies are provided, set an empty dict
        dependencies = dependencies or {}

        # Check if all dependencies are available in the project
        if dependencies and all(
            dependency in available_databases for dependency in dependencies.values()
        ):
            # If all dependencies are available, update the "source" attribute for each dependency
            for database in self.dependencies:
                database["source"] = dependencies[database["name"]]
        else:
            # If any dependencies are missing, prompt the user to input the necessary information
            print("The following databases are necessary to unfold the scenarios:")

            table = PrettyTable()
            table.field_names = ["No.", "Dependency", "System model", "Version"]

            # Add a row to the table for each dependency
            for _db, database in enumerate(self.dependencies):
                table.add_row(
                    [
                        _db + 1,
                        database["name"],
                        database.get("system model", ""),
                        database.get("version", ""),
                    ]
                )

            # Print the table of required dependencies
            print(table)
            print("")

            print("The following databases are available in your project:")

            table = PrettyTable()
            table.field_names = [
                "No.",
                "Database",
            ]

            # Add a row to the table for each available database
            for _db, database in enumerate(available_databases):
                table.add_row([_db + 1, database])

            # Print the table of available databases
            print(table)
            print("")

            # Prompt the user to input the database number for each dependency
            for _db, database in enumerate(self.dependencies):
                db_number = input(
                    f"Indicate the database number for dependency {_db + 1}: "
                )
                name = available_databases[int(db_number) - 1]
                database["source"] = name

    def build_mapping_for_dependencies(self, database) -> None:
        """
        Builds a mapping for dependencies based on the given database.

        Parameters:
        -----------
        database : list
            A list of activity dictionaries representing a database.

        Returns:
        --------
        None

        Notes:
        ------
        This method builds a mapping for dependencies based on the given database. The mapping is stored in the "dependency_mapping" attribute of the current object.

        The mapping is constructed using a dictionary comprehension, with each key being a tuple containing the name, reference product, location, and categories of the activity, and each value being a tuple containing the name and code of the database.

        """
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
        """
        Extracts the source database and builds a mapping for dependencies.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Notes:
        ------
        This method extracts the source database for each dependency in the "dependencies" attribute of the current object, and builds a mapping for dependencies based on the extracted databases. The mapping is stored in the "dependency_mapping" attribute of the current object.

        If a dependency has a type of "source", the extracted database is also added to the "database" attribute of the current object.

        """

        for dependency in self.dependencies:
            database = extract_brightway2_databases(dependency["source"])

            self.build_mapping_for_dependencies(database)
            if dependency.get("type") == "source":
                self.database.extend(database)

    def clean_imported_inventory(self, data):
        """
        Cleans the imported inventory.

        Parameters:
        -----------
        data : list
            A list of activity dictionaries representing the imported inventory.

        Returns:
        --------
        list
            A cleaned list of activity dictionaries representing the imported inventory.

        Notes:
        ------
        This method cleans the imported inventory by performing several operations on it:

        - remove_missing_fields(): Removes fields that are missing or contain only None values.
        - add_biosphere_links(): Adds biosphere links to exchanges that are missing them.
        - check_for_duplicates(): Checks for duplicate activities and exchanges.
        - add_product_field_to_exchanges(): Adds a "product" field to each exchange.
        - remove_categories_for_technosphere_flows(): Removes categories for technosphere exchanges.

        The cleaned inventory is returned as a list of activity dictionaries.

        """
        print("Cleaning imported inventory...")
        data = remove_missing_fields(data)
        add_biosphere_links(data)
        check_for_duplicates(self.database, data)
        add_product_field_to_exchanges(data, self.database)
        remove_categories_for_technosphere_flows(data)
        return data

    def extract_additional_inventories(self) -> None:
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

    def generate_factors(self) -> None:
        """
        Calculates factors for each flow in the scenario based on
        the scenario_df dataframe.

        The resulting factors are stored in the factors attribute,
        which is a dictionary that maps flow ids to their corresponding
        factors.

        Args:
            None

        Returns:
            None
        """
        self.factors = (
            self.scenario_df.groupby("flow id").sum(numeric_only=True).to_dict("index")
        )

    def get_list_unique_exchanges(self, databases: list) -> list:
        """
        Gets a list of all unique exchanges found in a list of databases.

        Args:
            databases (list): A list of Brightway2-style databases to extract unique exchanges from.

        Returns:
            A list of tuples representing unique exchanges, where each tuple contains the following information:
            - name: The name of the exchange.
            - product: The reference product of the exchange.
            - categories: The categories of the exchange.
            - location: The location of the exchange.
            - unit: The unit of the exchange.
            - type: The type of the exchange (either "technosphere" or "biosphere").
        """
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

    def store_datasets_metadata(self) -> None:
        """
        Stores metadata for each dataset in the database attribute.

        The resulting metadata is stored in the dict_meta attribute, which is a dictionary that maps dataset identifiers to their
        metadata.

        Args:
            None

        Returns:
            None
        """
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

    def generate_activities_indices(self) -> None:
        """Generates the indices of the unique activities."""
        # Get all unique activities from self.database.
        list_unique_acts = self.get_list_unique_exchanges(databases=[self.database])

        # Add additional exchanges to list_unique_acts.
        for act in list_unique_acts:
            if act[-1] == "production":
                new_id = list(act)
                new_id[-1] = "technosphere"
                new_id = tuple(new_id)
                if new_id not in list_unique_acts:
                    list_unique_acts.append(new_id)

        # Map each unique activity to a unique index.
        self.acts_indices = dict(enumerate(list_unique_acts))
        self.reversed_acts_indices = {act: i for i, act in enumerate(list_unique_acts)}

    def fetch_exchange_code(self, name: str, ref: str, loc: str) -> str:
        """
        :param name: name of the exchange
        :param ref: reference product of the exchange
        :param loc: location of the exchange
        :return: the exchange code corresponding to the specified name, reference product, and location
        Fetches the exchange code corresponding to the specified name,
        reference product, and location.
        """
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

    def populate_sparse_matrix(self) -> nsp.lil_matrix:
        """
        Generate a sparse matrix representation of the product system modeled by this object.

        The matrix is generated based on the data in the `database` attribute, which is assumed to be
        a list of dictionaries, where each dictionary represents a dataset in the product system.
        Each dataset has a list of exchanges, where each exchange is a dictionary with information
        about the input or output of a process.

        The matrix is generated as a `lil_matrix` object from the `scipy.sparse` module. The rows and
        columns of the matrix represent unique activities in the product system, which are determined
        by the unique combinations of the following exchange attributes: name, product, categories,
        location, unit, and type.

        The matrix is populated by looping over each exchange in each dataset, and adding the amount
        of the exchange to the corresponding row and column in the matrix.

        :return: A `lil_matrix` object representing the product system modeled by this object.
        """

        # Generate the indices for the activities
        self.generate_activities_indices()

        # Initialize a sparse matrix
        m = nsp.lil_matrix((len(self.acts_indices), len(self.acts_indices)))

        # Populate the matrix with exchange amounts
        for ds in self.database:
            for exc in ds["exchanges"]:
                # Source activity
                s = (
                    exc["name"],
                    exc.get("product"),
                    exc.get("categories"),
                    exc.get("location"),
                    exc["unit"],
                    exc["type"],
                )
                # Destination activity
                c = (
                    ds["name"],
                    ds.get("reference product"),
                    ds.get("categories"),
                    ds.get("location"),
                    ds["unit"],
                    "production",
                )
                # Add the exchange amount to the corresponding cell in the matrix
                m[self.reversed_acts_indices[s], self.reversed_acts_indices[c]] += exc[
                    "amount"
                ]

        return m

    def write_scaling_factors_in_matrix(
        self, matrix: np.ndarray, scenario_name: str
    ) -> np.ndarray:
        """
        Multiplies the elements of the given matrix with scaling factors for a given scenario.

        :param matrix: A 2D numpy array representing the matrix whose elements are to be scaled.
        :type matrix: numpy.ndarray

        :param scenario_name: A string representing the name of the scenario for which the scaling factors should be used.
        :type scenario_name: str

        :return: A 2D numpy array representing the scaled matrix, with the scaling factors applied to the appropriate elements.
        :rtype: numpy.ndarray
        """
        # Create a lambda function that returns 1.0 for zero values and the input value for non-zero values.
        # This is used to avoid multiplying by zero, which would result in a zero product.
        _ = lambda x: x if x != 0 else 1.0

        # Iterate over the scaling factors defined in `self.factors`.
        for flow_id, factor in self.factors.items():
            # Extract the components of the flow ID, which are used to look up the indices in the matrix.
            c_name, c_prod, c_loc, c_unit = list(flow_id)[:4]
            s_name, s_prod, s_loc, s_cat, s_unit, s_type = list(flow_id)[4:]

            # Look up the index of the consumer activity in the reversed activities index.
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

            # Look up the index of the supplier activity in the reversed activities index.
            supplier_id = (
                s_name,
                s_prod,
                s_cat,
                s_loc,
                s_unit,
                s_type,
            )
            supplier_idx = self.reversed_acts_indices[supplier_id]

            # Multiply the appropriate element of the matrix by the scaling factor for the given scenario.
            # Use the lambda function defined above to avoid multiplying by zero.
            matrix[supplier_idx, consumer_idx] = factor[scenario_name] * _(
                matrix[supplier_idx, consumer_idx]
            )

        # Return the scaled matrix.
        return matrix

    def get_act_dict_structure(self, ind: int, scenario_name: str) -> dict:
        """
        Get the structure of the activity/dataset dictionary.
        :param ind: index of the activity
        :param scenario_name: name of the scenario
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

    def build_superstructure_database(self, matrix: np.ndarray) -> list:
        """
        Builds a superstructure database from the given matrix, which is assumed to represent the exchanges between activities
        in a supply chain.

        :param matrix: A 3D numpy array representing the exchanges between activities in a supply chain.
        :type matrix: numpy.ndarray

        :return: A list of dictionaries representing the activities and exchanges in the supply chain.
        :rtype: list
        """
        # Print a message to indicate that the function is running.
        print(f"Generating superstructure database...")

        # Find the indices of the non-zero elements in the first matrix.
        non_zero_indices = sparse.argwhere(matrix[..., 0].T != 0)

        # Add the indices of the non-zero elements in the other matrices.
        for i in range(1, matrix.shape[-1]):
            non_zero_indices = np.concatenate(
                (non_zero_indices, sparse.argwhere(matrix[..., i].T != 0))
            )

        # Convert the indices to a list of tuples.
        non_zero_indices = list(map(tuple, non_zero_indices))

        # Group the indices by row in a dictionary.
        inds_d = defaultdict(list)
        for ind in non_zero_indices:
            if ind[1] not in inds_d[ind[0]]:
                inds_d[ind[0]].append(ind[1])

        # Build a list of activities and exchanges from the indices.
        new_db = []
        for k, v in inds_d.items():
            # Get the dictionary representing the current activity from the database.
            act = self.get_act_dict_structure(
                ind=k,
                scenario_name=self.package.descriptor["name"],
            )

            # Update the activity dictionary with metadata.
            act.update(self.dict_meta[self.acts_indices[k]])

            # Add exchanges to the activity dictionary for each non-zero element in the row.
            act["exchanges"].extend(
                self.get_exchange(
                    ind=j,
                    amount=matrix[j, k, 0],
                    scenario_name=self.package.descriptor["name"],
                )
                for j in v
            )

            # Add the updated activity dictionary to the list of activities and exchanges.
            new_db.append(act)

        # Return the list of activities and exchanges.
        return new_db

    def build_single_databases(
        self, matrix, databases_to_build: List[dict]
    ) -> list[list[dict]]:
        """
        Generate a list of single databases for each scenario specified in `databases_to_build`.

        :param matrix: A 3D numpy array representing the exchanges between activities in a supply chain.
        :type matrix: numpy.ndarray
        :param databases_to_build: A list of dictionaries, where each dictionary contains the name of a scenario and other scenario-specific metadata.
        :type databases_to_build: list[dict]
        :return: A list of single databases for each scenario specified in `databases_to_build`. Each database is a list of activity dictionaries.
        :rtype: list[list[dict]]

        :raises ValueError: If `databases_to_build` is empty.
        :raises ValueError: If the number of scenarios in `databases_to_build` does not match the number of scenarios in `matrix`.

        :notes:
        - The `matrix` parameter is a 3-dimensional array, with dimensions (n,m,p), where n and m are the number of rows
          and columns of the matrix and p is the number of scenarios.
        - The first two dimensions correspond to the producer and consumer indices of the exchanges, and the third dimension
          corresponds to the scenario index.
        - The matrix is assumed to be sparse, with zeros representing missing or invalid values.
        - Each database is represented as a list of activity dictionaries, where each activity dictionary contains the
          metadata and exchanges for a specific activity in the scenario.
        """

        databases_to_return = []

        for ix, i in enumerate(databases_to_build):
            # Print a message indicating that a database is being generated for the current scenario.
            print(f"Generating database for scenario {i['name']}...")

            # Get the indices of non-zero elements in the matrix for the current scenario.
            non_zero_indices = sparse.argwhere(matrix[..., ix].T != 0)
            non_zero_indices = list(map(tuple, non_zero_indices))

            # Create a dictionary that maps producer indices to lists of consumer indices for the non-zero elements in the matrix.
            inds_d = defaultdict(list)
            for ind in non_zero_indices:
                inds_d[ind[0]].append(ind[1])

            new_db = []

            # For each producer index, create an activity dictionary and add it to the current scenario's database.
            for k, v in inds_d.items():
                act = self.get_act_dict_structure(
                    ind=k,
                    scenario_name=i["name"],
                )
                act.update(self.dict_meta[self.acts_indices[k]])

                # For each consumer index associated with the current producer index, create an exchange dictionary and add it to the activity's exchanges list.
                act["exchanges"].extend(
                    self.get_exchange(
                        ind=j, amount=matrix[j, k, ix], scenario_name=i["name"]
                    )
                    for j in v
                )
                new_db.append(act)

            # Add the current scenario's database to the list of databases to return.
            databases_to_return.append(new_db)

        return databases_to_return

    def generate_superstructure_database(self) -> List[dict]:
        """
        Generates the superstructure database.

        Returns:
        - A list of activity dictionaries representing the superstructure database.

        Notes:
        - This function first populates a sparse matrix with the exchanges between activities in the supply chain.
        - It then creates a 3D numpy array by stacking the sparse matrix with scaled versions of the matrix for each scenario in `self.scenarios`.
        - Finally, it uses the 3D numpy array to generate the superstructure database by calling the `build_superstructure_database()` function.
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
        """
        Generates single databases for each scenario in `self.scenarios`.

        Returns:
        - A list of lists of activity dictionaries representing the single databases.

        Notes:
        - This function first populates a sparse matrix with the exchanges between activities in the supply chain.
        - It then creates a 3D numpy array by stacking scaled versions of the matrix for each scenario in `self.scenarios`.
        - Finally, it uses the 3D numpy array to generate single databases for each scenario by calling the `build_single_databases()` function.
        """
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
        """
        Formats the scenario dataframe.

        Parameters:
        - `scenarios`: A list of scenario indices to keep in the dataframe. If `None`, keeps all scenarios.
        - `superstructure`: If `True`, adds columns to the dataframe for the superstructure.

        Notes:
        - This function removes unused scenarios from `self.scenarios`, drops columns from the scenario dataframe that correspond to scenarios that were removed, and converts `None` and `np.nan` values to `None`.
        - If `superstructure` is `True`, this function adds columns to the scenario dataframe for the superstructure by calling the `get_suppliers()` and `get_consumers()` functions.
        """

        scenarios = scenarios or list(range(len(self.scenarios)))
        scenarios_to_keep = [self.scenarios[i]["name"] for i in scenarios]

        # Remove scenarios that are not in `scenarios_to_keep` from `self.scenarios`.
        self.scenarios = [s for s in self.scenarios if s["name"] in scenarios_to_keep]

        # Find the scenarios to leave out of the scenario dataframe.
        scenarios_to_leave_out = list(
            set(s["name"] for s in self.scenarios) - set(scenarios_to_keep)
        )

        # Load the scenario dataframe from the `scenario_data` resource.
        self.scenario_df = pd.DataFrame(
            self.package.get_resource("scenario_data").read(keyed=True)
        )

        # Remove rows corresponding to production flows.
        self.scenario_df = self.scenario_df.loc[
            (self.scenario_df["flow type"] != "production")
        ]

        # Drop columns corresponding to scenarios that were removed.
        self.scenario_df = self.scenario_df.drop(scenarios_to_leave_out, axis=1)

        # Convert "None" and np.nan values to None.
        self.scenario_df = self.scenario_df.replace("None", None)
        self.scenario_df = self.scenario_df.replace({np.nan: None})

        # Convert strings to Python objects in columns that contain lists or tuples.
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

        # Convert columns corresponding to scenario exchange amounts to float dtype.
        self.scenario_df[scenarios_to_keep] = self.scenario_df[
            scenarios_to_keep
        ].astype(float)

        # Add a "flow id" column to the scenario dataframe.
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

    def format_superstructure_dataframe(self) -> None:
        """
        Formats the superstructure dataframe.

        Notes:
        - This function creates a new scenario dataframe for the superstructure by populating a sparse matrix with data from the self.factors dictionary.
        - The superstructure dataframe contains columns for each scenario as well as additional columns for database information and metadata.
        """

        # Create a sparse matrix with data from self.factors
        matrix = self.populate_sparse_matrix()

        # Define a lambda function to replace zero values with 1.0
        _ = lambda x: x if x != 0 else 1.0

        # Loop through each flow in self.factors and update the corresponding factor values
        for flow_id, factor in self.factors.items():
            c_name, c_prod, c_loc, c_unit = list(flow_id)[:4]
            s_name, s_prod, s_loc, s_cat, s_unit, s_type = list(flow_id)[4:]

            # Get the indices of the consumer and supplier activities in the reversed_act_indices dictionary
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

            # Update the factor values for each scenario by multiplying with the corresponding matrix value
            for scenario, val in factor.items():
                factor[scenario] = val * _(matrix[consumer_idx, supplier_idx])

        # Create a new scenario dataframe from the updated factors dictionary
        self.scenario_df = pd.DataFrame.from_dict(self.factors).T.reset_index()

        # Rename columns and add new columns for database information and metadata
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

        # Use the dependency_mapping dictionary to add "from key" and "to key" columns
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

        # Add "from database" column based on flow type
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
        """
        unfold() is a method of the Unfol class, which extracts specific scenarios from the input LCA database and writes them as new databases.

        :param scenarios: A list of integers indicating the indices of the scenarios to extract. If None, all scenarios are extracted. Default is None.
        :param dependencies: A dictionary containing additional inventory databases that may be needed for extraction. Default is None.
        :param superstructure: A boolean indicating whether to generate a scenario difference file and a superstructure database. Default is False.
        :return: None

        Behavior:

        Calls the check_dependencies() method to ensure that all required inventory databases are present.
        Calls the extract_source_database() method to extract the original LCA database.
        Calls the extract_additional_inventories() method to extract any additional inventory databases required.
        Calls the format_dataframe() method to format the scenario data into a Pandas DataFrame.
        Calls the generate_factors() method to generate the factors for the scenarios.
        If superstructure is False, calls the generate_single_databases() method to generate a separate LCA database for each scenario, and stores them in the databases_to_export dictionary.
        If superstructure is True, calls the format_superstructure_dataframe() method to generate a scenario difference file and a superstructure database.
        Calls the write() method to write the databases to disk.

        """

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
