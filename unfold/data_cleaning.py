import csv
import uuid
from pathlib import Path
from typing import List, Tuple, Union

import wurst as ws
import yaml
from prettytable import PrettyTable

DATA_DIR = Path(__file__).resolve().parent / "data"
FILEPATH_BIOSPHERE_FLOWS = DATA_DIR / "flows_biosphere_38.csv"
OUTDATED_FLOWS = DATA_DIR / "outdated_flows.yaml"


def get_outdated_flows() -> dict:
    """
    Retrieve a list of outdated flows from the outdated flows file.
    """

    with open(OUTDATED_FLOWS, "r", encoding="utf-8") as stream:
        flows = yaml.safe_load(stream)

    return flows


def get_biosphere_code() -> dict:
    """
    Retrieve a dictionary with biosphere flow names and uuid codes.
    :returns: dictionary with biosphere flow names as keys and uuid codes as values

    """

    if not FILEPATH_BIOSPHERE_FLOWS.is_file():
        raise FileNotFoundError("The dictionary of biosphere flows could not be found.")

    csv_dict = {}

    with open(FILEPATH_BIOSPHERE_FLOWS, encoding="utf-8") as file:
        input_dict = csv.reader(file, delimiter=";")
        for row in input_dict:
            csv_dict[(row[0], row[1], row[2], row[3])] = row[4]

    return csv_dict


biosphere_dict = get_biosphere_code()
outdated_flows = get_outdated_flows()


def remove_missing_fields(data: List[dict]) -> List[dict]:
    """
    Remove any field that does not have information.
    """

    FORBIDDEN_FIELDS_TECH = [
        "categories",
    ]

    FORBIDDEN_FIELDS_BIO = [
        "location",
    ]

    def clean_up(exc: dict) -> dict:
        """Cleans up an exchange"""

        for field in list(exc.keys()):
            if exc[field] is None or exc[field] == "None":
                del exc[field]
            if exc["type"] == "biosphere" and field in FORBIDDEN_FIELDS_BIO:
                del exc[field]
            if exc["type"] == "technosphere" and field in FORBIDDEN_FIELDS_TECH:
                del exc[field]

        return exc

    for dataset in data:
        for key, value in list(dataset.items()):
            if not value:
                del dataset[key]

        dataset["exchanges"] = [clean_up(exc) for exc in dataset["exchanges"]]

    return data


def check_exchanges_input(database, input_mapping):
    """
    Checks that all biosphere exchanges are given an input code

    """
    for dataset in database:
        for exc in dataset["exchanges"]:
            if exc["type"] == "biosphere":
                if "input" not in exc or exc.get("input") is None:
                    exc["input"] = input_mapping.get(
                        (
                            exc["name"],
                            exc.get("product"),
                            exc.get("location"),
                            exc.get("categories"),
                        )
                    )

    return database


def add_biosphere_links(data: List[dict], delete_missing: bool = False) -> List[dict]:
    """Add links for biosphere exchanges to :attr:`import_db`
    Modifies the :attr:`import_db` attribute in place.
    Also checks for outdated biosphere flows and replaces them with the
    current ones.

    :param data: list of dictionaries with the data to be imported
    :param delete_missing: whether unlinked exchanges should be deleted or not.
    """
    for x in data:
        for y in x["exchanges"]:
            if y["type"] == "biosphere":
                if y["name"] in outdated_flows:
                    y["name"] = outdated_flows[y["name"]]

                if isinstance(y["categories"], str):
                    y["categories"] = tuple(y["categories"].split("::"))
                if len(y["categories"]) > 1:
                    try:
                        key = (
                            y["name"],
                            y["categories"][0],
                            y["categories"][1],
                            y["unit"],
                        )
                        if key in biosphere_dict:
                            y["input"] = (
                                "biosphere3",
                                biosphere_dict[key],
                            )
                        else:
                            if key[0] in outdated_flows:
                                new_key = list(key)
                                new_key[0] = outdated_flows[key[0]]
                                y["input"] = (
                                    "biosphere3",
                                    biosphere_dict[tuple(new_key)],
                                )
                            else:
                                if delete_missing:
                                    y["flag_deletion"] = True
                                else:
                                    print(f"Could not find a biosphere flow for {key}")

                    except KeyError:
                        if delete_missing:
                            y["flag_deletion"] = True
                        else:
                            raise
                else:
                    try:
                        y["input"] = (
                            "biosphere3",
                            biosphere_dict[
                                (
                                    y["name"].strip(),
                                    y["categories"][0].strip(),
                                    "unspecified",
                                    y["unit"].strip(),
                                )
                            ],
                        )
                    except KeyError:
                        if delete_missing:
                            print(
                                f"The following biosphere exchange: "
                                f"{y['name']}, {y['categories'][0]}, unspecified, {y['unit']} "
                                f"in {x['name']}, {x['location']}"
                                f" cannot be found and will be deleted."
                            )
                            y["flag_deletion"] = True
                        else:
                            raise
        x["exchanges"] = [ex for ex in x["exchanges"] if "flag_deletion" not in ex]

    return data


def check_for_duplicates(db: List[dict], data: List[dict]) -> List[dict]:
    """
    Check whether the inventories to be imported are not
    already in the source database.
    :param db: list of dictionaries representing the source database
    :param data: list of dictionaries representing the inventories to be imported
    :returns: inventories without duplicates
    """

    # print if we find datasets that already exist
    db_names = [
        (x["name"].lower(), x["reference product"].lower(), x["location"]) for x in db
    ]

    already_exist = [
        (x["name"].lower(), x["reference product"].lower(), x["location"])
        for x in data
        if (x["name"].lower(), x["reference product"].lower(), x["location"])
        in db_names
    ]

    if len(already_exist) > 0:
        print(
            "The following datasets from the import file already exist in the source database:"
        )
        table = PrettyTable(["Name", "Reference product", "Location"])

        for dataset in already_exist:
            table.add_row([dataset[0][:50], dataset[1][:30], dataset[2]])

        print(table)
        print("They will not be imported.")

    return [
        x
        for x in data
        if (x["name"], x["reference product"], x["location"]) not in db_names
    ]


def add_product_field_to_exchanges(data: List[dict], db: List[dict]) -> List[dict]:
    """Add the `product` key to the production and
    technosphere exchanges in :attr:`import_db`.
    Also add `code` field if missing.
    For production exchanges, use the value of the `reference_product` field.
    For technosphere exchanges, search the activities in :attr:`import_db` and
    use the reference product. If none is found, search the Ecoinvent :attr:`database`.
    Modifies the :attr:`import_db` attribute in place.
    :param data: list of dictionaries with the data to be imported
    :raises IndexError: if no corresponding activity (and reference product) can be found.

    """
    # Add a `product` field to the production exchange
    for dataset in data:
        for exchange in dataset["exchanges"]:
            if exchange["type"] == "production":
                if "product" not in exchange:
                    exchange["product"] = dataset["reference product"]

                if exchange["name"] != dataset["name"]:
                    exchange["name"] = dataset["name"]

    # Add a `product` field to technosphere exchanges
    for dataset in data:
        for exchange in dataset["exchanges"]:
            if exchange["type"] == "technosphere":
                # Check if the field 'product' is present
                if "product" not in exchange:
                    exchange["product"] = correct_product_field(exchange, data, db)

    # Add a `code` field if missing
    for dataset in data:
        if "code" not in dataset:
            dataset["code"] = str(uuid.uuid4().hex)

    return data


def correct_product_field(exc: dict, data: List[dict], database: List[dict]) -> str:
    """
    Find the correct name for the `product` field of the exchange
    :param exc: a dataset exchange
    :return: name of the product field of the exchange

    """
    # Look first in the imported inventories
    candidate = next(
        ws.get_many(
            data,
            ws.equals("name", exc["name"]),
            ws.equals("location", exc["location"]),
            ws.equals("unit", exc["unit"]),
        ),
        None,
    )

    # If not, look in the ecoinvent inventories
    if candidate is None:
        candidate = next(
            ws.get_many(
                database,
                ws.equals("name", exc["name"]),
                ws.equals("location", exc["location"]),
                ws.equals("unit", exc["unit"]),
            ),
            None,
        )

    if candidate is not None:
        return candidate["reference product"]

    print(
        f"An inventory exchange in cannot be linked to the "
        f"biosphere or the ecoinvent database: {exc}"
    )

    return exc["reference product"]


def remove_categories_for_technosphere_flows(data):
    """
    Remove the categories field for technosphere flows
    :param data: list of dictionaries representing the inventories to be imported
    :returns: inventories
    """
    for x in data:
        for y in x["exchanges"]:
            if y["type"] in ["technosphere", "production"]:
                if "categories" in y:
                    del y["categories"]
    return data


def correct_fields_format(data: list, name: str) -> list:
    """
    Correct the format of some fields.
    :param data: database to check
    :return: database with corrected fields
    """

    for dataset in data:
        if "parameters" in dataset:
            if not isinstance(dataset.get("parameters", []), list):
                dataset["parameters"] = [dataset["parameters"]]

            if (
                dataset["parameters"] is None
                or dataset["parameters"] == {}
                or dataset["parameters"] == []
            ):
                del dataset["parameters"]

            else:
                new_parameters_list = []
                for p in dataset["parameters"]:
                    for k, v in p.items():
                        new_parameters_list.append({"name": k, "amount": v})
                dataset["parameters"] = new_parameters_list

        if "categories" in dataset:
            if dataset["categories"] is None:
                del dataset["categories"]
            elif not isinstance(dataset["categories"], tuple):
                dataset["categories"] = tuple(dataset["categories"])
            else:
                pass

        if not dataset.get("database"):
            dataset["database"] = name

    return data


def check_mandatory_fields(data: list) -> list:
    """
    Check that the mandatory fields are present.
    :param data: list of dictionaries representing the inventories to be imported
    :raises ValueError: if a mandatory field is missing
    :return: list of dictionaries representing the inventories to be imported
    """

    dataset_fields = [
        "name",
        "reference product",
        "location",
        "unit",
        "exchanges",
    ]

    missing_fields = []

    for dataset in data:
        for field in dataset_fields:
            if field not in dataset:
                if (
                    field in ["reference product", "location", "unit", "name"]
                    and "exchanges" in dataset
                ):
                    for exc in dataset["exchanges"]:
                        if exc["type"] == "production":
                            if field == "reference product":
                                dataset[field] = exc.get("product")
                            else:
                                dataset[field] = exc.get(field)

                if dataset.get(field) is None:
                    missing_fields.append(
                        [
                            dataset.get("database", "unknown"),
                            dataset.get("name", "unknown"),
                            dataset.get("reference product", "unknown"),
                            dataset.get("location", "unknown"),
                            field,
                        ]
                    )

    if missing_fields:
        # print in prettytable the list of missing fields
        table = PrettyTable()
        table.field_names = [
            "Database",
            "Dataset",
            "Reference product",
            "Location",
            "Missing field",
        ]
        for row in missing_fields[:10]:
            table.add_row(row)
        print(table)
        raise ValueError(
            "Some mandatory fields are missing in the database. "
            "Ten first missing fields are displayed above."
        )

    return data


def get_list_of_unique_datasets(data: list) -> list:
    """
    Return a list of unique datasets from a list of datasets.
    :param data: list of dictionaries representing the inventories to be imported
    :return: list of dictionaries representing the inventories to be imported
    """
    unique_datasets = []
    for dataset in data:
        if dataset not in unique_datasets:
            unique_datasets.append(
                (
                    dataset["name"],
                    dataset["reference product"],
                    dataset["location"],
                    dataset["unit"],
                )
            )
    return unique_datasets


def check_commonality_between_databases(original_db, scenario_db, db_name):
    """
    Check that scenario databases have at least one dataset in common
    with the original database.
    :param original_db: original
    :param scenario_db: list of scenario databases
    :param db_name: name of the scenario database
    :raises ValueError: if no dataset is in common
    """

    original_db_unique_datasets = get_list_of_unique_datasets(original_db)

    scenario_db_unique_datasets = get_list_of_unique_datasets(scenario_db)
    if not set(original_db_unique_datasets).intersection(scenario_db_unique_datasets):
        raise ValueError(
            "Could not find datasets in common between the reference database "
            f"and {db_name}."
        )
