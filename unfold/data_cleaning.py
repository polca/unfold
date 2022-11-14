from pathlib import Path
import yaml
import csv
import uuid
import wurst as ws
from prettytable import PrettyTable
from typing import List, Tuple, Union

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

    for dataset in data:
        for key, value in list(dataset.items()):
            if not value:
                del dataset[key]

    return data

def add_biosphere_links(data: List[dict], delete_missing: bool = False) -> List[dict]:
    """Add links for biosphere exchanges to :attr:`import_db`
    Modifies the :attr:`import_db` attribute in place.

    :param data: list of dictionaries with the data to be imported
    :param delete_missing: whether unlinked exchanges should be deleted or not.
    """
    for x in data:
        for y in x["exchanges"]:
            if y["type"] == "biosphere":
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
                                    print(
                                        f"Could not find a biosphere flow for {key}"
                                    )

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
    db_names = [(x["name"].lower(), x["reference product"].lower(), x["location"]) for x in db]

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
                if not "product" in exchange:
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