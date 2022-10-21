#!/usr/bin/env python3

documentation = """
+ All data are assumed to be stored in a directory whose contents are:
  + categories.tsv: Level-1 categories (ignored at the moment)
  + food_976759.json: List of Level3 (and Level4, e.g. 976759_976794_7981173_5580286)
    categories)
    + features: "name", "id", "itemCount"
  + data: (directory) contains (JSON) dictionaries of pages of products
    + category_id = key in each dictionary
    + category_id has the form <level1 category>_<level2 category>_<level3 category>...
    + pages indexed by integers: "1", "2", ...
    + products indexed by key, e.g. "4PR4FBQD9P50" (=product_id)
    + We're interested in "shortDescription"
"""

notes = """
+ Each product belongs to a hierarchy of categories
  (TBC: each product seems to belong to exactly one hierarchy of categories)
+ Categories:
  + Each category has its own numeric id, e.g.:
    + Food is 976759
    + Pantry is 976794 but its id is 976759_976794 (it's a subcategory of food)
    + Condiments is 7981173 (id: 976759_976794_7981173)
    + Salad Dressings & Toppings is 5580286 (id: 976759_976794_7981173_5580286)

+ Data set is in $HOME/MachineLearning/mahdi/data-20221013

Notes:
+ categories.tsv is unlikely to change. We ignore it for the moment
+ Two types: products and level_n categories (with n >= 1)
+ Don't need pagination in product files
+ Minimize operations on original data
"""

import datasets as ds
import os
import json
import logging
import lxml.html as html

logger = logging.getLogger(__name__)
ds.utils.logging.set_verbosity(logging.ERROR)
ds.utils.logging.disable_progress_bar()


def scrub_html(string):
    """
    Remove all HTML elements from string
    """
    if string is None:
        return ""
    _html_list = html.fragments_fromstring(string)
    _str_list = []
    for e in _html_list:
        try:
            if type(e) is str:
                _str_list.append(e)
            else:
                if e.text is not None:
                    _str_list.append(e.text)
        except TypeError as e:
            logger.error("Type %s not supported", type(e))
            logger.error(e)
            raise e
    return ";".join(_str_list)


def dig(json_object, category_id = None):
    """
    We want to dig into the original data. They initially look like:
    { <cat_id>: 1: {product1, product2, product3...}, 2: {product10, product11, product12...}, ... }
    What we want is a list of all products and we want to add the cat_id in each product
    
    We explore (=recurrence) the json object. The cat_id is not known until the first level
    is explored (and there is only one cat_id so it's easy)
    Then we explore the page-level (1, 2, .. indices) to find each product
    """
    if "id" in json_object:
        # We know that it is a product since it has an "id" data member
        # Add the category_id to the object
        json_object["category_id"] = category_id
        # Scrub HTML from shortDescription 
        logger.debug("Before: %s", json_object["shortDescription"])
        json_object["shortDescription"] = scrub_html(json_object["shortDescription"])
        logger.debug(" After: %s", json_object["shortDescription"])
        # ... and return it as is, i.e. return a object
        # which has the JSON object type (this matters)
        return json_object
    else:
        # We are not at the product json-object level... It is one of the levels above
        if category_id is None:
            # category_id is not known which means that it is the first level
            # There is only one key which is the category_id
            category_id = next(iter(json_object.keys()))
        # results is a list where the products are collected
        results = []
        for value in json_object.values():  # We don't care about the keys here
            # We are interested in the values of the current level, i.e.
            # the pages (if at the top level) or the json-objects (if at the page one)
            result = dig(value, category_id)
            # We've returned from the inspection at lower level
            # There are only two possibilities:
            if type(result) is list:
                # We've returned from a page or a level-3 exploration, i.e.
                # we have a list of JSON objects and we want to add them all
                # to the results
                results.extend(result)
            else:
                # We've returned from a JSON object (i.e. not a list) and we
                # just want to add that object to the list of results
                results.append(result)
        # Return the list of JSON objects collected by the exploration of this level
        return results


NO_NEED_TO_TELL_ME_TWICE = set()
def consolidate_keys(products):
    """
    Here we just want to make sure that all products (JSON objects)
    have the same keys. We add None if a key is missing.
    """
    # Collect all keys for all products
    keys = set()
    for product in products:
        keys.update(product.keys())
    logger.debug("All keys = %s", keys)
    # Add missing keys to each product
    for product in products:
        for key in keys:
            if not key in product.keys():
                if not key in NO_NEED_TO_TELL_ME_TWICE:
                    logger.warning("Key %s is missing in %s", key, product["id"])
                    NO_NEED_TO_TELL_ME_TWICE.add(key)
                product[key] = None
    return products


def flatten(product, prefix = ""):
    """The products contain JSON objects. Similarly to the huggingface
    flatten() method, we flatten them, i.e. if they look like:
    { "k1": "v1", "k2": { "k21": "v21", "k22": "v22" } }
    they become:
    { "k1": "v1", "k2_k21": "v21", "k2_k22": "v22" }

    Note: 
    * For some reason, huggingface don't like '.' for separators,
      hence the use of '_'
    + If lists are added in the JSON objects/products, this method will 
      need some rewriting
    """
    if prefix != "":
        logger.debug("Flattening for prefix", prefix)
    n_product = {}
    for key,value in product.items():
        if not type(value) is dict:
            # the value is a primitive JSON type
            n_product[prefix + key] = value
        else:
            # the value is a Python-dictionary = JSON object 
            subdictionary = flatten(value, prefix = prefix + key + "_")
            n_product.update(subdictionary)
    return n_product


def process_arguments(arguments):
    """
    Process the arguments on the command line
    """
    import argparse
    parser = argparse.ArgumentParser(description="""
Consolidate data into a single dataset.

%s

""" % documentation)
    parser.add_argument('path_to_data', type=str,
                        help="The directory containing data")
    parser.add_argument('-d', '--debug', required=False, default=False,
                        help="Debugging message")
    args = parser.parse_args(arguments)
    loglevel = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s.%(msecs)03d | %(levelname)7s | %(message)s',
                        datefmt="%Y-%m-%dT%H:%M:%S",
                        level=loglevel)
    logger.info("Logging with level %s", loglevel)
    return args.path_to_data


def main(arguments):
    path_to_data = process_arguments(arguments[1:])
    # path_to_data = "/home/sc/MachineLearning/mahdi/data-20221013"

    # food_category = ds.load_dataset("json", data_files = "%s/%s" %
    #                                 (path_to_data, "food_976759.json"))
    # Not sure what to do with this: Do we care?
    
    product_files = ["%s/data/%s" % (path_to_data, filename) for filename
                     in os.listdir("%s/%s" % (path_to_data, "data"))]
    logger.info("There are %d product files to parse" % len(product_files))

    # Notes:
    # + ds.load_dataset("json", data_files=product_files) doesn't work
    #   "because column names don't match"
    # + ds.load_dataset("json", data_files=product_file) for product_file in product_files
    #   is awfully slow (if not parallelized)
    # + going thru pandas.read_json() didn't help or speed up
    products = []
    for product_file in product_files:  # This could be parallelized if needed
        logger.debug("Reading %s", product_file)
        with open(product_file) as pf:
            json_product = json.load(pf)
            products.extend(dig(json_product))
    # Make sure all products have the same keys
    products = consolidate_keys(products)
    # Flatten the products. TODO Parallelize? 
    flattened_products = [flatten(product) for product in products]
    # Write a JSON file containing all these entries
    with open("all.json", "w") as out:
        json.dump(flattened_products, out)

    # Read it as a Huggingface-dataset
    products = ds.load_dataset("json", data_files = json_outfile)
    # And dump it
    print(products)
    # Saving is idempotent but for the train/state.json fingerprint element
    outdir = "prepared-%s" % path_to_data.split('/')[-1]
    logger.info("Writing output huggingface dataset to %s", outdir)
    products.save_to_disk(outdir)
    os.unlink(json_outfile)
    return outdir
    
if __name__ == "__main__":
    import sys
    main(sys.argv)
