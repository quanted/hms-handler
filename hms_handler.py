import os
import time
import json
import csv
import sys
import sqlite3
import argparse
import logging
import multiprocessing as mp
from hms import HMS

logger = logging.getLogger(__name__)
db_path = "runoff_data.sqlite3"


def load_file(filepath, c_dict=False):
    logger.info("Checking for file: {}".format(filepath))
    if os.path.exists(filepath):

        with open(filepath, newline='') as csvfile:
            logger.info("Loading file: {}".format(filepath))
            if c_dict:
                data = {}
                csv_dict = csv.DictReader(csvfile)
                for row in csv_dict:
                    data = row
            else:
                data = []
                csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csv_reader:
                    data.append(row)
        return data
    else:
        logger.fatal("File {} not found".format(filepath))
        sys.exit()


#TODO: Create db loader, db saver, parallel catchment initializer and downloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMS Data downloader")
    parser.add_argument("-l", dest="locs", required=True,
                        help="Path to file containing location data: comid, physiogeographic division and province")
    parser.add_argument("-c", dest="configs", required=True,
                        help="Path to file containing request configuration data: start date, end date, dataset, "
                             "geometry")
    args = parser.parse_args()
    catchment_file = args.locs
    configs_file = args.configs

    catchments = load_file(catchment_file)
    configs = load_file(configs_file, c_dict=True)
