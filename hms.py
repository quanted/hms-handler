import os
import json
import requests
import time
import logging

logger = logging.getLogger(__name__)
hms_base_url = os.getenv("HMS_URL", "https://ceamstg.ceeopdev.net/hms/rest/api/v3/")
hms_data_url = os.getenv("HMS_DATA", "https://ceamstg.ceeopdev.net/hms/rest/v2/hms/data?job_id=")


class HMS:

    def __init__(self, start_date=None, end_date=None, source=None, dataset=None, module=None):
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.dataset = dataset
        self.module = module
        self.geometry = {}
        self.task_id = None
        self.task_status = None
        self.data = None

    def set_geometry(self, gtype="point", value=None):
        if gtype == "point":
            self.geometry["point"] = value
        elif gtype == "comid":
            self.geometry["comid"] = value
        else:
            logger.info("Supported geometry type")

    def get_request_body(self):
        if any((self.dataset, self.source, self.start_date, self.end_date, self.geometry, self.module)) is None:
            logger.info("Missing required parameters, unable to create request.")
            return None
        request_body = {
            "dataset": self.dataset,
            "source": self.source,
            "datetimespan": {
                "startdate": self.start_date,
                "enddate": self.end_date
            },
            "geometry": self.geometry
        }
        return request_body

    def submit_request(self):
        params = self.get_request_body()
        if params is None:
            self.task_status = "FAILED: Parameters invalid"
            return None
        request_url = hms_base_url + self.module + "/" + self.dataset + "/"
        logger.info("Submitting data request.")
        try:
            response_txt = requests.post(request_url, data=params).text
        except ConnectionError as error:
            self.task_status = "FAILED: Failed Request"
            logger.info("WARNING: Failed data request")
            return None
        response_json = json.loads(response_txt)
        self.task_id = response_json["job_id"]
        self.task_status = response_json["status"]

    def get_data(self):
        if self.task_id is None:
            logger.info("WARNING: No task id")
            self.task_status = "FAILED: No task id"
            return None
        time.sleep(2)
        retry = 0
        n_retries = 25
        data_url = hms_data_url + self.task_id
        success_fail = False
        while retry < n_retries or success_fail:
            response_txt = requests.get(data_url).text
            response_json = json.loads(response_txt)
            self.task_status = response_json["status"]
            if self.task_status == "SUCCESS":
                self.data = response_json["data"]
                success_fail = True
            elif self.task_status == "FAILURE":
                success_fail = True
            else:
                retry += 1
                time.sleep(0.5 * retry)
        if self.data is None:
            self.task_status = "FAILED: Retry timeout"
