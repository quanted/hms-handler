import os
import json
import requests
import time
import logging

logger = logging.getLogger(__name__)
# hms_base_url = os.getenv("HMS_URL", "https://ceamdev.ceeopdev.net/hms/rest/api/v3/")
# hms_data_url = os.getenv("HMS_DATA", "https://ceamdev.ceeopdev.net/hms/rest/api/v2/hms/data?job_id=")
hms_base_url = os.getenv("HMS_URL", "https://qed.edap-cluster.com/hms/rest/api/v3/")
hms_data_url = os.getenv("HMS_DATA", "https://qed.edap-cluster.com/hms/rest/api/v2/hms/data?job_id=")


class HMS:

    def __init__(self, start_date=None, end_date=None, source=None, dataset=None, module=None, cookies=None):
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.dataset = dataset
        self.module = module
        self.geometry = {}
        self.task_id = None
        self.task_status = None
        self.data = None
        self.cookies = cookies
        self.comid = None
        self.metadata = None

    def set_geometry(self, gtype="point", value=None, metadata=None):
        if gtype == "point":
            self.geometry["point"] = value
        elif gtype == "comid":
            self.comid = value
            self.geometry["comid"] = value
        else:
            logger.info("Supported geometry type")
        if metadata:
            self.geometry["geometryMetadata"] = metadata

    def get_request_body(self):
        if any((self.dataset, self.source, self.start_date, self.end_date, self.geometry, self.module)) is None:
            logger.info("Missing required parameters, unable to create request.")
            return None
        request_body = {
            "source": self.source,
            "dateTimeSpan": {
                "startDate": self.start_date,
                "endDate": self.end_date
            },
            "geometry": self.geometry,
            "temporalResolution": "daily"
        }
        return request_body

    def submit_request(self):
        params = json.dumps(self.get_request_body())
        if params is None:
            self.task_status = "FAILED: Parameters invalid"
            return None
        request_url = hms_base_url + self.module + "/" + self.dataset + "/"
        header = {"Referer": request_url}
        logger.info("Submitting data request.")
        try:
            response_txt = requests.post(request_url, data=params, cookies=self.cookies, headers=header).text
        except ConnectionError as error:
            self.task_status = "FAILED: Failed Request"
            logger.info("WARNING: Failed data request")
            return None
        response_json = json.loads(response_txt)
        self.task_id = response_json["job_id"]
        self.task_status = "SENT"
        self.get_data()

    def get_data(self):
        if self.task_id is None:
            logger.info("WARNING: No task id")
            self.task_status = "FAILED: No task id"
            return None
        time.sleep(5)
        retry = 0
        n_retries = 100
        data_url = hms_data_url + self.task_id
        success_fail = False
        while retry < n_retries and not success_fail:
            response_txt = requests.get(data_url, cookies=self.cookies).text
            response_json = json.loads(response_txt)
            self.task_status = response_json["status"]
            if self.task_status == "SUCCESS":
                self.data = response_json["data"]
                success_fail = True
            elif self.task_status == "FAILURE":
                success_fail = True
                print("Failure: {}".format(response_json))
            else:
                retry += 1
                time.sleep(0.5 * retry)
        if retry == n_retries:
            self.task_status = "FAILED: Retry timeout"


if __name__ == "__main__":
    start_date = "01-01-2000"
    end_date = "12-31-2017"
    source = "nldas"
    dataset = "surfacerunoff"
    module = "hydrology"
    # cookies = {'sessionid': 'lmufmudjybph2r3ju0la15x5vuovz1pw'}
    cookies = {'sessionid': 'b5c5ev7usauevf2nro7e8mothmekqsnj'}
    t0 = time.time()
    hms = HMS(start_date=start_date,
              end_date=end_date,
              source=source,
              dataset=dataset,
              module=module,
              cookies=cookies)
    geometry = 20867042
    hms.set_geometry('comid', value=geometry)
    hms.submit_request()
    t1 = time.time()
    print("Results")
    print("Status: {}, Task ID: {}".format(hms.task_status, hms.task_id))
    print("{}".format(hms.data))
    print("Runtime: {} sec".format(round(t1-t0, 4)))
