import os
import time
import json
import csv
import sys
import sqlite3
import argparse
import logging
import copy
import multiprocessing as mp
from hms import HMS

# print("Number of processors: {}".format(mp.cpu_count()))

logger = logging.getLogger(__name__)
db_path = "runoff_data.sqlite3"
cookies = {'sessionid': 'lmufmudjybph2r3ju0la15x5vuovz1pw'}
# cookies = {'sessionid': 'b5c5ev7usauevf2nro7e8mothmekqsnj'}


def get_db_connection():
    """
    Connect to sqlite database located at db_path
    :return: sqlite connection
    """
    conn = sqlite3.connect(db_path)
    return conn


def get_failed():
    conn = get_db_connection()
    c = conn.cursor()
    query = "SELECT Status.comid, Catchment.division, Catchment.province, Catchment.section FROM Status INNER JOIN Catchment ON Catchment.comid=Status.comid WHERE cn=? OR nldas=? OR cng=? OR gldas=?"
    c.execute(query, ("FAILED", "FAILED", "FAILED", "FAILED",))
    locations = []
    for i in c.fetchall():
        l = {
            'comid': i[0],
            'division': str(i[1]),
            'province': str(i[2]),
            'section': str(i[3])
        }
        locations.append(l)
    c.close()
    conn.close()
    return locations


def add_data(comid, data, source):
    s_nldas = None if source is not "nldas" else "SUCCESS"
    s_cn = None if source is not "cn" else "SUCCESS"
    s_gldas = None if source is not "gldas" else "SUCCESS"
    s_cng = None if source is not "cng" else "SUCCESS"
    metadata_tables = ["NLDASMetadata", "GLDASMetadata", "CNMetadata", "CNGMetadata"]
    metadata_table = list(filter(lambda m: source in m.lower(), metadata_tables))[0]
    conn = get_db_connection()
    conn.isolation_level = None
    c = conn.cursor()
    if c.execute("SELECT comid FROM Status WHERE comid=?", (int(comid),)).fetchone():
        query = "UPDATE Status SET {}='{}' WHERE comid={}".format(source, "SUCCESS", int(comid))
        m_query = "INSERT OR REPLACE INTO {}(comid, metadata) VALUES(?,?)".format(metadata_table)
        m_values = (comid, json.dumps(data["metadata"]),)
    else:
        values = (int(comid), str(s_nldas), str(s_cn), str(s_gldas), str(s_cng))
        query = "INSERT INTO Status VALUES{}".format(tuple(values))
        m_query = "INSERT OR REPLACE INTO {}(comid, metadata) VALUES(?,?)".format(metadata_table)
        m_values = (comid, json.dumps(data["metadata"]),)
    c.execute(query)
    c.execute(m_query, m_values)
    data_tables = ["RunoffNLDAS", "RunoffGLDAS", "RunoffCN", "RunoffCNG"]
    d_table = list(filter(lambda d: source in d.lower(), data_tables))[0]
    c.execute("BEGIN")
    i = 0
    i_max = 400
    for k, d in data["data"].items():
        d_values = (int(comid), str(k), float(d[0]),)
        data_query = "INSERT OR REPLACE INTO {}(comid, date, value) VALUES(?,?,?)".format(d_table)
        c.execute(data_query, d_values)
        i += 1
        if i >= i_max:
            c.execute("COMMIT")
            c.execute("BEGIN")
            i = 0
    c.execute("COMMIT")
    conn.close()


def load_file(filepath, c_dict=False):
    logger.info("Checking for file: {}".format(filepath))
    if os.path.exists(filepath):

        with open(filepath, newline='') as csvfile:
            logger.info("Loading file: {}".format(filepath))
            if c_dict:
                data = []
                csv_dict = csv.DictReader(csvfile)
                for row in csv_dict:
                    data.append(row)
            else:
                data = []
                csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csv_reader:
                    data.append(row)
        return data
    else:
        logger.fatal("File {} not found".format(filepath))
        sys.exit()


def save_location(comid, division, province, section):
    conn = get_db_connection()
    c = conn.cursor()
    query = "SELECT comid FROM Catchment WHERE comid=?"
    current = c.execute(query, (comid,)).fetchone()
    if not current:
        values = (int(comid), division, province, section,)
        query = "INSERT INTO Catchment (comid, division, province, section) VALUES(?,?,?,?)"
        c.execute(query, values)
    conn.commit()


def get_status(locations):
    location_status = {}
    comid_list = []
    location_details = {}
    for l in locations:
        comid_list.append(int(l['comid']))
        location_details[int(l['comid'])] = {
            'division': str(l['division']) if l['division'] is not None else "-1",
            'province': str(l['province']) if l['province'] is not None else "-1",
            'section': str(l['section']) if l['section'] is not None else "-1"
        }
        location_status[int(l['comid'])] = {'nldas': None, 'cn': None, 'gldas': None, 'cng': None}
    conn = get_db_connection()
    c = conn.cursor()
    if len(comid_list) > 1:
        query = "SELECT comid, nldas, cn, gldas, cng FROM Status WHERE comid IN {}".format(tuple(comid_list))
    else:
        query = "SELECT comid, nldas, cn, gldas, cng FROM Status WHERE comid={}".format(comid_list[0])
    for r in c.execute(query):
        location_status[int(r[0])] = {'nldas': r[1], 'cn': r[2], 'gldas': r[3], 'cng': r[4]}
    conn.close()
    return location_status, location_details


def get_comid_data(comid, configs, loc_status):
    nldas = None
    cn = None
    gldas = None
    cng = None
    nldas_status = None
    cn_status = None
    gldas_status = None
    cng_status = None
    if loc_status["nldas"] == "SUCCESS" and loc_status["cn"] == "SUCCESS" and loc_status["gldas"] == "SUCCESS" and loc_status["cng"] == "SUCCESS":
        return None, None, None, None
    if loc_status["nldas"] != "SUCCESS":
        t0 = time.time()
        hms_a = HMS(start_date=configs["startdate"],
                  end_date=configs["enddate"],
                  source=configs["source"],
                  dataset=configs["dataset"],
                  module=configs["module"],
                  cookies=cookies)
        hms_a.set_geometry("comid", comid)
        hms_a.submit_request()
        if hms_a.task_status == "SUCCESS":
            nldas = hms_a
            nldas_status = "SUCCESS"
        t1 = time.time()
        hms_a.print_info()
        print("Time: {} sec".format(round(t1 - t0, 4)))
    if loc_status["gldas"] != "SUCCESS":
        t0 = time.time()
        hms_a = HMS(start_date=configs["startdate"],
                  end_date=configs["enddate"],
                  source="gldas",
                  dataset=configs["dataset"],
                  module=configs["module"],
                  cookies=cookies)
        hms_a.set_geometry("comid", comid)
        hms_a.submit_request()
        if hms_a.task_status == "SUCCESS":
            gldas = hms_a
            gldas_status = "SUCCESS"
        t1 = time.time()
        hms_a.print_info()
        print("Time: {} sec".format(round(t1 - t0, 4)))
    if loc_status["cn"] != "SUCCESS":
        t0 = time.time()
        hms_b = HMS(start_date=configs["startdate"],
                  end_date=configs["enddate"],
                  source='curvenumber',
                  dataset=configs["dataset"],
                  module=configs["module"],
                  cookies=cookies)
        hms_b.set_geometry("comid", comid, metadata={"precipSource": "nldas"})
        hms_b.submit_request()
        if hms_b.task_status == "SUCCESS":
            cn = hms_b
            cn_status = "SUCCESS"
        t1 = time.time()
        hms_b.print_info()
        print("Time: {} sec".format(round(t1 - t0, 4)))
    if loc_status["cng"] != "SUCCESS":
        t0 = time.time()
        hms_b = HMS(start_date=configs["startdate"],
                  end_date=configs["enddate"],
                  source='curvenumber',
                  dataset=configs["dataset"],
                  module=configs["module"],
                  cookies=cookies)
        hms_b.set_geometry("comid", comid, metadata={"precipSource": "gldas"})
        hms_b.submit_request()
        if hms_b.task_status == "SUCCESS":
            cng = hms_b
            cng_status = "SUCCESS"
        t1 = time.time()
        hms_b.print_info()
        print("Time: {} sec".format(round(t1 - t0, 4)))
    return {
        "status": [nldas_status, cn_status, gldas_status, cng_status],
        "data": [nldas, cn, gldas, cng],
        "comid": comid
    }


def get_n_locations(n, locations):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT comid FROM Status WHERE nldas=? AND cn=? AND gldas=? AND cng=?", ("SUCCESS", "SUCCESS", "SUCCESS", "SUCCESS",))
    completed = [int(i[0]) for i in list(c.fetchall())]
    c.close()
    todo_locations = []
    check_completed = set()
    for l in locations:
        if int(l["comid"]) not in completed:
            todo_locations.append(l)
        check_completed.add(int(l["comid"]))

    locations = todo_locations

    loc_status, details = get_status(locations)
    c_locations = []
    c_status = {}
    c_details = {}
    for l in loc_status.keys():
        l = int(l)
        if loc_status[l]["nldas"] == "SUCCESS" and loc_status[l]["cn"] == "SUCCESS" and loc_status[l]["gldas"] == "SUCCESS" and loc_status[l]["cng"] == "SUCCESS":
            continue
        else:
            c_locations.append(l)
            c_status[l] = loc_status[l]
            c_details[l] = details[l]
            if len(c_locations) == n:
                break
    new_locations = []
    for l in locations:
        if int(l["comid"]) not in c_locations:
            new_locations.append(l)
    return new_locations, c_locations, c_status, c_details


def download_parallel(locations, configs, failed=False):
    if failed:
        locations = get_failed()
    if len(locations) == 0:
        return
    n = 1
    pool = mp.Pool(n)

    locations, comids, n_status, n_details = get_n_locations(n, locations)
    print("Remaining Locations: {}".format(len(locations)))
    args = list(zip(comids, [configs]*len(comids), n_status.values()))
    results = pool.starmap_async(get_comid_data, [a for a in args]).get()
    pool.close()
    for r in results:
        if (all(r["status"]) is None) or \
                (n_status[r["comid"]]["nldas"] is None and r["status"][0] is None) or \
                (n_status[r["comid"]]["cn"] is None and r["status"][1] is None) or \
                (n_status[r["comid"]]["gldas"] is None and r["status"][2] is None) or \
                (n_status[r["comid"]]["cng"] is None and r["status"][3] is None):
            locations.append({
                'comid': r[4],
                'division': n_details[r[4]]['division'],
                'province': n_details[r[4]]['province'],
                'section': n_details[r[4]]['section']
            })
            continue
        comid = int(r["comid"])
        if not failed:
            save_location(comid, n_details[comid]['division'], n_details[comid]['province'], n_details[comid]['section'])
        if r["status"][0]:
            add_data(comid, json.loads(r["data"][0].data), "nldas")
        if r["status"][1]:
            add_data(comid, json.loads(r["data"][1].data), "cn")
        if r["status"][2]:
            add_data(comid, json.loads(r["data"][2].data), "gldas")
        if r["status"][3]:
            add_data(comid, json.loads(r["data"][3].data), "cng")
    download_parallel(locations, configs)


def download_data(locations, configs):
    loc_status, details = get_status(locations)
    download_complete = False
    locs_todo = list(loc_status.keys())
    max_tries = 2
    i_tries = 0
    while not download_complete:
        new_locs_todo = []
        for l in locs_todo:
            save_location(l, details[l]['division'], details[l]['province'], details[l]['section'])
            should_retry = False
            if loc_status[l]["nldas"] == "SUCCESS" and loc_status[l]["cn"] == "SUCCESS":
                continue
            elif loc_status[l]["nldas"] != "SUCCESS":
                t0 = time.time()
                hms = HMS(start_date=configs["startdate"],
                          end_date=configs["enddate"],
                          source=configs["source"],
                          dataset=configs["dataset"],
                          module=configs["module"],
                          cookies=cookies)
                hms.set_geometry("comid", l)
                hms.submit_request()
                if hms.task_status == "SUCCESS":
                    add_data(l, json.loads(hms.data), "nldas")
                    loc_status[l]["nldas"] = "SUCCESS"
                else:
                    should_retry = True
                t1 = time.time()
                print("COMID: {}, Success: {}, Source: {}, Time: {} sec".format(l, hms.task_status, "nldas", round(t1 - t0, 4)))
            if loc_status[l]["cn"] != "SUCCESS":
                t0 = time.time()
                hms = HMS(start_date=configs["startdate"],
                          end_date=configs["enddate"],
                          source='curvenumber',
                          dataset=configs["dataset"],
                          module=configs["module"],
                          cookies=cookies)
                hms.set_geometry("comid", l, metadata={"precipSource": "nldas"})
                hms.submit_request()
                if hms.task_status == "SUCCESS":
                    add_data(l, json.loads(hms.data), "cn")
                    loc_status[l]["cn"] = "SUCCESS"
                else:
                    should_retry = True
                t1 = time.time()
                print("COMID: {}, Success: {}, Source: {}, Time: {} sec".format(l, hms.task_status, "cn", round(t1 - t0, 4)))
            if should_retry:
                new_locs_todo.append(l)
        i_tries += 1
        if i_tries > max_tries:
            download_complete = True
        locs_todo = copy.copy(new_locs_todo)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="HMS Data downloader")
    # parser.add_argument("-l", dest="locs", required=True,
    #                     help="Path to file containing location data: comid, physiographic division and province")
    # parser.add_argument("-c", dest="configs", required=True,
    #                     help="Path to file containing request configuration data: start date, end date, dataset, "
    #                          "geometry")
    # args = parser.parse_args()
    # catchment_file = args.locs
    # configs_file = args.configs

    catchment_file = "catchments-list-cleaned.csv"
    configs_file = "configs.csv"

    catchments = load_file(catchment_file, c_dict=True)
    configs = load_file(configs_file, c_dict=True)[0]
    t0 = time.time()
    # download_data(catchments, configs)
    download_parallel(catchments, configs)
    t1 = time.time()
    print("Processed {} catchments".format(len(catchments)))
    print("Total runtime: {} sec".format(round(t1-t0, 4)))
