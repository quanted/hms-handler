from hms_handler import load_file
import sqlite3
import datetime
import copy
import csv

db_path = "runoff_data.sqlite3"
n = 6575

def get_db_connection():
    """
    Connect to sqlite database located at db_path
    :return: sqlite connection
    """
    conn = sqlite3.connect(db_path)
    return conn


def daily_sequence_check(time_series):
    i = 0
    if len(time_series) == 0:
        print("No data")
        return
    print("Start-Date: {}, End-Date: {}".format(time_series[0][1], time_series[-1][1]))
    for v in time_series:
        if i == 0:
            date0 = datetime.datetime.strptime(v[1], "%Y-%m-%d %H")
        else:
            date1 = datetime.datetime.strptime(v[1], "%Y-%m-%d %H")
            if (date0 + datetime.timedelta(days=1)) != date1:
                print("Daily mis-step, day0: {}, date1: {}".format(date0, date1))
            date0 = copy.copy(date1)
        i += 1


def check_all_data(locations, update=True):
    conn = get_db_connection()
    c = conn.cursor()
    uncompleteCN = []
    uncompleteNldas = []
    uncompleteCNG = []
    uncompleteGldas = []
    completed_list = []
    for l in locations:
        query = "SELECT * FROM RunoffNLDAS WHERE comid=?"
        c.execute(query, (int(l["comid"]),))
        nldas_data = list(c.fetchall())
        n_complete = False
        if len(nldas_data) < n:
            print("Uncompleted NLDAS request for COMID: {}, #: {}".format(l["comid"], len(nldas_data)))
            uncompleteNldas.append(int(l["comid"]))
            daily_sequence_check(nldas_data)
        else:
            n_complete = True
        query = "SELECT * FROM RunoffCN WHERE comid=?"
        c.execute(query, (int(l["comid"]),))
        cn_data = list(c.fetchall())
        c_complete = False
        if len(cn_data) < n:
            print("Uncompleted CN request for COMID: {}, #: {}".format(l["comid"], len(cn_data)))
            uncompleteCN.append(int(l["comid"]))
            daily_sequence_check(cn_data)
        else:
            c_complete = True
        query = "SELECT * FROM RunoffGLDAS WHERE comid=?"
        c.execute(query, (int(l["comid"]),))
        gldas_data = list(c.fetchall())
        g_complete = False
        if len(gldas_data) < n:
            print("Uncompleted GLDAS request for COMID: {}, #: {}".format(l["comid"], len(gldas_data)))
            uncompleteGldas.append(int(l["comid"]))
            daily_sequence_check(gldas_data)
        else:
            g_complete = True
        query = "SELECT * FROM RunoffCNG WHERE comid=?"
        c.execute(query, (int(l["comid"]),))
        cng_data = list(c.fetchall())
        cg_complete = False
        if len(cng_data) < n:
            print("Uncompleted CNG request for COMID: {}, #: {}".format(l["comid"], len(cng_data)))
            uncompleteCNG.append(int(l["comid"]))
            daily_sequence_check(cng_data)
        else:
            cg_complete = True
        if n_complete and c_complete and g_complete and cg_complete:
            completed_list.append(l)
    print("NLDAS uncompleted total: {}".format(len(uncompleteNldas)))
    print("CN uncompleted total: {}".format(len(uncompleteCN)))
    print("GLDAS uncompleted total: {}".format(len(uncompleteGldas)))
    print("GLDAS-CN uncompleted total: {}".format(len(uncompleteCNG)))
    if update:
        print("Updating database statuses for incomplete data.")
        for i in uncompleteNldas:
            query = "UPDATE Status SET nldas=? WHERE comid=?"
            c.execute(query, ("FAILED", i,))
            query = "DELETE FROM RunoffNLDAS WHERE comid=?"
            c.execute(query, (i,))
            print("Updated NLDAS status for COMID: {}".format(i))
        for i in uncompleteCN:
            query = "UPDATE Status SET cn=? WHERE comid=?"
            c.execute(query, ("FAILED", i,))
            query = "DELETE FROM RunoffCN WHERE comid=?"
            c.execute(query, (i,))
            print("Updated CN status for COMID: {}".format(i))
        for i in uncompleteGldas:
            query = "UPDATE Status SET gldas=? WHERE comid=?"
            c.execute(query, ("FAILED", i,))
            query = "DELETE FROM RunoffGLDAS WHERE comid=?"
            c.execute(query, (i,))
            print("Updated GLDAS status for COMID: {}".format(i))
        for i in uncompleteCNG:
            query = "UPDATE Status SET cng=? WHERE comid=?"
            c.execute(query, ("FAILED", i,))
            query = "DELETE FROM RunoffCNG WHERE comid=?"
            c.execute(query, (i,))
            print("Updated GLDAS-CN status for COMID: {}".format(i))
        conn.commit()
        with open("catchments-completed-list.csv", "w", newline='') as csvfile:
            fieldnames = ["comid", "division", "province", "section"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for l in completed_list:
                writer.writerow(l)
        


if __name__ == "__main__":
    catchment_file = "catchments-list-cleaned.csv"
    catchments = load_file(catchment_file, c_dict=True)
    check_all_data(catchments, update=True)
