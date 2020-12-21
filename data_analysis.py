import sqlite3
import pandas as pd
import json

db_path = "runoff_data.sqlite3"


def get_db_connection():
    """
    Connect to sqlite database located at db_path
    :return: sqlite connection
    """
    conn = sqlite3.connect(db_path)
    return conn


def get_comid_data(comid):
    conn = get_db_connection()
    c = conn.cursor()
    query_0 = "SELECT date, value FROM RunoffNLDAS WHERE comid=?"
    query_1 = "SELECT date, value FROM RunoffCN WHERE comid=?"
    query_2 = "SELECT date, value FROM RunoffGLDAS WHERE comid=?"
    query_3 = "SELECT date, value FROM RunoffCNG WHERE comid=?"
    c.execute(query_0, (comid,))
    nldas_data = c.fetchall()
    c.execute(query_1, (comid,))
    cn_data = c.fetchall()
    c.execute(query_2, (comid,))
    gldas_data = c.fetchall()
    c.execute(query_3, (comid,))
    gcn_data = c.fetchall()
    conn.close()
    n = min(len(nldas_data), len(cn_data), len(gldas_data), len(gcn_data))
    data = []
    for i in range(0, n):
        r = [nldas_data[i][0], nldas_data[i][1], cn_data[i][1], gldas_data[i][1], gcn_data[i][1]]
        data.append(r)
    df = pd.DataFrame(data=data, columns=["date", "nldas", "cn", "gldas", "gcn"])
    df["date"] = pd.to_datetime(df["date"])
    df["dif"] = df["nldas"] - df["cn"]
    df["gdif"] = df["gldas"] - df["gcn"]
    df = df.astype({"nldas": float, "cn": float, "dif": float, "gldas": float, "gcn": float, "gdif": float})
    return df


# def plot_catchment_data(data, column="nldas_pearsons"):
#     sections = get_catchments(category="section")
#     # x-axis = section name
#     # y-axis = pearson's coefficient





# if __name__ == "__main__":
#     filename = "catchment-stats-data.csv"
#     data = pd.read_csv(filename)
#     plot_catchment_data(data)

    # comid = 2674088
    # get_comid_data(comid)
