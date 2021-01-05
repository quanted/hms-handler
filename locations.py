from hms_handler import load_file
from data_analysis import get_comid_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import datetime
import time
import csv

plt.style.use('seaborn-white')

start_date = datetime.datetime(2000, 1, 1)
end_date = datetime.datetime(2017, 12, 31)


def write_csv(filename, labels, data):
    with open(filename, 'w', newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(labels)
        csvwriter.writerows(data)


def get_catchments(locations=None, category="division"):
    if locations is None:
        catchment_file = "catchments-completed-list.csv"
        locations = load_file(catchment_file, c_dict=True)
    result = {}
    for l in locations:
        if l[category] is not "":
            if str(l[category]) in result.keys():
                result[str(l[category])].append(l)
            else:
                result[str(l[category])] = [l]
        else:
            if 'UA' in result.keys():
                result['UA'].append(l)
            else:
                result['UA'] = [l]
    return result


def calculate_nse(qm: pd.Series, qo: pd.Series, normalize=False) -> float:
    """
    https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    :param qm: Modelled data
    :param qo: Observed data
    :param normalize: normalize the NSE value for NNSE
    :return: NSE
    """
    qo_mean = qo.mean()
    nse = 1.0 - (qm.sub(qo).pow(2.0).sum())/((qo - qo_mean).pow(2.0).sum())
    if normalize:
        nse = 1.0 / (2.0 - nse)
    return float(nse)


class Region:
    def __init__(self, type, name):
        self.type = type
        self.name = name
        self.catchments = {}
        self.nldas = None
        self.cn = None
        self.gldas = None
        self.gcn = None
        self.dif = None
        self.gdif = None
        self.pearsons, self.p = None, None
        self.gpearsons, self.gp = None, None
        self.dif_mean = None
        self.monthly = None
        self.pearsons_m, self.p_m = None, None
        self.gpearsons_m, self.gp_m = None, None
        self.yearly = None
        self.pearsons_y, self.p_y = None, None
        self.gpearsons_y, self.gp_y = None, None
        self.n25, self.g25 = None, None
        self.n1, self.g1 = None, None
        self.std, self.variance, self.mean = None, None, None

    def get_info(self):
        return [
            [
                "nldas_pearsons", "nldas_monthly_pearsons", "nldas_yearly_pearsons", "nldas_n0_pearsons", "nldas_25p_pearsons", "nldas_mean", "nldas_std", "nldas_variance",
                "gldas_pearsons", "gldas_monthly_pearsons", "gldas_yearly_pearsons", "gldas_n0_pearsons", "gldas_25p_pearsons", "gldas_mean", "gldas_std", "gldas_variance"
            ],
            [
                self.pearsons, self.pearsons_m, self.pearsons_y, self.n1, self.n25, self.mean["nldas"], self.std["nldas"], self.variance["nldas"],
                self.gpearsons, self.gpearsons_m, self.gpearsons_y, self.g1, self.g25, self.mean["gldas"], self.std["gldas"], self.variance["gldas"]
            ]
        ]

    def print_info(self):
        print("------------ REGION ----------------")
        print("Type: {}, Name: {}".format(self.type, self.name))
        print("Number of catchments: {}".format(len(self.catchments)))
        print("Start Date: {}, End Date: {}".format(self.nldas["date"].iloc[0], self.nldas["date"].iloc[-1]))
        print("Pearson's Coefficient: {}, P-Value: {}".format(self.pearsons, self.p))
        print("NLDAS Average Non-Zero Pearson's Coefficient: {}".format(self.n1))
        print("NLDAS Average 25 percentile Pearson's Coefficient: {}".format(self.n25))
        print("GLDAS Average Non-Zero Pearson's Coefficient: {}".format(self.g1))
        print("GLDAS Average 25 percentile Pearson's Coefficient: {}".format(self.g25))
        print("NLDAS STD: {}, GLDAS STD: {}".format(self.std["nldas"], self.std["gldas"]))
        print("NLDAS Variance: {}, GLDAS Variance: {}".format(self.variance["nldas"], self.variance["gldas"]))

        for c, d in self.catchments.items():
            print("------------ Catchment: {} ----------------".format(c))
            d.print_info()

    def set_catchments(self, locations):
        for l in locations:
            self.catchments[l['comid']] = Catchment(l['comid'], l['division'], l['province'], l['section'])
        self.combine_data()

    def combine_data(self):
        for l, d in self.catchments.items():
            if self.monthly is None:
                self.monthly = pd.DataFrame(d.monthly_sum[["date", "nldas", "cn", "dif", "gldas", "gcn", "gdif"]])
            else:
                self.monthly["dif"] = (d.monthly_sum["dif"] + self.monthly["dif"]) / 2.0
                self.monthly["nldas"] = (d.monthly_sum["nldas"] + self.monthly["nldas"]) / 2.0
                self.monthly["cn"] = (d.monthly_sum["cn"] + self.monthly["cn"]) / 2.0
                self.monthly["gdif"] = (d.monthly_sum["gdif"] + self.monthly["gdif"]) / 2.0
                self.monthly["gldas"] = (d.monthly_sum["gldas"] + self.monthly["gldas"]) / 2.0
                self.monthly["gcn"] = (d.monthly_sum["gcn"] + self.monthly["gcn"]) / 2.0
            if self.yearly is None:
                self.yearly = pd.DataFrame(d.yearly_sum[["date", "nldas", "cn", "dif", "gldas", "gcn", "gdif"]])
            else:
                self.yearly["dif"] = (d.yearly_sum["dif"] + self.yearly["dif"]) / 2.0
                self.yearly["nldas"] = (d.yearly_sum["nldas"] + self.yearly["nldas"]) / 2.0
                self.yearly["cn"] = (d.yearly_sum["cn"] + self.yearly["cn"]) / 2.0
                self.yearly["gdif"] = (d.yearly_sum["gdif"] + self.yearly["gdif"]) / 2.0
                self.yearly["gldas"] = (d.yearly_sum["gldas"] + self.yearly["gldas"]) / 2.0
                self.yearly["gcn"] = (d.yearly_sum["gcn"] + self.yearly["gcn"]) / 2.0
            if self.nldas is None:
                self.nldas = pd.DataFrame(d.timeseries[["date", "nldas"]])
                self.nldas = self.nldas.rename(columns={'nldas': str(l)})
                self.nldas["date"] = pd.to_datetime(self.nldas["date"])
            else:
                self.nldas[str(l)] = d.timeseries["nldas"]
            if self.cn is None:
                self.cn = pd.DataFrame(d.timeseries[["date", "cn"]])
                self.cn = self.cn.rename(columns={'cn': str(l)})
                self.cn["date"] = pd.to_datetime(self.cn["date"])
            else:
                self.cn[str(l)] = d.timeseries["cn"]
            if self.dif is None:
                self.dif = pd.DataFrame(d.timeseries[["date", "dif"]])
                self.dif = self.dif.rename(columns={'dif': str(l)})
                self.dif["date"] = pd.to_datetime(self.dif["date"])
            else:
                self.dif[str(l)] = d.timeseries["dif"]
            if self.gldas is None:
                self.gldas = pd.DataFrame(d.timeseries[["date", "gldas"]])
                self.gldas = self.gldas.rename(columns={'gldas': str(l)})
                self.gldas["date"] = pd.to_datetime(self.gldas["date"])
            else:
                self.gldas[str(l)] = d.timeseries["gldas"]
            if self.gcn is None:
                self.gcn = pd.DataFrame(d.timeseries[["date", "gcn"]])
                self.gcn = self.gcn.rename(columns={'gcn': str(l)})
                self.gcn["date"] = pd.to_datetime(self.cn["date"])
            else:
                self.gcn[str(l)] = d.timeseries["gcn"]
            if self.gdif is None:
                self.gdif = pd.DataFrame(d.timeseries[["date", "gdif"]])
                self.gdif = self.gdif.rename(columns={'gdif': str(l)})
                self.gdif["date"] = pd.to_datetime(self.dif["date"])
            else:
                self.gdif[str(l)] = d.timeseries["gdif"]

            if not np.isnan(d.q1_np[0]):
                if self.n1 is None:
                    self.n1 = d.q1_np[0]
                else:
                    self.n1 = (self.n1 + d.q1_np[0]) / 2.0
            if not np.isnan(d.q1_gp[0]):
                if self.g1 is None:
                    self.g1 = d.q1_gp[0]
                else:
                    self.g1 = (self.g1 + d.q1_gp[0]) / 2.0
            if not np.isnan(d.q25_np[0]):
                if self.n25 is None:
                    self.n25 = d.q25_np[0]
                else:
                    self.n25 = (self.n25 + d.q25_np[0]) / 2.0
            if not np.isnan(d.q25_gp[0]):
                if self.g25 is None:
                    self.g25 = d.q25_gp[0]
                else:
                    self.g25 = (self.g25 + d.q25_gp[0]) / 2.0
            if not np.isnan(d.pearsons):
                if self.pearsons is None:
                    self.pearsons = d.pearsons
                else:
                    self.pearsons = (self.pearsons + d.pearsons) / 2.0
            if not np.isnan(d.gpearsons):
                if self.gpearsons is None:
                    self.gpearsons = d.gpearsons
                else:
                    self.gpearsons = (self.gpearsons + d.gpearsons) / 2.0

            if not np.isnan(d.pearsons_m):
                if self.pearsons_m is None:
                    self.pearsons_m = d.pearsons_m
                else:
                    self.pearsons_m = (self.pearsons_m + d.pearsons_m) / 2.0
            if not np.isnan(d.gpearsons_m):
                if self.gpearsons_m is None:
                    self.gpearsons_m = d.gpearsons_m
                else:
                    self.gpearsons_m = (self.gpearsons_m + d.gpearsons_m) / 2.0

            if not np.isnan(d.pearsons_year):
                if self.pearsons_y is None:
                    self.pearsons_y = d.pearsons_year
                else:
                    self.pearsons_y = (self.pearsons_y + d.pearsons_year) / 2.0
            if not np.isnan(d.gpearsons_year):
                if self.gpearsons_y is None:
                    self.gpearsons_y = d.gpearsons_year
                else:
                    self.gpearsons_y = (self.gpearsons_y + d.gpearsons_year) / 2.0
            if not np.isnan(d.p):
                if self.p is None:
                    self.p = d.p
                else:
                    self.p = (self.p + d.p) / 2.0
            # d.plot(timeseries=False, monthly=True)
        self.dif_mean = pd.DataFrame(self.dif.mean(axis=1), columns=["dif_mean"])
        self.dif_mean.reset_index(inplace=True)
        self.dif_mean['date'] = pd.to_datetime(self.dif["date"])
        self.calculate_stats()

    def plot_data(self, df, title):
        col = list(df.columns)
        if "index" in col:
            col.remove("index")
        if "date" in col:
            col.remove("date")
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        l1 = ":"
        lw = 1
        if "nldas" in col[0] or "cn" in col[0]:
            l1 = "-"
            lw = 1.2
        ax = df.plot(x='date', y=col[0], kind='line', linestyle=l1, linewidth=lw, label=col[0], color=colors[0], figsize=(12, 8))
        for i in range(1, len(col)):
            l1 = ":"
            lw = 1
            if "nldas" in col[i] or "cn" in col[i]:
                l1 = "-"
                lw = 1.2
            df.plot(x='date', y=col[i], kind='line', linestyle=l1, linewidth=lw, label=col[i], ax=ax, color=colors[i%len(colors)])
        ax.set_title("{} - {} : {}".format(self.type, self.name, title))
        plt.show()

    def calculate_stats(self):
        n = []
        g = []
        for c, d in self.catchments.items():
            if not np.isnan(d.pearsons):
                n.append(d.pearsons)
            if not np.isnan(d.gpearsons):
                g.append(d.gpearsons)
        n_std = np.std(n)
        g_std = np.std(g)
        n_var = np.var(n)
        g_var = np.var(g)
        self.std = {
            "nldas": n_std,
            "gldas": g_std
        }
        self.variance = {
            "nldas": n_var,
            "gldas": g_var
        }
        self.mean = {
            "nldas": np.mean(n),
            "gldas": np.mean(g)
        }

class Catchment:
    def __init__(self, comid, division, province, section=None):
        self.comid = int(comid)
        self.division = division
        self.province = province
        self.section = section if section else ""
        self.timeseries = get_comid_data(self.comid)
        self.timeseries["date"] = pd.to_datetime(self.timeseries["date"])
        date_mask = (self.timeseries["date"] >= start_date) & (self.timeseries["date"] <= end_date)
        self.timeseries = self.timeseries.loc[date_mask]
        self.quantiles_cn = self.timeseries[self.timeseries.cn > 0.0].quantile(q=0.25, axis=0, interpolation='linear')
        self.quantiles_gcn = self.timeseries[self.timeseries.gcn > 0.0].quantile(q=0.25, axis=0, interpolation='linear')

        self.yearly_sum = self.timeseries[['date', 'nldas', 'cn', 'gldas', 'gcn']].groupby(self.timeseries["date"].dt.year).sum()
        self.yearly_sum['dif'] = self.yearly_sum['nldas'].sub(self.yearly_sum['cn'], axis=0)
        self.yearly_sum['gdif'] = self.yearly_sum['gldas'].sub(self.yearly_sum['gcn'], axis=0)
        self.yearly_sum.reset_index(inplace=True)
        self.yearly_sum = self.yearly_sum.rename(columns={'index': 'date'})

        self.monthly_sum = self.timeseries[['date', 'nldas', 'cn', 'gldas', 'gcn']].groupby([self.timeseries["date"].dt.year.rename('year'), self.timeseries["date"].dt.month.rename('month')]).sum()
        self.monthly_sum['dif'] = self.monthly_sum['nldas'].sub(self.monthly_sum['cn'], axis=0)
        self.monthly_sum['gdif'] = self.monthly_sum['gldas'].sub(self.monthly_sum['gcn'], axis=0)

        self.monthly_sum.reset_index(inplace=True)
        self.monthly_sum["date"] = pd.to_datetime((self.monthly_sum[["year", "month"]].assign(DAY=1)))
        self.monthly_sum = self.monthly_sum.drop(['year', 'month'], axis=1)
        self.monthly_sum = self.monthly_sum.rename(columns={'index': 'date'})
        self.pearsons, self.p = scipy.stats.pearsonr(self.timeseries["nldas"], self.timeseries["cn"])
        self.pearsons_m, self.p_m = scipy.stats.pearsonr(self.monthly_sum["nldas"], self.monthly_sum["cn"])
        self.pearsons_year, self.p_year = scipy.stats.pearsonr(self.yearly_sum["nldas"], self.yearly_sum["cn"])
        self.gpearsons, self.gp = scipy.stats.pearsonr(self.timeseries["gldas"], self.timeseries["gcn"])
        self.gpearsons_m, self.gp_m = scipy.stats.pearsonr(self.monthly_sum["gldas"], self.monthly_sum["gcn"])
        self.gpearsons_year, self.gp_year = scipy.stats.pearsonr(self.yearly_sum["gldas"], self.yearly_sum["gcn"])

        self.q1_n = self.timeseries[self.timeseries.cn > 0.0]
        self.q1_n.reset_index(inplace=True)
        self.q1_n = self.q1_n.drop(["gldas", "gcn", "gdif"], axis=1)
        if self.q1_n.shape[0] > 1:
            self.q1_np = scipy.stats.pearsonr(self.q1_n["nldas"], self.q1_n["cn"])
        else:
            self.q1_np = [np.nan, np.nan]
        self.q1_g = self.timeseries[self.timeseries.gcn > 0.0]
        self.q1_g.reset_index(inplace=True)
        self.q1_g = self.q1_g.drop(["nldas", "cn", "dif"], axis=1)
        if self.q1_g.shape[0] > 1:
            self.q1_gp = scipy.stats.pearsonr(self.q1_g["gldas"], self.q1_g["gcn"])
        else:
            self.q1_gp = [np.nan, np.nan]

        self.q25_n = self.timeseries[self.timeseries.cn > self.quantiles_cn.cn] # and self.timeseries.cn > self.quantiles.cn])
        self.q25_n.reset_index(inplace=True)
        self.q25_n = self.q25_n.drop(["gldas", "gcn", "gdif"], axis=1)
        if self.q25_n.shape[0] > 1:
            self.q25_np = scipy.stats.pearsonr(self.q25_n["nldas"], self.q25_n["cn"])
        else:
            self.q25_np = [np.nan, np.nan]
        self.q25_g = self.timeseries[self.timeseries.gcn > self.quantiles_gcn.gcn] # and self.timeseries.gcn > self.quantiles.gcn])
        self.q25_g = self.q25_g.drop(["nldas", "cn", "dif"], axis=1)
        self.q25_g.reset_index(inplace=True)
        if self.q25_g.shape[0] > 1:
            self.q25_gp = scipy.stats.pearsonr(self.q25_g["gldas"], self.q25_g["gcn"])
        else:
            self.q25_gp = [np.nan, np.nan]

    def get_info(self):
        n_monthly_pc = self.pearsons_m
        n_monthly_nse = calculate_nse(self.monthly_sum["cn"], self.monthly_sum["nldas"])
        n_yearly_pc = self.pearsons_year
        n_yearly_nse = calculate_nse(self.yearly_sum["cn"], self.yearly_sum["nldas"])

        g_monthly_pc = self.gpearsons_m
        g_monthly_nse = calculate_nse(self.monthly_sum["gcn"], self.monthly_sum["gldas"])
        g_yearly_pc = self.gpearsons_year
        g_yearly_nse = calculate_nse(self.yearly_sum["gcn"], self.yearly_sum["gldas"])

        n_pc = self.pearsons
        n_nse = calculate_nse(self.timeseries["cn"], self.timeseries["nldas"])
        n_n0_pc = self.q1_np[0]
        n_n25_pc = self.q25_np[0]

        g_pc = self.gpearsons
        g_nse = calculate_nse(self.timeseries["gcn"], self.timeseries["gldas"])
        g_n0_pc = self.q1_gp[0]
        g_n25_pc = self.q25_gp[0]
        return [["nldas_pearsons", "nldas_nse", "nldas_nonzero_pearsons", "nldas_25p_pearsons", "nldas_monthly_pearsons",
                 "nldas_monthly_nse", "nldas_yearly_pearsons", "nldas_yearly_nse",
                 "gldas_pearsons", "gldas_nse", "gldas_nonzero_pearsons", "gldas_25p_pearsons", "gldas_monthly_pearsons",
                 "gldas_monthly_nse", "gldas_yearly_pearsons", "gldas_yearly_nse"],
                [n_pc, n_nse, n_n0_pc, n_n25_pc, n_monthly_pc, n_monthly_nse, n_yearly_pc, n_yearly_nse,
                g_pc, g_nse, g_n0_pc, g_n25_pc, g_monthly_pc, g_monthly_nse, g_yearly_pc, g_yearly_nse]
                ]

    def print_info(self):
        print("COMID: {}, Physiographic Division: {}, Province: {}, Section: {}".format(
            self.comid, self.division, self.province, self.section)
        )
        print("Start Date: {}, End Date: {}".format(self.timeseries["date"].iloc[0], self.timeseries["date"].iloc[-1]))
        print("NLDAS - Pearson's Coefficient: {}, NSE: {}".format(self.pearsons, calculate_nse(self.timeseries["cn"], self.timeseries["nldas"])))
        print("NLDAS - Non-Zero Pearson's Coefficient: {}, P: {}".format(self.q1_np[0], self.q1_np[1]))
        print("NLDAS - 25% Pearson's Coefficient: {}, P: {}".format(self.q25_np[0], self.q25_np[1]))
        print("GLDAS - Pearson's Coefficient: {}, NSE: {}".format(self.gpearsons, calculate_nse(self.timeseries["gcn"], self.timeseries["gldas"])))
        print("GLDAS - Non-Zero Pearson's Coefficient: {}, P: {}".format(self.q1_gp[0], self.q1_gp[1]))
        print("GLDAS - 25% Pearson's Coefficient: {}, P: {}".format(self.q25_gp[0], self.q25_gp[1]))
        print("Monthly Pearson's Coefficient: {}, Monthly NSE: {}".format(self.pearsons_m, calculate_nse(self.monthly_sum["cn"], self.monthly_sum["nldas"])))
        print("Yearly Pearson's Coefficient: {}, Yearly NSE: {}".format(self.pearsons_year, calculate_nse(self.yearly_sum["cn"], self.yearly_sum["nldas"])))

    def plot(self, timeseries=True, yearly=False, monthly=False):
        if timeseries:
            ax = self.timeseries.plot(y='nldas', x='date', kind='line', linewidth=0.75, label='nldas', color='b', figsize=(12, 8))
            self.timeseries.plot(y='cn', x='date', kind='line', linewidth=0.75, label='cn', color='g', ax=ax)
            self.timeseries.plot(y='gldas', x='date', kind='line', linewidth=0.75, label='gldas', color='c', ax=ax)
            self.timeseries.plot(y='gcn', x='date', kind='line', linewidth=0.75, label='gcn', color='m', ax=ax)
            self.timeseries.plot(y='dif', x='date', kind='line', linestyle=":", linewidth=1.5, label='dif', color='r', ax=ax)
            self.timeseries.plot(y='gdif', x='date', kind='line', linestyle=":", linewidth=1.5, label='gdif', color='y', ax=ax)
            ax.set_ylabel("Runoff (mm)")
        if monthly:
            ax = self.monthly_sum.plot(y='nldas', x='date', kind='line', linewidth=0.75, label='nldas', color='b', figsize=(12, 8))
            self.monthly_sum.plot(y='cn', x='date', kind='line', linewidth=0.75, label='cn', color='g', ax=ax)
            self.monthly_sum.plot(y='gldas', x='date', kind='line', linewidth=0.75, label='gldas', color='c', ax=ax)
            self.monthly_sum.plot(y='gcn', x='date', kind='line', linewidth=0.75, label='gcn', color='m', ax=ax)
            self.monthly_sum.plot(y='dif', x='date', kind='line', linestyle=":", linewidth=1.5, label='dif', color='r', ax=ax)
            self.monthly_sum.plot(y='gdif', x='date', kind='line', linestyle=":", linewidth=1.5, label='gdif', color='y', ax=ax)
            ax.set_ylabel("Runoff (mm)")
            ax.set_title("{}, {}, {}: {}".format(self.division, self.province, self.section, self.comid))
        if yearly:
            ax = self.yearly_sum.plot(y='nldas', x='date', kind='line', linewidth=0.75, label='nldas', color='b', figsize=(12, 8))
            self.yearly_sum.plot(y='cn', x='date', kind='line', linewidth=0.75, label='cn', color='g', ax=ax)
            self.yearly_sum.plot(y='gldas', x='date', kind='line', linewidth=0.75, label='gldas', color='c', ax=ax)
            self.yearly_sum.plot(y='gcn', x='date', kind='line', linewidth=0.75, label='gcn', color='m', ax=ax)
            self.yearly_sum.plot(y='dif', x='date', kind='line', linestyle=":", linewidth=1.5, label='dif', color='r', ax=ax)
            self.yearly_sum.plot(y='gdif', x='date', kind='line', linestyle=":", linewidth=1.5, label='gdif', color='y', ax=ax)
            ax.set_ylabel("Runoff (mm)")
            ax.set_title("{}, {}, {}: {}".format(self.division, self.province, self.section, self.comid))
        plt.show()


if __name__ == "__main__":

    t0 = time.time()
    by_division = get_catchments(None, "division")
    by_province = get_catchments(None, "province")
    by_section = get_catchments(None, "section")

    phsio = {
        "section": by_section,
        "province": by_province,
        "division": by_division
    }
    for s, b in phsio.items():
        n = len(b)
        catchment_labels = []
        catchments_data = []
        section_labels = []
        section_data = []
        i = 1
        for k, v in b.items():
            r = Region(s, k)
            r.set_catchments(b[k])
            s_info = r.get_info()
            for c, d in r.catchments.items():
                info = d.get_info()
                if type(k) == str:
                    info[1].insert(0, v[0]["section"])
                else:
                    info[1].insert(0, "NA")
                info[1].insert(0, v[0]["province"])
                info[1].insert(0, v[0]["division"])
                info[1].insert(0, c)
                if len(catchment_labels) == 0:
                    info[0].insert(0, "section")
                    info[0].insert(0, "province")
                    info[0].insert(0, 'division')
                    info[0].insert(0, "comid")
                    catchment_labels = info[0]
                catchments_data.append(info[1])
            s_info[1].insert(0, k)
            s_info[1].insert(0, v[0]["province"])
            s_info[1].insert(0, v[0]["division"])
            if len(section_labels) == 0:
                s_info[0].insert(0, "section")
                s_info[0].insert(0, "province")
                s_info[0].insert(0, 'division')
                section_labels = s_info[0]
            section_data.append(s_info[1])
            print("section: {}/{}".format(i, n))
            r.print_info()
            i += 1
        #write_csv("catchment-stats-data.csv", catchment_labels, catchments_data)
        write_csv("{}-stats-data.csv".format(s), section_labels, section_data)
        print("-----------------------------------\n-----------------------------------")
    t1 = time.time()
    print("Processing time: {} sec".format(round(t1-t0, 4)))

    # p_name = "WHITE MOUNTAIN"
    # r = Region(p_type, p_name)
    # r.set_catchments(by_section[p_name])
    # r.print_info()
    # r.plot_data(r.monthly, "Monthly Difference")
    # r.plot_data(r.yearly, "Yearly Difference")
    # r.plot_data(r.nldas, "NLDAS")
    # r.plot_data(r.cn, "CN")
    # r.plot_data(r.dif, "Difference")

    # comid = 8430816
    # division, province, section = "INTERMONTANE PLATEAUS", "BASIN AND RANGE", "SACRAMENTO"
    # catchment = Catchment(comid, division, province, section)
    # catchment.print_info()

