import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from locations import get_catchments

plt.style.use('seaborn-white')


def plot_catchment_data(data, y="nldas_pearsons", x="section", save_legend=False):
    sections = get_catchments(category="section")
#     # x-axis = section name
#     # y-axis = pearson's coefficient
    labels = set(data[x])
    if np.nan in labels:
        labels.remove(np.nan)
        labels.add("NA")
        data[x] = data[x].fillna("NA")
    labels_i = [i for i in range(1, len(labels)+1)]
    if save_legend:
        with open("plot-legend.csv", 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(labels_i)
            csvwriter.writerow(labels)
    if type(y) is list:
        colors = ['b', 'm', 'g', 'c', 'y', 'k']
        ax = data.plot.scatter(y=y[0], x=x, marker='.', s=7, label=y[0], color=colors[0], figsize=(16, 8))
        for i in range(1, len(y)):
            data.plot.scatter(y=y[i], x=x, marker='.', s=7, label=y[i], color=colors[i], ax=ax)
    else:
        ax = data.plot.scatter(y=y, x=x, marker='.', s=7, label=y, color='b', figsize=(16, 8))
        ax.set_ylabel(y)
    ax.set_xticklabels(labels_i)
    plt.axhline(y=1.0, color='r', linewidth=0.5)
    plt.axhline(y=0.0, color='r', linewidth=0.25)
    plt.axhline(y=-1.0, color='r', linewidth=0.25)
    ax.set_title("{} - {}".format(y, x))
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # data columns
    # comid,division,province,section,
    # nldas_pearsons,nldas_nse,nldas_nonzero_pearsons,nldas_25p_pearsons,
    # nldas_monthly_pearsons,nldas_monthly_nse,nldas_yearly_pearsons,nldas_yearly_nse,
    # gldas_pearsons,gldas_nse,gldas_nonzero_pearsons,gldas_25p_pearsons,
    # gldas_monthly_pearsons,gldas_monthly_nse,gldas_yearly_pearsons,gldas_yearly_nse

    filename = "catchment-stats-data.csv"
    data = pd.read_csv(filename)
    plot_catchment_data(data, y=["nldas_pearsons", "nldas_monthly_pearsons", "nldas_yearly_pearsons"], save_legend=True)
    plot_catchment_data(data, y=["gldas_pearsons", "gldas_monthly_pearsons", "gldas_yearly_pearsons"])
    plot_catchment_data(data, y=["nldas_pearsons", "nldas_nonzero_pearsons", "nldas_25p_pearsons"])
    plot_catchment_data(data, y=["gldas_pearsons", "gldas_nonzero_pearsons", "gldas_25p_pearsons"])
