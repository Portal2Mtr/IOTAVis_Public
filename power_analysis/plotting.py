# Plots energy reading data from raspberry pi

import glob
import pandas as pd
import numpy as np
import statistics
import pickle
import matplotlib
import matplotlib.pyplot as plt


def convert_time(str_val):
    """
    # Convert to seconds format
    :param str_val:
    :return:
    """
    strmults = [3600, 60, 1, 0.001]

    numval = 0
    strlist = str_val.split(':')
    for val, mult in zip(strlist, strmults):
        numval += float(val) * mult

    return numval

def getdata():
    """
    Collects data entries for each tip selection algorithm.
    :return:
    """

    algos_test = ["URTS", "MCMC", "EIOTA", "almostURTS", "DT","DTLight"]
    headers = ['??(V)','??(A)','??(h:m:s:ms)']
    allalgospower = []
    algotimes = []
    algoavgpowerval = []

    for readalgo in algos_test:

        algoPower = []
        print("Gettting data for {}".format(readalgo))
        for file in glob.glob("./power_cons/{}/*.csv".format(readalgo)):

            dataframe = pd.read_csv(file,usecols=headers).T
            algotimes.append(list(dataframe.iloc[0]))
            voltage = list(dataframe.iloc[1])
            amperage = list(dataframe.iloc[2])
            power = [voltage[i] * amperage[i] for i in range(len(voltage))]
            algoPower.append(power)

        # Trim values to keep same length for average
        minval = len(algoPower[0])
        for sublist in algoPower:
            minval = min(len(sublist),minval)

        for idx,sublist in enumerate(algoPower):
            algoPower[idx] = sublist[0:minval]

        arrays = [np.array(x) for x in algoPower]
        algoAvg = [np.mean(k) for k in zip(*arrays)]
        algoAvgVal = statistics.mean(algoAvg)
        algoavgpowerval.append(algoAvgVal)
        allalgospower.append(algoAvg)

    # Find x array and minimum time
    minxval = len(algotimes[0])
    for sublist in algotimes:
        minxval = min(minxval, len(sublist))

    xarray = []
    xstrarray = []
    for sublist in algotimes:
        if len(sublist) == minxval:
            xstrarray = sublist
            break

    # Reduce power measurements
    for idx,powermeas in enumerate(allalgospower):
        allalgospower[idx] = powermeas[0:minxval]

    for reading in xstrarray:
        xarray.append(convert_time(reading))

    pickle.dump([algos_test,xarray,allalgospower,algoavgpowerval], open('power_cons_final/energyMeas.p', 'wb'))

def plotdata():
    """
    Plots energy data in matplotlib.
    :return:
    """

    algos_test,xarray,allalgospower,algoavgpowerval = pickle.load(open('power_cons_final/energyMeas.p', 'rb'))

    font = {'size': 18}
    matplotlib.rc('font', **font)

    # Plot Power measurements and show average values
    print(algos_test)
    print("{} {}".format(algoavgpowerval,"(W)"))

    fig,ax = plt.subplots()

    for algo, powermeas in zip(algos_test,allalgospower):
        ax.plot(xarray,powermeas,label=algo)

    ax.set_xlim([xarray[100],xarray[-200]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DC Power (W)")
    ax.legend(loc='best')
    ax.set_ylim([2.0,2.5])
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('./power_cons_final/energyMeas.png', bbox_layout='tight', dpi=400)


if __name__ == "__main__":

    getdata()
    plotdata()




