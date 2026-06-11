#!/bin/python3

import wfdb
import matplotlib.pyplot as plt

class Plotter:
    
    def plot_sample(self, record: wfdb.Record, title="Record Plot"):
        wfdb.plot_wfdb(record=record, title=title, figsize=(15, 8))

    
    def plot_raw_voltages(self, raw_voltages, colour="blue", start=None, end=None):
        plt.figure(figsize=(15, 5))
        if start is not None and end is not None:
            plt.plot(raw_voltages[start:end, 0], label='Raw Voltage Plotting', color=colour)    
        else:
            plt.plot(raw_voltages[:,0], label='Raw Voltage Plotting', color=colour)
        plt.legend(loc='upper left')
        plt.title('Raw Voltage Signal')
        plt.show()

    def plot_raw_voltages_mult_leads(self, raw_voltages, colour="blue"):

        num_sub_plots = raw_voltages.shape[1]
        fig, plots = plt.subplots(num_sub_plots, sharex=True, figsize=(15, 10))
        fig.suptitle('Raw Voltages Multiple Leads')

        for lead in range(num_sub_plots):
            plots[lead].plot(raw_voltages[:,lead], color=colour)
        fig.tight_layout(pad=50.0)

        plt.show()
