import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import uuid


class Plot:
    def __init__(self, title='Training', x_label='Episode', y_label='Reward', x_data=None, y_data=None, plot_freq=1):
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []

        plt.ion()

        self.title = title

        self.x_label = x_label
        self.y_label = y_label

        self.figure = plt.figure()
        self.figure.set_figwidth(8)

        self.x_data = x_data
        self.y_data = y_data

        self.plot_num = 0
        self.plot_freq = plot_freq

        sns.set_theme(style="darkgrid")

    def post(self, reward):
        self.plot_num += 1

        self.x_data.append(len(self.x_data))
        self.y_data.append(reward)

        if self.plot_num % self.plot_freq == 0:
            self.__create_plot()
            plt.pause(10e-10)

    def plot(self):
        plt.ioff()
        self.__create_plot()
        plt.show()

    def __create_plot(self):
        data_dict = {self.x_label: self.x_data, self.y_label: self.y_data}
        df = pd.DataFrame(data=data_dict)

        sns.lineplot(data=df, x=self.x_label, y=self.y_label)
        plt.title(self.title)

    def save_plot(self, file_name=str(uuid.uuid4().hex)):
        self.__create_plot()

        dir_exists = os.path.exists("figures")

        if not dir_exists:
            os.makedirs("figures")

        plt.savefig(f"figures/{file_name}")

    def save_csv(self, file_name=str(uuid.uuid4().hex)):
        dir_exists = os.path.exists("data")

        if not dir_exists:
            os.makedirs("data")

        data_dict = {self.x_label: self.x_data, self.y_label: self.y_data}
        df = pd.DataFrame(data=data_dict)

        df.to_csv(f"data/{file_name}", index=False)
