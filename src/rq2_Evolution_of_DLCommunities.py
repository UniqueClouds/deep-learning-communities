import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
import re


def load_stars(data_path: str, community: str):
    # file = f"{data_path}{community}_monthly.csv"
    df = pd.read_csv(f"{data_path}{community}_monthly.csv")
    dates = df['timestamp'].tolist()
    stars = df['stars'].tolist()
    return dates, stars

# def parse_timestamp(timestamp):
#     formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y.%m.%d"]
    
#     for fmt in formats:
#         try:
#             return datetime.datetime.strptime(timestamp, fmt)
#         except ValueError:
#             pass
    
#     return datetime.datetime(1970, 1, 1)
# def parse_time(timestamp):
#     return datetime.datetime.strptime(str(timestamp), "%Y-%m-%d")
def parse_timestamp(timestamp):
    # YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD or YYYY.MM.DD
    if not isinstance(timestamp, str):
        return datetime.datetime(1970, 1, 1)

    if (
        not re.match("^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", timestamp)
        and not re.match("\d{4}.\d{2}.\d{2}$", timestamp)
        and not re.match("\d{4}-\d{2}-\d{2}$", timestamp)
    ):
        return datetime.datetime(1970, 1, 1)

    year, month, day = int(timestamp[:4]), int(timestamp[5:7]), int(timestamp[8:10])
    if len(timestamp) == 20:
        hour, minute, second = (
            int(timestamp[11:13]),
            int(timestamp[14:16]),
            int(timestamp[17:19]),
        )
        return datetime.datetime(year, month, day, hour, minute, second)
    else:
        return datetime.datetime(year, month, day)


def parse_time(timestamp):
    print(timestamp)
    timestamp = str(timestamp)
    # parse time that format is "xxxx-xx-xx"
    year = int(timestamp[:4])
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    return datetime.datetime(year, month, day)


class Plotter:
    """
    Plotter class for creating and saving plots based on software release information.

    Attributes:
    - info: DataFrame containing software release information.
    - target: Name of the target software/project.
    - save_dir: Directory to save the plots.
    - ... (other attributes related to plotting aesthetics)
    """

    def __init__(self,data_path, save_dir, target, info, info_=None):
        """
        Initializes the Plotter with the provided information.

        Parameters:
        - save_dir (str): Directory to save the plots.
        - target (str): Name of the target software/project.
        - info (pd.DataFrame): DataFrame containing software release information.
        - info_ (pd.DataFrame, optional): Additional information (default is None).
        """
        self.info = info
        self.info_ = info_
        self.target = target
        self.data_path = data_path
        self.versions = self.info["version"]
        self.dates = []
        self.x_lim = 10000
        self.edge_lim = self.x_lim * 0.02
        self.x = self.versions2x()
        self.legend_font = {"family": "Times New Roman", "weight": "normal", "size": 10}
        self.legend_font_contri = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 15,
        }
        self.legend_font_ratio = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 15,
        }
        self.legend_font_tran = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 15,
        }
        self.legend_font_pop = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 18,
        }
        self.tick_font = {"weight": "bold", "size": 12}
        self.label_font = {"weight": "bold", "size": 15}
        self.save_dir = save_dir

    def draw_and_save(self):
        """Creates and saves all the plots."""
        # self.time_interval_visualizer()
        self.draw_contributors()
        self.draw_ratio()
        self.draw_transition()
        self.draw_pop(self.data_path)

    def versions2x(self):
        """Converts version release dates to x-coordinates for plotting."""
        t2 = parse_time(self.info["release_date"].iloc[-1])
        t1 = parse_time(self.info["release_date"].iloc[0])
        ratio = (self.x_lim - 2 * self.edge_lim) / (t2 - t1).total_seconds()
        st = self.edge_lim
        ans = []
        for version in self.versions:
            tmp = parse_time(
                self.info.loc[self.info["version"] == version, "release_date"].iloc[0]
            )
            ans.append(st + ratio * (tmp - t1).total_seconds())
        return ans

    def draw_contributors(self, show=False):
        """Creates and saves a plot showing the number of contributors."""
        au = []
        core = []
        peri = []
        for version in self.versions:
            au.append(
                self.info.loc[self.info["version"] == version, "active_users"].iloc[0]
            )
            core.append(
                self.info.loc[self.info["version"] == version, "core_developers"].iloc[
                    0
                ]
            )
            peri.append(
                self.info.loc[
                    self.info["version"] == version, "peripheral_developers"
                ].iloc[0]
            )
            # au.append(self.info[version]["active_users"])
            # core.append(self.info[version]["core_developers"])
            # peri.append(self.info[version]["peripheral_developers"])
        # plt.cla()
        fig = plt.figure()
        plt.plot(self.x, au, "o-", color="red", label="active users")
        plt.plot(self.x, peri, "^--", color="green", label="peripheral developers")
        plt.plot(self.x, core, "x-.", color="blue", label="core developers")
        plt.xticks(
            self.x,
            self.versions,
            rotation=60,
            size=self.tick_font["size"],
            weight=self.tick_font["weight"],
        )
        plt.yticks(size=self.tick_font["size"], weight=self.tick_font["weight"])
        plt.legend(loc="upper left", prop=self.legend_font_contri)
        plt.xlabel("Releases", fontdict=self.label_font)
        plt.ylabel("Num of Contributors", fontdict=self.label_font)
        plt.grid(linestyle="--", linewidth=0.5)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            fig.savefig(self.save_dir + f"contributors_{self.target}.pdf", dpi=fig.dpi)

    def draw_ratio(self, show=False,save=False):
        """Creates and saves a plot showing the contributor ratios."""
        r_pc = []
        r_ad = []
        for version in self.versions:
            # r_pc.append(self.info[version]["ratio p-c"])
            # r_ad.append(self.info[version]["ratio a-d"])
            r_pc.append(
                self.info.loc[self.info["version"] == version, "ratio_peri_core"].iloc[
                    0
                ]
            )
            r_ad.append(
                self.info.loc[self.info["version"] == version, "ratio_active_dev"].iloc[
                    0
                ]
            )
            # r_pc.append(self.info[version]["ratio_peri_core"])
            # r_ad.append(self.info[version]['ratio_active_dev'])
        # plt.cla()
        fig = plt.figure()
        plt.plot(
            self.x,
            r_pc,
            "o-",
            color="red",
            label="peripheral developers / core developers",
        )
        plt.plot(self.x, r_ad, "^--", color="green", label="active users / developers")
        # plt.xticks(self.x, self.versions, rotation=60)
        plt.xticks(
            self.x,
            self.versions,
            rotation=60,
            size=self.tick_font["size"],
            weight=self.tick_font["weight"],
        )
        plt.yticks(size=self.tick_font["size"], weight=self.tick_font["weight"])
        # plt.yticks(size=self.tick_font['size'])
        plt.ylim([0, max(max(r_ad), max(r_pc)) + 2])
        plt.legend(loc="lower right", prop=self.legend_font_ratio)
        # plt.legend(loc='lower right')
        plt.xlabel("Releases", fontdict=self.label_font)
        # plt.xlabel('Releases')
        plt.ylabel("Ratio", fontdict=self.label_font)
        plt.grid(linestyle="--", linewidth=0.5)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            fig.savefig(self.save_dir + f"ratio_{self.target}.pdf", dpi=fig.dpi)

    def draw_transition(self, show=False,save=False):
        """Creates and saves a plot showing the role transition rates."""
        c2c = []
        c2p = []
        c2a = []
        inf = []
        for version in self.versions:
            if version == "0.2.0" or version == "0.6.0":
                continue
            c2c.append(
                self.info.loc[self.info["version"] == version, "retention_rate"].iloc[0]
            )
            c2p.append(
                self.info.loc[
                    self.info["version"] == version, "core2peripheral_rate"
                ].iloc[0]
            )
            c2a.append(
                self.info.loc[self.info["version"] == version, "dropout_rate"].iloc[0]
            )
            inf.append(
                self.info.loc[self.info["version"] == version, "inflow_rate"].iloc[0]
            )

        # plt.cla()
        fig = plt.figure()
        plt.plot(self.x[1:], c2c, "o-", color="red", label="retention")
        plt.plot(self.x[1:], c2p, "^--", color="green", label="core2peripheral")
        plt.plot(self.x[1:], c2a, "x-.", color="blue", label="dropout")
        plt.plot(self.x[1:], inf, "+:", color="orange", label="inflow")
        plt.xticks(
            self.x,
            self.versions,
            rotation=60,
            size=self.tick_font["size"],
            weight=self.tick_font["weight"],
        )
        plt.yticks(size=self.tick_font["size"], weight=self.tick_font["weight"])
        plt.legend(loc="center right", prop=self.legend_font_tran)
        plt.xlabel("Releases", fontdict=self.label_font)
        plt.ylabel("Role Transition Rate", fontdict=self.label_font)
        plt.grid(linestyle="--", linewidth=0.5)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            fig.savefig(self.save_dir + f"transition_{self.target}.pdf", dpi=fig.dpi)

    def draw_pop(self, show=False, save=False):
        data_path = self.data_path
        pt_dates, pt_stars = load_stars(data_path, "pytorch")
        tf_dates, tf_stars = load_stars(data_path, "tensorflow")
        st, ed = parse_time(tf_dates[0]), parse_time(tf_dates[-1])
        ratio = (self.x_lim - 2 * self.edge_lim) / (ed - st).total_seconds()
        ans = []
        for date in pt_dates:
            tmp = parse_time(date)
            ans.append(self.edge_lim + ratio * (tmp - st).total_seconds())
        # plt.cla()
        fig = plt.figure()
        plt.plot(ans, pt_stars, color="red", label="PyTorch")

        ans = []
        for date in tf_dates:
            tmp = parse_time(date)
            ans.append(self.edge_lim + ratio * (tmp - st).total_seconds())
        plt.plot(ans, tf_stars, color="green", label="TensorFlow")

        # labeled version of tensorflow
        versions = ["0.11.0", "1.0.0", "1.14.0", "2.0.0"]
        dates = ["2016.11.09", "2017.02.15", "2019.06.19", "2019.10.01"]
        stars = [32660, 40934, 121121, 127779]
        for i in range(len(versions)):
            tmp = parse_time(dates[i])
            x = self.edge_lim + ratio * (tmp - st).total_seconds()
            plt.vlines(x, 0, stars[i], linestyles="dotted", colors="green")
            plt.scatter(x, stars[i], color="green")
            plt.text(
                x,
                stars[i],
                versions[i],
                color="green",
                horizontalalignment="right",
                verticalalignment="bottom",
            )

        dates = [
            "2016.03.01",
            "2017.01.01",
            "2018.01.01",
            "2019.01.01",
            "2020.01.01",
            "2021.01.01",
            "2022.01.01",
        ]
        names = [
            "2016\nMar 1",
            "2017\nJan 1",
            "2018\nJan 1",
            "2019\nJan 1",
            "2020\nJan 1",
            "2021\nJan 1",
            "2022\nJan 1",
        ]
        x_set = []
        for date in dates:
            tmp = parse_time(date)
            x = self.edge_lim + ratio * (tmp - st).total_seconds()
            x_set.append(x)

        plt.xticks(x_set, names, weight=self.tick_font["weight"])
        plt.yticks(weight=self.tick_font["weight"])
        plt.ylim(bottom=0)
        plt.legend(loc="best", prop=self.legend_font_pop)
        plt.ylabel("Stars", fontdict=self.label_font)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            fig.savefig(self.save_dir + f"stars.pdf", dpi=fig.dpi)
