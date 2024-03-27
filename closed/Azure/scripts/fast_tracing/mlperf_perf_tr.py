#!/home/scratch.jesusc_sw/bin/release/dl-inference-bridge-generator/venv/bin/python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#################################
#   packages
#################################

# global packages
import os
import sys
import re
import argparse
import json
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


#################################
#   visualization
#################################

# Matplotlib
# ----------

def plt_plot_series(df, y_tag, color='green', out_file=None):
    df.plot(y=y_tag, ylabel=y_tag, color=color, linewidth=2)
    if out_file is None or out_file == 'gui':
        plt.show()
    else:
        plt.savefig(out_file)


def plt_plot_scatter(df, x_tag, y_tag, color='green', out_file=None):
    df.plot.scatter(x=x_tag, y=y_tag, xlabel=x_tag, ylabel=y_tag, color=color, s=1.0, alpha=0.5)
    if out_file is None or out_file == 'gui':
        plt.show()
    else:
        plt.savefig(out_file, dpi=200)


def plt_plot_hist(df, tag, out_file=None):
    df_tmp = df[df[tag] != 0]
    df_tmp[tag].plot.hist(bins=1200, alpha=0.5)
    if out_file is None or out_file == 'gui':
        plt.show()
    else:
        plt.savefig(out_file)


# plotly
# ------

def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def plotly_plot_scatter(df, x_tag, y_tag, color=None):

    # fig = px.scatter(curr_df, x=metric_base_list[0], y=metric_base_list[1],
    #                  hover_data=["Layer", "Kernel", time_base, "At", "ResMma", "ResDram"],
    #                  color=color_base, size=size_base, range_x=range_x, range_y=range_y, title=title)

    fig = px.scatter(df, x=x_tag, y=y_tag, color=color)
    return fig

#################################
#   backend tracing
#################################


def perfetto_add_inst_event(df, trace_list, tag, tid, subtag=""):

    for (time, pid, count) in zip(df["time"].to_list(), df["pid"].to_list(), df[tag].to_list()):
        # print("{}: {},{},{}".format(tag, time, pid, count))
        if count == 0:
            continue

        event = {
            "name": tag + subtag + "_" + str(count),
            "cat": tag + subtag,
            "ph": "i",
            "ts": time * 1000,
            # "dur" : duration,
            "pid": pid & 0xFFFFFFFF,
            "tid": tid,
            "s": "t"
            # "args" : { "bs" : bs, "isl" : isl, "osl" : osl }
        }
        trace_list.append(event)


def perfetto_add_counter(df, trace_list, tag, tag2=""):

    last_value = None
    for (time, pid, count) in zip(df["time"].to_list(), df["pid"].to_list(), df[tag].to_list()):
        # print("{},{},{}".format(time, pid, count))
        if last_value == count:
            continue
        counter = {
            "name": "stats:",
            "ph": "C",
            "ts": time * 1000,
            "pid": pid,
            "tid": 0,
            "args": {tag + tag2: count}
        }
        last_value = count
        trace_list.append(counter)


def generate_chrome_trace(data_l):

    trace_list = list()
    base_time = None

    for item in data_l:
        # (isl, osl, bs, start, end, gen) = item
        (pid, bs, isl, osl, start, gen, end) = (item['pid'], item['bs'], item['isl'], item['osl'], item['start'], item['gen'], item['end'])

        if base_time is None:
            base_time = start * 1000.0

        duration = round((end - start), 3) * 1000
        norm_start = (start * 1000.0) - base_time

        # print((isl, osl, bs, start, end))
        # events
        event = {
            "name": "bs{}_i{}_o{}".format(bs, int(isl), int(osl)),
            "cat": "batch",
            "ph": "X",
            "ts": norm_start,
            "dur": duration,
            "pid": pid,
            "tid": 0,
            "args": {"bs": bs, "isl": isl, "osl": osl}
        }

        # stat counters
        counter = {
            "name": "stats::",
            "ph": "C",
            "ts": norm_start,
            "pid": pid,
            "tid": 0,
            "args": {"bs": bs, "isl": isl, "osl": osl, "tps": 1000000.0 * float(bs) * float(osl) / float(duration)}
        }
        trace_list.append(event)
        trace_list.append(counter)

        # gen/context
        context_duration = round((gen - start), 3) * 1000
        context_event = {
            "name": "c_{}".format(isl),
            "cat": "batch,gen",
            "ph": "X",
            "ts": norm_start,
            "dur": context_duration,
            "pid": pid, "tid": 0,
            "args": {"bs": bs, "isl": isl, "osl": osl}
        }
        gen_duration = duration - context_duration
        gen_event = {
            "name": "g_{}".format(osl),
            "cat": "batch,gen",
            "ph": "X",
            "ts": norm_start + context_duration,
            "dur": gen_duration,
            "pid": pid, "tid": 0,
            "args": {"bs": bs, "isl": isl, "osl": osl}
        }
        trace_list.append(context_event)
        trace_list.append(gen_event)

    # print(json.dumps(trace_list, indent=4))
    return trace_list


#################################
#   postprocessing prim
#################################

def get_interarrival_time(df, field):
    ''' Given a series of event arrivals, return a series with inter-arrival times '''

    df_index = (df[field] != 0)
    time_l = df.loc[df_index, "time"].to_list()
    count_l = df.loc[df_index, field].to_list()
    res_l = list()
    curr_time = None

    for instance, time in zip(count_l, time_l):
        if curr_time is not None:
            delta = float(time - curr_time) / float(instance)
            # print(f'{time} - {curr_time} / {instance} = {delta}')
            for i in range(instance):
                res_l.append(delta)

        # epilogue
        curr_time = time

    # print(res_l)
    df_out = pd.DataFrame.from_dict({field: res_l})
    return (df_out)


def get_expanded_series(df, time, field):
    ''' Expand time series where field is an event count '''

    df_index = (df[field] != 0)
    time_l = df.loc[df_index, time].to_list()
    count_l = df.loc[df_index, field].to_list()
    res_l = list()

    for instance, time in zip(count_l, time_l):
        for i in range(instance):
            res_l.append(time)
    # print("{} - {}".format(field,res_l[0:40]))
    return (res_l)


def get_latency(df, arriv, depart):
    ''' Get latencies give arriv and depart events '''
    arriv_l = get_expanded_series(df, "time", arriv)
    depart_l = get_expanded_series(df, "time", depart)
    latency_l = [(depart - arriv) for depart, arriv in zip(depart_l, arriv_l)]
    depart_l = depart_l[:len(latency_l)]  # FIXME
    # print(latency_l[0:100]); exit(0)
    df_latency = pd.DataFrame.from_dict({"time": depart_l, "latency": latency_l})
    return (df_latency)


def get_rate(df, field):
    df_index = (df[field] != 0)
    arriv_l = df.loc[df_index, "time"].to_list()
    count_l = df.loc[df_index, field].to_list()
    next_arriv_l = arriv_l[1:]

    latency_l = [(narriv - arriv) for narriv, arriv in zip(next_arriv_l, arriv_l[:-2]) if ((narriv - arriv) > 0)]
    time_l = [arriv for narriv, arriv in zip(next_arriv_l, arriv_l[:-2]) if ((narriv - arriv) > 0)]
    rate_cnt_l = [cnt for narriv, arriv, cnt in zip(next_arriv_l, arriv_l[:-2], count_l[1:]) if ((narriv - arriv) > 0)]

    # for count, latency, t in zip(rate_cnt_l,latency_l, time_l) : print(f'c={count} l={latency} t={t}')
    rate_l = [1000.0 * float(count) / float(latency) for count, latency in zip(rate_cnt_l, latency_l)]
    df_rate = pd.DataFrame.from_dict({"time": time_l, "rate-" + field: rate_l})
    df_rate["pid"] = 0
    return (df_rate)


def get_compressed_series(df, sortL, field, function=max):
    ''' Compress a time series by reducing entries with same time (max/min/sum) '''
    df_filt = df[df[field] > 0]

    if function is sum:
        df_comp = df_filt.groupby(sortL).sum()
    elif function is max:
        df_comp = df_filt.groupby(sortL).max()
    elif function is min:
        df_comp = df_filt.groupby(sortL).min()
    df_comp = df_comp.reset_index()
    returnL = sortL + [field]
    return (df_comp[returnL])


def get_filtered_series(df, time, field, function=max):
    ''' Filter a time series by reducing entries with same time (max/min) '''
    idx = df.groupby([time])[field].transform(function) == df[field]
    return (df[idx], idx)

#################################
#   front-end
#################################

# ------------------------------------------------------------------
# PTR :: thredId :: time :: record :: tag :: count :: field :: value
# ------------------------------------------------------------------
# enum ptrecord
# {
#     PUSH = 0,
#     POP = 1,
#     EVENT = 2
# };
#
# enum ptfield
# {
#     UNDEF = 0,
#     BS = 1,
#     ISL = 2,
#     OSL = 3
# };

# PTR:140164889231360;1697838117460;2;0;1;1;1
# PTR:140164889231360;1697838117460;2;0;1;1;4
# PTR:140164889231360;1697838117460;2;0;1;1;3
# PTR:140164889231360;1697838117460;2;0;1;1;7
# PTR:140164889231360;1697838117460;2;0;1;1;1
# PTR:140164889231360;1697838117460;2;0;1;1;3
# PTR:140164889231360;1697838117460;2;0;1;1;4
# PTR:140164889231360;1697838117460;2;0;1;1;2


def parse_ptr_line(line):

    match = re.search(r'PTR:(\d+);(\d+);(\d+);(\d+);(\d+);(\d+);(\d+)\s*$', line)
    if match:
        (pid, time, record, tag, count, field, value) = match.groups()
        return (True, {"pid": int(pid), "time": int(time), "record": int(record), "tag": int(tag), "count": int(count), "field": int(field), "value": int(value)})
    else:
        return (False, None)


def process_file(file_name):

    fh = open(file_name)

    queue = 0

    stats_d = {"time": list(), "queue": list(), "pid": list(), "enq": list(), "deq": list()}

    for line in fh:
        (hit, data) = parse_ptr_line(line)

        # if hit : print(data)
        if hit:
            if data['tag'] == 0:
                queue = queue + data['value']
                enq = data['value']
                deq = 0
                tag_pid = "enq-pid"
                notag_pid = "deq-pid"
            elif data['tag'] == 1:
                queue = queue - data['value']
                enq = 0
                deq = data['value']
            else:
                continue
            # print("{}:{} - {} : {} : {}".format(data['tag'],data['time'], enq, deq, queue))

            # check for cache hit
            stats_d["time"].append(data['time'])
            stats_d["queue"].append(queue)
            stats_d["enq"].append(enq)
            stats_d["deq"].append(deq)
            stats_d["pid"].append(data['pid'])

    # build dataframe
    df = pd.DataFrame.from_dict(stats_d)
    # df.to_csv("debug.csv"); exit(0)
    return df


def report_percentiles(df_latency):
    print("lat min  :\t{}".format(float(df_latency['latency'].min())))
    print("lat mean :\t{}".format(round(float(df_latency['latency'].mean()), 3)))
    print("lat p90  :\t{}".format(float(df_latency['latency'].quantile(q=0.9))))
    print("lat p99  :\t{}".format(float(df_latency['latency'].quantile(q=0.99))))
    print("lat p99.9:\t{}".format(float(df_latency['latency'].quantile(q=0.999))))
    print("lat max  :\t{}".format(float(df_latency['latency'].max())))


def trace_generation(df, out_file="test.json"):

    if True:
        trace_list = list()
        # Add basic counters
        df_enq = get_compressed_series(df, ["time", "pid"], "enq", sum)
        perfetto_add_inst_event(df_enq, trace_list, "enq", 0)
        df_deq = get_compressed_series(df, ["time", "pid"], "deq", sum)
        perfetto_add_inst_event(df_deq, trace_list, "deq", 1)

        df_counter = get_compressed_series(df, ["time", "pid"], "queue", max)
        perfetto_add_counter(df_counter, trace_list, "queue")

        # add arrival rates
        df_enq = get_compressed_series(df, ["time"], "enq", sum)
        df_arrival_rate = get_rate(df_enq, "enq")
        perfetto_add_counter(df_arrival_rate, trace_list, "rate-enq")

        df_deq = get_compressed_series(df, ["time"], "deq", sum)
        df_depart_rate = get_rate(df_deq, "deq")
        perfetto_add_counter(df_depart_rate, trace_list, "rate-deq")

        # add latency percentiles
        df_latency = get_latency(df, "enq", "deq")
        report_percentiles(df_latency)
        df_latency['pid'] = 0
        p90_0 = float(df_latency['latency'].quantile(q=0.9))
        p99_0 = float(df_latency['latency'].quantile(q=0.99))
        p99_9 = float(df_latency['latency'].quantile(q=0.999))
        perfetto_add_inst_event(df_latency[(df_latency['latency'] > p99_0) & (df_latency['latency'] < p99_9)],
                                trace_list, "latency", 2, "-perc99")
        perfetto_add_inst_event(df_latency[df_latency['latency'] > p99_9],
                                trace_list, "latency", 3, "-perc99.9")

        # Write file
        fh = open(out_file, 'w')
        fh.write(json.dumps(trace_list, indent=4))
        fh.close()


def trace_visualization(df, out_file=None):

    # get latencies
    if False:
        df_latency = get_latency(df, "enq", "deq")
        # plt_plot_series(df_latency, "latency");
        pass_series = pd.qcut(df_latency['latency'], [0, 0.5, 0.99, 0.999, 1.0], labels=["lime", "forestgreen", "yellow", "red"], duplicates='drop')
        # plt_plot_scatter(df_latency, "time", "latency", pass_series)

        df_latency, idx = get_filtered_series(df_latency, "time", "latency")
        plt_plot_scatter(df_latency, "time", "latency", pass_series[idx], out_file)

    if True:
        df_latency = get_latency(df, "enq", "deq")
        p99_0 = float(df_latency['latency'].quantile(q=0.99))
        p99_9 = float(df_latency['latency'].quantile(q=0.999))
        df_latency, idx = get_filtered_series(df_latency, "time", "latency")
        df_latency['color'] = "green"
        df_latency.loc[df_latency['latency'] > p99_0, 'color'] = "red"
        plt_plot_scatter(df_latency, "time", "latency", df_latency['color'], out_file)

    # fig_list = list()
    # # fig_list.append(plotly_plot_scatter(df_latency, "time", "latency", pass_series))
    # fig_list.append(plotly_plot_scatter(df, "time", "enq"))
    # figures_to_html(fig_list, "test.html")

    # print(float(df_latency.min()))
    # print(float(df_latency.mean()))
    # print(float(df_latency.quantile(q=0.9)))
    # print(float(df_latency.quantile(q=0.99)))
    # print(float(df_latency.quantile(q=0.999)))
    # print(float(df_latency.max()))

    # -- visualize
    # print(df)
    # plot_scatter(df, "time", "queue")
    # plot_scatter(df, "time", "enq", "red")
    # plot_scatter(df, "time", "deq", "blue")
    # plot_scatter(df, "time", ["queue", "enq"], ["blue", "red"])

    # plot_hist(df, ["enq", "deq"])
    # plot_hist(df, ["enq"])
    # plot_hist(df, ["deq"])

    # df_deq_time = get_interarrival_time(df, "enq")
    # plot_series(df_deq_time, "enq")
    # print(len(df_deq_time["deq"].to_list()))
    # print(df_deq_time["deq"].min())
    # print(df_deq_time["deq"].mean())
    # print(df_deq_time["deq"].max())
    # df.to_csv("test.csv")


#################################
#   main
#################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input log file")
    parser.add_argument("-o", "--output", required=False, default=None, help="Json trace file")
    parser.add_argument("-v", "--view", required=False, default=None, help="Visualize latencies (gui|svg|png)")
    args = parser.parse_args()

    df = process_file(args.input)
    if args.output is not None:
        trace_generation(df, args.output)
    if args.view is not None:
        trace_visualization(df, args.view)


if __name__ == "__main__":
    main()
