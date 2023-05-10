from itertools import groupby
from collections import defaultdict

import latextable
import reportlab.lib.pagesizes as pagesizes
from reportlab.pdfgen import canvas
from texttable import Texttable
import matplotlib
import matplotlib.pyplot as plt

def group_by_lambda(lst, key_func):
    grouped_dict = defaultdict(list)
    for item in lst:
        key = key_func(item)
        grouped_dict[key].append(item)
    return grouped_dict


pagesize = (pagesizes.A2[0], int(pagesizes.A2[1] * 1.2))
columns = 6
column_width = pagesize[0] / columns



def add_image(my_canvas, image_path, column=0, row=0, description=""):
    # my_canvas.drawImage(image_path)
    my_canvas.setFont("Helvetica", 10)
    my_canvas.drawString(column_width * column + 10, pagesize[1] - column_width * (row) - 20, description)
    my_canvas.drawImage(image_path,
                        column_width * column,
                        pagesize[1] - column_width * (row + 1),
                        width=column_width,
                        height=column_width,
                        preserveAspectRatio=True,
                        )


my_canvas = canvas.Canvas("hello2.pdf", pagesize=pagesize)
# my_canvas.drawString(100, 750, "Welcome to Reportlab!")
# add_image(my_canvas, "output/data base dense raw_target_final.png")
# add_image(my_canvas, "output/data base dense raw_target_final.png", column=1)
# add_image(my_canvas, "output/data base dense raw_target_final.png", row=1)
# my_canvas.save()

import json

f = open("aa.json")
raw = f.read()
f.close()
read = json.loads(raw)
print(read)
print()
print([x["data"] for x in read])

result_by_data = {key: list(val) for key, val in groupby(read, key=lambda x: x["data"])}

r = 0

for data, v in result_by_data.items():

    result_by_noise_type = {key: list(val) for key, val in groupby(v, key=lambda x: x["noise_type"])}

    for noise_type, v1 in result_by_noise_type.items():
        for i, x in enumerate(v1):
            nn_type = x["nn_type"]
            rms = float(x["result_error"])
            rms = "{:.4f}".format(rms)
            add_image(my_canvas, x["filename"], i, r, f"{data} {noise_type} {nn_type} {rms}")
        r += 1

my_canvas.save()

# table 1

# table
table_data = [x for x in read if x["plot_variant"] == "target_final"]
table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"], x["noise_type"], x["nn_type"]))
def create_new_measurement_object(object, avg):
    return {
        "window_size": object["window_size"],
        "data": object["data"],
        "noise_type": object["noise_type"],
        "nn_type": object["nn_type"],
        "result_error": str(avg),
}
table_data = [create_new_measurement_object(v[0], sum([float(v1["result_error"]) for v1 in v])/len(v)) for k,v in table_data_grouped.items()]
table_data.sort(key=lambda x: x["nn_type"])
table_data.sort(key=lambda x: x["noise_type"])
table_data.sort(key=lambda x: x["data"])

table_1 = Texttable()
# table_1.set_cols_align(["l", "r", "c", "c"])
# table_1.set_cols_valign(["t", "m", "b", "c"])

table_1.add_rows(
    [["Data source", "Noise type", "Neural network", "RMS error"]] +
    [
        [x["data"], x["noise_type"], x["nn_type"], x["result_error"]]
        for
        x in table_data
    ]
)
print('Texttable Output:')
print(table_1.draw())
print('\nLatextable Output:')
latex_table = latextable.draw_latex(table_1, caption="An example table.", label="table:example_table")
print(latex_table)

###########################################################################################
# table 2 - nn type as column



# table
table_data = [x for x in read if x["plot_variant"] == "target_final"]
table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"], x["noise_type"], x["nn_type"]))
table_data = [create_new_measurement_object(v[0], sum([float(v1["result_error"]) for v1 in v])/len(v)) for k,v in table_data_grouped.items()]
table_data.sort(key=lambda x: int(x["window_size"]))
table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"], x["noise_type"]))
table_data_grouped = {k: {v["nn_type"]: v for v in v} for k, v in table_data_grouped.items()}
# table_data.sort(key=lambda x: x["nn_type"])
# table_data.sort(key=lambda x: x["noise_type"])
# table_data.sort(key=lambda x: x["data"])

table_1 = Texttable()
# table_1.set_cols_align(["l", "r", "c", "c"])
# table_1.set_cols_valign(["t", "m", "b", "c"])

table_1.add_rows(
    [["Window size", "Data source", "Noise type", "Dense", "RNN", "ESN"]] +
    [
        [k[0], k[1], k[2], vals["dense"]["result_error"], vals["rnn"]["result_error"], vals["esn"]["result_error"]]
        for
        k, vals in table_data_grouped.items()
    ]
)
print('Texttable Output:')
print(table_1.draw())
print('\nLatextable Output:')
latex_table = latextable.draw_latex(table_1, caption="An example table.", label="table:example_table")
print(latex_table)


######################################################
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Computer Modern"
matplotlib.rcParams["font.size"] = 10



#quality vs windows size
for nn_type in ["dense", "rnn", "esn"]:
    table_data = [x for x in read if x["plot_variant"] == "target_final"]
    table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"], x["noise_type"], x["nn_type"]))
    table_data = [create_new_measurement_object(v[0], sum([float(v1["result_error"]) for v1 in v])/len(v)) for k,v in table_data_grouped.items()]
    table_data = [x for x in table_data if x["data"] == "data3"]
    table_data = [x for x in table_data if x["nn_type"] == nn_type]
    table_data1 = [x for x in table_data if x["noise_type"] == "low freq"]
    table_data2 = [x for x in table_data if x["noise_type"] == "base"]
    table_data3 = [x for x in table_data if x["noise_type"] == "high freq"]
    table_data1.sort(key=lambda x: int(x["window_size"]))
    table_data2.sort(key=lambda x: int(x["window_size"]))
    table_data3.sort(key=lambda x: int(x["window_size"]))

    for table_data in [table_data1, table_data2, table_data3]:
        plt.plot(
            [float(x["window_size"]) for x in table_data],
            [float(x["result_error"]) for x in table_data],
        )
    plt.xscale('log')
    xticks = list(set([int(x["window_size"])for x in table_data]))
    xticks.sort()
    plt.xticks(xticks)
    tick_labels = ['{}'.format(tick) for tick in xticks]
    plt.gca().set_xticklabels(tick_labels)
    plt.grid(which='major', alpha=0.2)

    plt.savefig('plot_' + nn_type + '.pdf')
    plt.show()




########################################
# second type


# table
table_data = [x for x in read if x["plot_variant"] == "target_final"]
table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"], x["noise_type"], x["nn_type"]))
table_data = [create_new_measurement_object(v[0], sum([float(v1["result_error"]) for v1 in v])/len(v)) for k,v in table_data_grouped.items()]
table_data.sort(key=lambda x: int(x["window_size"]))
table_data_grouped = group_by_lambda(table_data, lambda x: (x["window_size"], x["data"].split("_")[0],x["data"].split("_")[1]))
table_data_grouped = {k: {v["nn_type"]: v for v in v} for k, v in table_data_grouped.items()}
table_data_grouped = dict(sorted(table_data_grouped.items(), key=lambda item: int(item[0][2])))
table_data_grouped = dict(sorted(table_data_grouped.items(), key=lambda item: int(item[0][1])))
table_data_grouped = dict(sorted(table_data_grouped.items(), key=lambda item: int(item[0][0])))
# table_data.sort(key=lambda x: x["nn_type"])
# table_data.sort(key=lambda x: x["noise_type"])
# table_data.sort(key=lambda x: x["data"])

table_1 = Texttable()
# table_1.set_cols_align(["l", "r", "c", "c"])
# table_1.set_cols_valign(["t", "m", "b", "c"])

table_1.add_rows(
    [["Window size", "Measurements", "Noise floor", "Dense", "RNN", "ESN"]] +
    [
        [k[0], k[1], k[2], vals["dense"]["result_error"], vals["rnn"]["result_error"], vals["esn"]["result_error"]]
        for
        k, vals in table_data_grouped.items()
    ]
)
print('Texttable Output:')
print(table_1.draw())
print('\nLatextable Output:')
latex_table = latextable.draw_latex(table_1, caption="An example table.", label="table:example_table")
print(latex_table)