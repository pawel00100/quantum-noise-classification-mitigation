import pandas as pd

import os
import math
import matplotlib
import numpy as np

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.layers import Reshape, Softmax
from scipy.fft import fft, ifft, fftfreq, dct, idct
from scipy import signal

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, AveragePooling2D, Embedding, LSTM, SimpleRNN
from keras.utils import np_utils

import torch
import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN, DeepESN
from auto_esn.esn.reservoir.util import NRMSELoss
from auto_esn.esn.reservoir.activation import tanh

import random
import filterpy
from itertools import groupby



class OutputIndex():
    def __init__(self, data, noise_type, window_size, nn_type=None, plot_variant=None, filename=None, result_error = None) -> None:
        self.data = data
        self.noise_type = noise_type
        self.window_size = window_size
        self.nn_type = nn_type
        self.plot_variant = plot_variant
        self.filename = filename
        self.result_error = result_error

    def with_nn_type(self, nn_type):
        return OutputIndex(self.data, self.noise_type, self.window_size, nn_type, self.plot_variant, self.filename)

    def with_plot_variant(self, plot_variant, filename, error):
        return OutputIndex(self.data, self.noise_type, self.window_size, self.nn_type, plot_variant, filename, error)


output_folder = "output/"
output_indexes = []

input_files = ["../data_noise_sinus/data_noise_sinus.txt", "../data_noise_multitone/data_noise_multitone.txt", "../data_feb_2023/data.txt"]


def parse_line(line: str):
    strings = [x for x in line.split(" ") if x != "" and x != "->"][0:-1]
    strings = [float(x) if i > 1 else x for i, x in enumerate(strings)]
    datetime_str = strings[0] + 'T' + strings[1]
    datetime_str = datetime_str[:19] + "." + datetime_str[20:]
    strings = [pd.Timestamp(datetime_str)] + strings[2:]
    return strings

def prepare_raw_data(data, oscillation):
    theta_offsets = [oscillation.angle_offsets(t)[0] for t in range(len(data))]
    phi_offsets = [oscillation.angle_offsets(t)[1] for t in range(len(data))]

    thetas = [d[0] + t for d, t in zip(data, theta_offsets)]
    phis = [d[1] + t for d, t in zip(data, phi_offsets)]

    phi_offsets = [y + 44 for y in phi_offsets]

    return thetas, phis, theta_offsets, phi_offsets

class Input: # [(theta, phi, target_theta, tharget_phi)] + labda
    def open_and_parse_file(self, file_name: str):
        raise NotImplementedError

class DeviceInput(Input):
    def open_and_parse_file(self, file_name: str):
        input = open(file_name).readlines()
        # return [parse_line(l) for l in input[6:]]
        lines = [parse_line(l) for l in input[6:]]
        return [(l[1], l[2], 0.0, 0.0) for l in lines], lambda data, oscillation: prepare_raw_data(data, oscillation)

class PyquilInput(Input):
    def open_and_parse_file(self, file_name: str, target_file_name: str):
        input = open(file_name).readlines()
        input2 = open(target_file_name).readlines()
        phis = [float(l) for l in input]
        phi_targets = [float(l) for l in input2]
        return [(None, p, None, pt) for (p, pt) in zip(phis, phi_targets)], lambda data, oscillation: zip(*data)

# columns = ['datetime', 'Theta', 'Phi', 'X', 'Y', 'Z', '|0>', '|1> r', '|1> i']

# input = open("../data_noise_sinus/data_noise_sinus.txt").readlines()
# data = [parse_line(l) for l in input[6:]]
#
# input2 = open("../data_noise_multitone/data_noise_multitone.txt").readlines()
# data2 = [parse_line(l) for l in input2[6:]]
#
# input3 = open("../data_feb_2023/data.txt").readlines()
# data3 = [parse_line(l) for l in input3[6:]]

data = DeviceInput().open_and_parse_file("../data_noise_sinus/data_noise_sinus.txt")

data2 = DeviceInput().open_and_parse_file("../data_noise_multitone/data_noise_multitone.txt")

data3 = DeviceInput().open_and_parse_file("data_feb_2023/data.txt")

data4 = PyquilInput().open_and_parse_file("one.txt", "target.txt")

pyquil_location = "../pyquil_playground/out"
files = os.listdir(pyquil_location)
data_pquil = {f[:-9]: PyquilInput().open_and_parse_file(pyquil_location+"/"+f, pyquil_location+"/"+f[:-9] + "_target.txt") for f in files if not f.endswith("target.txt")}

print(data[0])
print(data4[0])

class Filter:
    def filter_1d_array(self, array):
        raise NotImplemented


class EmptyFilter(Filter):
    def filter_1d_array(self, array):
        return array


class IIR(Filter):
    def __init__(self, r: float, start=None):
        self.r = r
        self.start = start

    def filter_1d_array(self, array):
        output = []
        if self.start is None:
            x = array[0]
        else:
            x = self.start
        for i in range(len(array)):
            x = self.r * x + (1 - self.r) * array[i]
            output.append(x)
        return output


class RunningAverage(Filter):
    def __init__(self, width: int):
        self.width = width

    def filter_1d_array(self, array):
        f = np.array([1 for _ in range(self.width)])
        return signal.convolve(f, array)


class Gaussian(Filter):
    def __init__(self, width: int, std_dev: float):
        self.width = width
        self.std_dev = std_dev

    def filter_1d_array(self, array):
        f = signal.windows.gaussian(self.width, self.std_dev)
        return signal.convolve(f, array)


def plot_with_fft(amplitude, name, base_filename):
    avg = sum(amplitude) / len(amplitude)
    amplitude_offseted = list(map(lambda x: x - avg, amplitude))
    amplitude_fft = fft(amplitude_offseted)
    amplitude_fft_amplitude = np.sqrt(np.real(amplitude_fft) ** 2 + np.imag(amplitude_fft) ** 2)
    # N =len(amplitude_fft_amplitude)
    # xf = fftfreq(N, average_delta)[:N//2]
    amplitude_fft_amplitude = amplitude_fft_amplitude[0:len(amplitude_fft_amplitude) // 2]
    # f = IIR(0.9975)
    # f = RunningAverage(1)
    f = Gaussian(50, 20)
    amplitude_fft_real_filtered = f.filter_1d_array(amplitude_fft_amplitude)

    plt.plot(amplitude[:100])
    plt.ylabel(f"Amplitude {name}")
    # plt.show()
    plt.savefig(output_folder + base_filename + '_ampl.png')
    plt.close()

    plt.plot(amplitude_fft_real_filtered)
    plt.ylabel(f"fft: {name}")
    # plt.show()
    plt.savefig(output_folder + base_filename + '_fft.png')
    plt.close()

    abs_dev = [abs(a - avg) for a in amplitude]
    square_dev = [(a - avg) ** 2 for a in amplitude]
    print(f"Average: {avg}")
    print(f"Average absolute deviation: {sum(abs_dev) / len(abs_dev)}")
    print(f"std dev: {math.sqrt(sum(square_dev) / len(square_dev))}")
    max_x = max(enumerate(amplitude_fft_real_filtered), key=lambda x: x[1])[0]
    print(f"x with highest y in fft {max_x}")


class Oscillation:
    def __init__(self, freq, ampl) -> None:
        self.freq = freq
        self.ampl = ampl

    def angle_offsets(self, time):
        return math.sin(time * self.freq) * self.ampl, math.cos(time * self.freq) * self.ampl


# window_size = 20


def into_windows(array, window_len):
    N = len(array)
    output = []
    for i in range(N - window_len + 1):
        output.append(array[i:i + window_len])
    return np.array(output)


def create_train_data(phis, phi_offsets, window_size):
    N = len(phis)
    xs_train = phis[:int(N * 0.7)]
    xs_test = phis[int(N * 0.7):]
    ys_train = phi_offsets[:int(N * 0.7)]
    ys_test = phi_offsets[int(N * 0.7):]

    xs_windowed_train = into_windows(xs_train, window_size)
    xs_windowed_test = into_windows(xs_test, window_size)
    xs_windowed_train /= 90
    xs_windowed_test /= 90

    ys_windowed_train = into_windows(ys_train, window_size)
    ys_windowed_test = into_windows(ys_test, window_size)
    ys_windowed_train /= 90
    ys_windowed_test /= 90

    ys_train_last_in_window = np.array([x[-1] for x in ys_windowed_train])
    ys_test_last_in_window = np.array([x[-1] for x in ys_windowed_test])

    return xs_train, xs_test, ys_train, ys_test, xs_windowed_train, xs_windowed_test, ys_windowed_train, ys_windowed_train, ys_train_last_in_window, ys_test_last_in_window


def get_dense_model(window_size):
    model = Sequential()
    model.add(Dense(256, input_shape=(window_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))  # Dropout helps protect the model from memorizing or "overfitting" the training data

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def rms_error(pred, ys):
    return (sum([float((y - p)**2) for y, p in zip(np.array(ys), np.array(pred))])/len(pred))**0.5



def evaluate(model, xs_windowed_test, ys, base_filename, output_index: OutputIndex, plot_range=200):
    eval_results = model.evaluate(xs_windowed_test, ys, verbose=0)
    pred = model.predict(xs_windowed_test)
    rms_err = rms_error(pred, ys)

    print(f"{xs_windowed_test[0][-1]} {model.predict(xs_windowed_test, verbose=0)}")
    print({k: v for k, v in zip(model.metrics_names, eval_results)})
    print(f"rms err: {rms_err}")

    plt.plot([x[-1] for x in xs_windowed_test[:plot_range]])
    plt.plot(pred[:plot_range])
    plt.plot(ys[:plot_range])
    # plt.show()
    filename = output_folder + base_filename + " raw_target_final.png"
    plt.savefig(filename)
    plt.close()
    output_indexes.append(output_index.with_plot_variant("raw_target_final", filename, rms_err))

    plt.plot(pred[:plot_range])
    plt.plot(ys[:plot_range])
    # plt.show()
    filename = output_folder + base_filename + " target_final.png"
    plt.savefig(filename)
    plt.close()
    output_indexes.append(output_index.with_plot_variant("target_final", filename, rms_err))


def get_rnn_model(window_size):
    model = Sequential()
    # model.add(SimpleRNN((window_size,), input_shape=(window_size,), activation="relu"))
    # model.add(Dense(units=1, activation="relu"))
    model.add(SimpleRNN(32, input_shape=(window_size, 1)))
    # model.add(LSTM(32, input_shape=(window_size,1)))
    # model.add(LSTM(128, input_shape=(window_size,1), return_sequences=True))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def evaluate_with_pred(xs_windowed_test, pred, ys_exp, base_filename, output_index: OutputIndex, plot_range=200):

    rms_err = rms_error(pred, ys_exp)
    print(f"rms err: {rms_err}")


    plt.plot([x[-1] for x in xs_windowed_test[:plot_range]])
    plt.plot(pred[:plot_range])
    plt.plot(ys_exp[:plot_range])
    # plt.show()
    filename = output_folder + base_filename + " raw_target_final.png"
    plt.savefig(filename)
    plt.close()
    output_indexes.append(output_index.with_plot_variant("raw_target_final", filename, rms_err))

    plt.plot(pred[:plot_range])
    plt.plot(ys_exp[:plot_range])
    # plt.show()
    filename = output_folder + base_filename + " target_final.png"
    plt.savefig(filename)
    plt.close()
    output_indexes.append(output_index.with_plot_variant("target_final", filename, rms_err))


# harder data

def run_model(xs_windowed_train, ys_train_last_in_window, xs_windowed_test, ys_test_last_in_window, model, base_filename, output_index: OutputIndex, plot_range=200):
    model.fit(xs_windowed_train, ys_train_last_in_window,
              batch_size=1024, epochs=128,
              verbose=0,
              validation_data=(xs_windowed_test, ys_test_last_in_window))
    evaluate(model, xs_windowed_test, ys_test_last_in_window, base_filename, output_index, plot_range)


def run_whole_esn(xs_windowed_train, ys_train_last_in_window, xs_windowed_test, ys_test_last_in_window, description, window_size, output_index: OutputIndex, plot_range=200):
    activation = tanh(leaky_rate=0.9)

    esn = DeepESN(
        input_size=window_size,
        num_layers=1,
        hidden_size=500,
        activation=activation, )
    xs_windowed_train_torch = torch.from_numpy(xs_windowed_train[:, :])
    ys_train_last_in_window_torch = torch.from_numpy(ys_train_last_in_window).reshape((-1, 1))
    esn.fit(xs_windowed_train_torch, ys_train_last_in_window_torch)

    xs_windowed_test_torch = torch.from_numpy(xs_windowed_test)
    ys_test_last_in_window_torch = torch.from_numpy(ys_test_last_in_window).reshape((-1, 1))
    output = esn(xs_windowed_test_torch)

    evaluate_with_pred(xs_windowed_test_torch, output, ys_test_last_in_window_torch, description, output_index, plot_range=plot_range)


def run_models(data, oscillation, description, output_index: OutputIndex, window_size, plot_range=200):
    thetas, phis, theta_offsets, phi_offsets = data[1](data[0], oscillation)
    xs_train, xs_test, ys_train, ys_test, xs_windowed_train, xs_windowed_test, ys_windowed_train, ys_windowed_train, ys_train_last_in_window, ys_test_last_in_window = \
        create_train_data(phis, phi_offsets, window_size)

    run_model(xs_windowed_train, ys_train_last_in_window, xs_windowed_test, ys_test_last_in_window, get_dense_model(window_size), description + " dense", output_index.with_nn_type("dense"), plot_range=plot_range)
    run_model(xs_windowed_train, ys_train_last_in_window, xs_windowed_test, ys_test_last_in_window, get_rnn_model(window_size), description + " rnn", output_index.with_nn_type("rnn"), plot_range=plot_range)
    run_whole_esn(xs_windowed_train, ys_train_last_in_window, xs_windowed_test, ys_test_last_in_window, description + " esn", window_size, output_index.with_nn_type("esn"), plot_range=plot_range)



# for (data, data_name) in [(data, "data"), (data2, "data2"), (data3, "data3")]:
#     for (oscillation, freq_name) in [(Oscillation(0.5, 1.5), "base"), (Oscillation(0.05, 1.5), "low freq"), (Oscillation(5, 1.5), "high freq")]:
#         for window_size in [5, 10, 20, 40, 80]:
#             for i in range(5):
#                 run_models(data, oscillation, data_name + " " + freq_name, OutputIndex(data_name, freq_name, str(window_size)), window_size=window_size)


for name, data in data_pquil.items():
    for window_size in [5, 10, 20, 40, 80]:
        for i in range(1):
            print(name, window_size, i)
            run_models(data, None, name, OutputIndex(name, "", str(window_size)), window_size=window_size)


print([x.__dict__ for x in output_indexes])

# result = {key:list(v[0] for v in valuesiter)
#           for key,valuesiter in groupby(input, key=lambda x: x.data)}
# print(result)
import json
serialized  = json.dumps(output_indexes, default=vars)
f  = open("aa.json", "w")
f.write(serialized)
f.close()