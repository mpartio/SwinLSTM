import numpy as np
import glob
from datetime import datetime, timedelta
from enum import Enum
import copy
import torch

from torch.utils.data import IterableDataset


N_HIST = 4
N_PRED = 16


def create_squeezed_leadtime_conditioning(img_size, depth, active_leadtime):
    return np.expand_dims(
        np.full(img_size, active_leadtime / depth), axis=(0, 3)
    ).astype(np.float32)


def read_times_from_preformatted_files_directory(dirname):
    toc = {}
    for f in glob.glob("{}/*-times.npy".format(dirname)):
        times = np.load(f)

        for i, t in enumerate(times):
            toc[t] = {"filename": f.replace("-times", ""), "index": i, "time": t}

    times = list(toc.keys())

    times.sort()
    print("Read {} times from {}".format(len(times), dirname))
    return times, toc


def read_datas_from_preformatted_files_directory(dirname, toc, times):
    datas = []
    print("Reading data for {}".format(times))
    for t in times:
        e = toc[t]
        idx = e["index"]
        filename = e["filename"]
        datafile = np.load(filename, mmap_mode="r")
        datas.append(datafile[idx])

    return datas, times


def read_times_from_preformatted_file(filename):
    ds = np.load(filename)
    data = ds["arr_0"]
    times = ds["arr_1"]

    toc = {}
    for i, t in enumerate(times):
        toc[t] = {"index": i, "time": t}

    print("Read {} times from {}".format(len(times), filename))

    return times, data, toc


def read_datas_from_preformatted_file(all_times, all_data, req_times, toc):
    datas = []
    for t in req_times:
        index = toc[t]["index"]
        datas.append(all_data[index])

    return datas, req_times


class SAFDataGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        assert len(self.placeholder) > 0
        print(
            "Generator number of batches: {} batch size: {}".format(
                len(self), self.batch_size
            )
        )

    def __len__(self):
        """Return number of batches in this dataset"""
        return len(self.placeholder) // self.batch_size

    def __getitem__(self, idx):
        # placeholder X elements:
        # 0.. N_HIST: history of actual data (YYYYMMDDTHHMMSS, string)
        # N_HIST + 1: include datetime (bool)
        # N_HIST + 2: include topography (bool)
        # N_HIST + 3: include terrain type (bool)
        # N_HIST + 4: include sun elevation angle (bool)

        ph = self.placeholder[idx]

        X = ph[0]
        Y = ph[1]

        x_hist = X[0 : N_HIST]

        x, y, xtimes, ytimes = self.get_xy(x_hist, Y)

        x = np.asarray(x)
        y = np.asarray(y)

        assert np.max(x) < 1.01, "x max: {:.2f}".format(np.max(x))

        ts = datetime.strptime(xtimes[-1], "%Y%m%dT%H%M%S")  # "analysis time"

        x = np.squeeze(x)
        y = np.squeeze(y, axis=-1)

        if N_PRED > 1:
            y = np.squeeze(y)

        assert x.shape == (N_HIST, 128, 128), "{}".format(x.shape)
        assert y.shape == (N_PRED, 128, 128), "{}".format(y.shape)

        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)

        x = torch.from_numpy(x).contiguous()
        y = torch.from_numpy(y).contiguous()
        return (x, y)

    def get_xy(self, x_elems, y_elems):
        xtimes = []
        ytimes = []

        if self.dataseries_file is not None:
            x, xtimes = read_datas_from_preformatted_file(
                self.elements, self.data, x_elems, self.toc
            )
            y, ytimes = read_datas_from_preformatted_file(
                self.elements, self.data, y_elems, self.toc
            )

        return x, y, xtimes, ytimes

    def __call__(self):
        for i in range(len(self.placeholder)):
            elem = self.__getitem__(i)
            yield elem


class SAFDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        a = self.generator()
        return a
        # x, y = self.generator()
        return (x, y)
        # return self.generator()

    def __len__(self):
        return len(self.generator)

    def flip(self, x, y):
        n = 4
        # flip first n dimensions as they contain the payload data
        # concatenate the flipped data with the rest
        x = tf.concat([tf.image.flip_up_down(x[..., 0:n]), x[..., n:]], axis=-1)
        y = tf.image.flip_up_down(y)
        return (x, y)

    def normalize(x, y):
        # normalize input data (x) to 0..1
        # scale output data to 0..1
        # mean, variance = tf.nn.moments(x, axes=[0, 1], keepdims=True)
        # x = (x - mean) / tf.sqrt(variance + tf.keras.backend.epsilon())

        if tf.math.reduce_max(x[..., 0]) <= 1.01:
            return (x, y)

        # scale all data to 0..1, to preserve compatibility with older models
        # trained with this software
        x = tf.concat([0.01 * x[..., 0:n], x[..., n:]], axis=-1)
        y = y * 0.01
        return (x, y, t)


class SAFDataLoader:
    def __init__(self, **kwargs):
        self.img_size = (128, 128)
        self.include_datetime = False
        self.include_topography = False
        self.include_terrain_type = False
        self.include_sun_elevation_angle = False
        self.dataseries_file = "/tmp/nwcsaf-effective-cloudiness-20190801-20200801-img_size=128x128-float32.npz"

        self.batch_size = int(kwargs.get("batch_size", 8))

        self._placeholder = []

        self.initialize()

    def initialize(self):
        # Read static datas, so that each dataset generator
        # does not have to read them

        if self.include_topography:
            self.topography_data = np.expand_dims(
                create_topography_data(self.img_size), axis=0
            )

        if self.include_terrain_type:
            self.terrain_type_data = np.expand_dims(
                create_terrain_type_data(self.img_size), axis=0
            )

        if self.include_sun_elevation_angle:
            if self.operating_mode == OpMode.INFER:
                self.sun_elevation_angle_data = {}
                for i in range(N_PRED):
                    ts = self.analysis_time + timedelta(minutes=(1 + i) * 15)
                    self.sun_elevation_angle_data[
                        ts.strftime("%Y%m%dT%H%M%S")
                    ] = create_sun_elevation_angle(ts, (128, 128))
            else:
                self.sun_elevation_angle_data = create_sun_elevation_angle_data(
                    self.img_size,
                )

        # create placeholder data

        if self.dataseries_file is not None:
            self.elements, self.data, self.toc = read_times_from_preformatted_file(
                self.dataseries_file
            )

        elif self.dataseries_directory is not None:
            self.elements, self.toc = read_times_from_preformatted_files_directory(
                self.dataseries_directory
            )

        i = 0

        step = N_HIST + N_PRED
        n_fut = N_PRED

        assert (
            len(self.elements) - step
        ) >= 0, "Too few data to make a prediction: {} (need at least {})".format(
            len(self.elements), step
        )

        while i <= len(self.elements) - step:
            x = list(self.elements[i : i + N_HIST])
            y = list(self.elements[i + N_HIST : i + step])

            self._placeholder.append([x, y])

            i += step

        assert len(self._placeholder) > 0, "Placeholder array is empty"

        print(
            "Placeholder timeseries length: {} number of samples: {} sample length: x={},y={}".format(
                len(self.elements),
                len(self._placeholder),
                len(self._placeholder[0][0]),
                len(self._placeholder[0][1]),
            )
        )

        np.random.shuffle(self._placeholder)

    def __len__(self):
        """Return number of samples"""
        return len(self._placeholder)

    def get_dataset(self, take_ratio=None, skip_ratio=None):
        placeholder = None

        if take_ratio is not None:
            l = int(len(self._placeholder) * take_ratio)
            placeholder = self._placeholder[0:l]

        if skip_ratio is not None:
            l = int(len(self._placeholder) * skip_ratio)
            placeholder = self._placeholder[l:]

        if placeholder is None:
            placeholder = copy.deepcopy(self._placeholder)

        assert len(placeholder) > 0

        gen = SAFDataGenerator(placeholder=placeholder, **self.__dict__)

        dataset = SAFDataset(gen)

        return dataset
