from __future__ import absolute_import, division, print_function, unicode_literals

import os
import subprocess
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


def create_blobs_if_not_existed(blob_names):
    existd_names = set(workspace.Blobs())
    for xx in blob_names:
        if xx not in existd_names:
            workspace.CreateBlob(str(xx))


def load_model_pb(net_file, init_file):
    subprocess.check_call(["adb", "push", net_file, "/data/local/tmp/predict_net.pb"])
    subprocess.check_call(["adb", "push", init_file, "/data/local/tmp/init_net.pb"])

    net = caffe2_pb2.NetDef()
    if net_file is not None:
        net.ParseFromString(open(net_file, "rb").read())
    return (net, None)
