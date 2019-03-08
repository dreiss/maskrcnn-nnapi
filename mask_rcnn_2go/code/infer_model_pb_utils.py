from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
import os
import tempfile
import subprocess

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import utils2
from caffe2.python import core, workspace, utils
from caffe2.proto import caffe2_pb2


FORMAT = "%(levelname)s %(filename)s:%(lineno)4d: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


PIXEL_MEANS_DEFAULT = np.array([[[0.0, 0.0, 0.0]]])
PIXEL_STDS_DEFAULT = np.array([[[1.0, 1.0, 1.0]]])

LIMIT = 100
CLASSES = 81
RES = 12


def run_single_segms(
    net,
    image,
    target_size,
    pixel_means=PIXEL_MEANS_DEFAULT,
    pixel_stds=PIXEL_STDS_DEFAULT,
    rle_encode=True,
    max_size=1333,
):
    inputs = utils2.prepare_blobs(
        image,
        target_size=target_size,
        max_size=max_size,
        pixel_means=pixel_means,
        pixel_stds=pixel_stds,
    )

    # Prepare inputs for AABB and Int8AABB operators
    im_info = inputs["im_info"]
    scale = im_info[0][2]
    inputs["im_infoq"] = np.rint(im_info[:,:2] * 8.0).astype(np.uint16)
    inputs["im_info2"] = im_info[:,:2]

    blob_names = []
    ser_blobs = []

    # Serialize inputs for remote device
    for k, v in inputs.items():
        workspace.FeedBlob(k, v)
        blob_names.append(k)
        ser_blobs.append(workspace.SerializeBlob(k))

    # Serialize output templates for remote device
    fully_quantized = any(op.type == "Int8AABBRoIProposals" for op in net.op)
    bbox_type = np.uint16 if fully_quantized else np.float32
    output_templates = {
            "score_nms": np.zeros((LIMIT,), np.float32),
            "bbox_nms": np.zeros((LIMIT, 4), bbox_type),
            "class_nms": np.zeros((LIMIT,), np.int32),
            "mask_fcn_probs": np.zeros((LIMIT, CLASSES, RES, RES), np.float32),
            }
    for out_name in net.external_output:
        fake_name = out_name + "_empty_template"
        blob_names.append(out_name)
        workspace.FeedBlob(fake_name, output_templates[out_name])
        ser_blobs.append(workspace.SerializeBlob(fake_name))

    # Package inputs and output templates
    inout_netdef = caffe2_pb2.NetDef()
    inout_netdef.arg.extend([
            utils.MakeArgument("blob_names", blob_names),
            utils.MakeArgument("ser_blobs", ser_blobs),
        ])

    # Send in/out to the remote device
    with tempfile.NamedTemporaryFile() as inout_file:
        inout_file.write(inout_netdef.SerializeToString())
        inout_file.flush()
        subprocess.check_call(["adb", "push", inout_file.name, "/data/local/tmp/input_output.pb"])

    try:
        # Run the model
        use_caffe2 = "--use_caffe2_reference true" if os.environ.get("USE_CAFFE2_REFERENCE") in ("1", "true", "yes", "on") else ""
        subprocess.check_call("adb shell 'cd /data/local/tmp ; GLOG_logtostderr=true GLOG_v=0 ./nnapi_runner %s --init_net init_net.pb --predict_net predict_net.pb --inout_net input_output.pb --out_path output_blobs.pb'" % use_caffe2, shell=True)

        # Retrieve and deserialize outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output_blobs.pb")
            subprocess.check_call(["adb", "pull", "/data/local/tmp/output_blobs.pb", output_file])

            out_net = caffe2_pb2.NetDef()
            with open(output_file, "rb") as handle:
                out_net.ParseFromString(handle.read())

        all_outputs = utils.ArgsToDict(out_net.arg)["outputs"]
        for output in all_outputs:
            bp = caffe2_pb2.BlobProto()
            bp.ParseFromString(output)
            workspace.DeserializeBlob(bp.name, output)

        classids = workspace.FetchBlob("class_nms")
        scores = workspace.FetchBlob("score_nms")  # bbox scores, (R, )
        boxes = workspace.FetchBlob("bbox_nms")  # i.e., boxes, (R, 4*1)
        masks = workspace.FetchBlob("mask_fcn_probs")  # (R, cls, mask_dim, mask_dim)
        if boxes.dtype == np.uint16:
            boxes = boxes.astype(np.float32) * 0.125
            boxes /= scale
    except Exception as e:
        print(e)
        # may not detect anything at all
        R = 0
        scores = np.zeros((R,), dtype=np.float32)
        boxes = np.zeros((R, 4), dtype=np.float32)
        classids = np.zeros((R,), dtype=np.float32)
        masks = np.zeros((R, 1, 1, 1), dtype=np.float32)

    # included in the model
    # scale = inputs["im_info"][0][2]
    # boxes /= scale

    R = boxes.shape[0]
    im_masks = []
    if R > 0:
        im_dims = image.shape
        im_masks = utils2.compute_segm_results(
            masks, boxes, classids, im_dims[0], im_dims[1], rle_encode=rle_encode
        )

    boxes = np.column_stack((boxes, scores))

    ret = {"classids": classids, "boxes": boxes, "masks": masks, "im_masks": im_masks}
    return ret
