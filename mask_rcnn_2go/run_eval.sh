MODEL_SUFFIX="$1"
DATA_SUFFIX="$2"
DATASET_IM_DIR="/tmp/val2014"
DATASET_ANN="val2014/instances_minival2014$DATA_SUFFIX.json"

python code/eval_seg_cpu.py \
    --net "model/int8$MODEL_SUFFIX/model-nnapi.pb" \
    --init_net "model/int8$MODEL_SUFFIX/model_init.pb" \
    --dataset "coco_2014_minival$DATA_SUFFIX" \
    --dataset_dir "$DATASET_IM_DIR" \
    --dataset_ann "$DATASET_ANN" \
    --output_dir "output$MODEL_SUFFIX" \
    --min_size 320 \
    --max_size 640 \
