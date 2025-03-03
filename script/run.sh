FILE=$1
OUTPUT_FILE=${FILE%.jsonl}_detected.jsonl
DEVICE_1=$2
DEVICE_2=$3

python3 eval.py \
    --input $FILE \
    --device-1 $DEVICE_1 \
    --device-2 $DEVICE_2 \
    --batch-size 8 \
    --max-token-observed 2048 \
    --output $OUTPUT_FILE
