cd ../src

HOSTNAME="localhost"
PORT="8028"

# Change the model name to the model you want to evaluate
EVAL_MODEL="AURA"
MODEL_PATH="aurateam/AURA"
CHUNKED_1S_DIR="./data/chunked_1s_videos"

# -1 means all context, i. e. (0, query_time); any integer t greater than 0 means (query_time - t, query_time)
CONTEXT_TIME=-1

AURA_ARGS="--model_name $EVAL_MODEL --base_url http://$HOSTNAME:$PORT/v1 --model_path $MODEL_PATH --chunked_1s_dir $CHUNKED_1S_DIR --context_time $CONTEXT_TIME"

# real-time visual understanding
TASK="real"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="Streaming"
python eval.py  --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE $AURA_ARGS

# omni-source understanding
TASK="omni"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="Streaming"
python eval.py --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE $AURA_ARGS

# sequential question answering
TASK="sqa"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="StreamingSQA"
python eval.py --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE $AURA_ARGS

# proactive output
TASK="proactive"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="StreamingProactive"
python eval.py --benchmark_name $BENCHMARK --data_file $DATA_FILE --output_file $OUTPUT_FILE $AURA_ARGS
