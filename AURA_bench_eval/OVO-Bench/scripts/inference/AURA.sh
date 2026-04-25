HOSTNAME="localhost"
PORT="8028"

python inference.py \
    --mode offline \
    --model AURA \
    --chunked_dir data/chunked_videos \
    --chunked_1s_dir data/chunked_1s_videos \
    --base_url "http://$HOSTNAME:$PORT/v1"
