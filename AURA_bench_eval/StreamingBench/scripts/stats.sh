cd ../src/data

# python count.py --model "<model_name>" --task "<real/omni/sqa/proactive>" --src "<output_file>"

python count.py --model "AURA" --task "real" --src "real_output_AURA.json"
python count.py --model "AURA" --task "omni" --src "omni_output_AURA.json"
python count.py --model "AURA" --task "sqa" --src "sqa_output_AURA.json"
python count.py --model "AURA" --task "proactive" --src "proactive_output_AURA.json"
