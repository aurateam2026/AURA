#!/bin/bash
# TTS 独立服务启动脚本
# GPU 由上游 start_all.sh 通过 CUDA_VISIBLE_DEVICES 指定

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u tts_service.py \
    --port 8002 \
    --gpu 0 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --language Chinese \
    --ref-audio shuhan2.mp3 \
    --ref-text "你的笑容好甜啊，像糖果一样融化在我的心里。哎呀，你怎么突然这么说，我都不好意思了。有你在身边，我觉得好安心，你是我的避风港。"
    # --ref-audio zhengtai.mp3 \
    # --ref-text "从前在一个遥远的王国里住着一对国王夫妇，这个王国中的皇后是一位善良又美丽的人。有一次，王国正在处于严冬时节的时分。皇后正坐在王宫里的一扇窗子，她一边欣赏在外面的雪景一边用象牙做个针刺绣为自己将要出生的孩子做一个衣服。 皇后注意到寒风卷着雪片飘进乌木窗的窗子台上飘落不少雪花，她抬头向窗外望去十一不留神使针刺进自己的手指，皇后注意到她手上的鲜血从针口流出来中有三点血滴落在飘进窗子的雪花上，她若有所思地凝视着点缀在白雪上的鲜红血滴又看着窗外的雪花慢慢落下，她心想她希望自己将来要出生的孩子拥有像雪一样白的肤色、像血一样红的嘴唇和一头像乌木窗框的颜色的头发。"