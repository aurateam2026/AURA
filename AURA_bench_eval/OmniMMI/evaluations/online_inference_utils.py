import os
import re
import json
import math
import time
import random
import subprocess  
from tqdm import tqdm

import numpy as np
import torch
import torchvision
# torchvision.set_video_backend('video_reader')
import torch.multiprocessing as mp
import transformers
logger = transformers.logging.get_logger('liveinfer')
from QWen3VL_online import EvalQWen3VL_online


def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_random_chunk(lst, n, k, seed=42):
    lst = lst[:]
    random.seed(seed)
    random.shuffle(lst)
    chunks = split_list(lst, n)
    return chunks[k]

    
def online_inference(
    model,
    model_name,
    benchmark_name,
    questions_file,
    num_chunks,
    chunk_id,
    video_dir,
    output_dir,
    seed=42,
):
    setup_seed(seed)
    

    questions = json.load(open(questions_file))
    # questions = get_chunk(questions, num_chunks, chunk_id)
    questions = get_random_chunk(questions, num_chunks, chunk_id)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if "qwen" in model_name.lower():
    #     preds_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}_{chunk_id}_qwen.json")
    # else:
    #     preds_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}_{chunk_id}_llama.json")
    preds_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}_{chunk_id}.json")
    pred_file = open(preds_file, "w")

    skipped_samples = []
    debug_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}_{chunk_id}_debug_skipped.json")

    for sample in tqdm(questions):

        # action prediction
        if benchmark_name == "ap":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            
            src_video_path = video_path
            question = sample["question"]
            vocab = sample["vocab"]
            question = f"{question}, Please select the next step from the provided vocabulary list: {vocab}."
            
            if isinstance(model, EvalQWen3VL_online):
                model.load_video(src_video_path)
                output = model.inference(src_video_path, question, question_time=max(0, int(model.video_duration * model.frame_fps) - 5))
            else:
                name, ext = os.path.splitext(src_video_path)
                ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
                if not os.path.exists(ffmpeg_video_path):
                    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                    logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
                model.load_video(ffmpeg_video_path)
                output = ""
                history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
                duration = model.video_duration
                for i in range(int(duration * model.frame_fps)):
                    if i == max(0, int(duration * model.frame_fps) - 5):
                        model.input_query_stream(question)
                    model.input_video_stream(i / model.frame_fps)
                    query, response, _ = model()
                    if query:
                        history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                    if response:
                        history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                        output = response
                        break
                model.reset()

            sample["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")
            
        # speaker identification
        elif benchmark_name == "si":
            vid = sample["video"]
            timestamp = sample["timestamp"]
            video_path = os.path.join(video_dir, "videos", vid)
            src_video_path = video_path
            question = sample["question"]

            if isinstance(model, EvalQWen3VL_online):
                model.load_video(src_video_path)
                question_time = int(timestamp * model.frame_fps)
                if question_time > model.video_duration:
                    print(f"[SKIP] si: video={vid}, question_time={question_time} > video_duration={model.video_duration}")
                    skipped_samples.append({
                        "benchmark": "si", "video": vid,
                        "question_time": question_time,
                        "video_duration": model.video_duration,
                        "reason": "question_time exceeds video duration",
                        "sample": sample,
                    })
                    sample["pred"] = ""
                    pred_file.write(json.dumps(sample) + "\n")
                    continue
                try:
                    output = model.inference(src_video_path, question, question_time=question_time)
                except Exception as e:
                    print(f"[SKIP] si: video={vid}, question_time={question_time}, error={e}")
                    skipped_samples.append({
                        "benchmark": "si", "video": vid,
                        "question_time": question_time,
                        "video_duration": model.video_duration,
                        "reason": f"inference error: {e}",
                        "sample": sample,
                    })
                    sample["pred"] = ""
                    pred_file.write(json.dumps(sample) + "\n")
                    continue
            else:
                name, ext = os.path.splitext(src_video_path)
                ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
                save_history_path = src_video_path.replace('.mp4', '.json')
                if not os.path.exists(ffmpeg_video_path):
                    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                    logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
                model.load_video(ffmpeg_video_path)
                
                output = ""
                history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
                duration = model.video_duration
                num_frames = int(duration * model.frame_fps)
                for i in range(num_frames):
                    if i == int(timestamp * model.frame_fps):
                        model.input_query_stream(question)
                    model.input_video_stream(i / model.frame_fps)
                    query, response, _ = model()
                    if query:
                        history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                    if response:
                        history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                        output = response
                        break
                model.reset()
            sample["pred"] = output
            pred_file.write(json.dumps(sample)+"\n")
            
        # multiturn dependency
        elif benchmark_name == "md":
            
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            src_video_path = video_path

            if isinstance(model, EvalQWen3VL_online):
                model.load_video(src_video_path)
                md_skipped_any = False
                for idx in range(len(sample["qa"])):
                    s, e = map(float, sample["qa"][idx]["timestamp"].split("--"))
                    question_time = math.ceil((s+e)/2 * model.frame_fps)
                    if question_time > model.video_duration:
                        print(f"[SKIP] md: video={vid}, qa_idx={idx}, question_time={question_time} > video_duration={model.video_duration}")
                        if not md_skipped_any:
                            skipped_samples.append({
                                "benchmark": "md", "video": vid,
                                "video_duration": model.video_duration,
                                "reason": "question_time exceeds video duration",
                                "sample": sample,
                            })
                            md_skipped_any = True
                        sample["qa"][idx]["pred"] = ""
                        continue
                    question = sample["qa"][idx]["question"]
                    if idx != 0 and "pred" in sample["qa"][idx-1]:
                        question = question.replace("##ANSWER##", sample["qa"][idx-1]["pred"])
                    try:
                        output = model.inference(src_video_path, question, question_time=question_time)
                    except Exception as e:
                        print(f"[SKIP] md: video={vid}, qa_idx={idx}, question_time={question_time}, error={e}")
                        if not md_skipped_any:
                            skipped_samples.append({
                                "benchmark": "md", "video": vid,
                                "video_duration": model.video_duration,
                                "question_time": question_time,
                                "reason": f"inference error: {e}",
                                "sample": sample,
                            })
                            md_skipped_any = True
                        sample["qa"][idx]["pred"] = ""
                        continue
                    sample["qa"][idx]["pred"] = output
            else:
                name, ext = os.path.splitext(src_video_path)
                ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
                save_history_path = src_video_path.replace('.mp4', '.json')
                if not os.path.exists(ffmpeg_video_path):
                    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                    logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
                model.load_video(ffmpeg_video_path)
            
                output = ""
                history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
                duration = model.video_duration
                num_frames = int(duration * model.frame_fps)
                query_idx = 0
                answer_idx = 0
                for i in range(num_frames):
                    if query_idx < len(sample["qa"]):
                        qa = sample["qa"][query_idx]
                        s, e = map(float, qa["timestamp"].split("--"))
                        if (s+e)/2 * model.frame_fps <= i:
                            question = qa["question"]
                            if query_idx != 0 and "pred" in sample["qa"][answer_idx]:
                                question = question.replace("##ANSWER##", sample["qa"][answer_idx]["pred"])
                            model.input_query_stream(question)
                            query_idx += 1
                        
                    model.input_video_stream(i / model.frame_fps)
                    query, response, _ = model()
                    if query:
                        history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                    if response:
                        history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                        output = response
                        if answer_idx < query_idx:
                            sample["qa"][answer_idx]["pred"] = output
                            answer_idx += 1
                    if answer_idx >= len(sample["qa"]): break
                model.reset()
            pred_file.write(json.dumps(sample)+"\n")
        
        # dynamic state grounding
        elif benchmark_name == "sg":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            src_video_path = video_path

            if isinstance(model, EvalQWen3VL_online):
                model.load_video(src_video_path)
                sg_skipped_any = False
                for idx in range(len(sample["qa"])):
                    s, e = map(float, sample["qa"][idx]["timestamp"].split("--"))
                    question_time = math.ceil((s+e)/2 * model.frame_fps)
                    if question_time > model.video_duration:
                        print(f"[SKIP] sg: video={vid}, qa_idx={idx}, question_time={question_time} > video_duration={model.video_duration}")
                        if not sg_skipped_any:
                            skipped_samples.append({
                                "benchmark": "sg", "video": vid,
                                "video_duration": model.video_duration,
                                "reason": "question_time exceeds video duration",
                                "sample": sample,
                            })
                            sg_skipped_any = True
                        sample["qa"][idx]["pred"] = ""
                        continue
                    question = sample["qa"][idx]["question"]
                    try:
                        output = model.inference(src_video_path, question, question_time=question_time)
                    except Exception as e:
                        print(f"[SKIP] sg: video={vid}, qa_idx={idx}, question_time={question_time}, error={e}")
                        if not sg_skipped_any:
                            skipped_samples.append({
                                "benchmark": "sg", "video": vid,
                                "video_duration": model.video_duration,
                                "question_time": question_time,
                                "reason": f"inference error: {e}",
                                "sample": sample,
                            })
                            sg_skipped_any = True
                        sample["qa"][idx]["pred"] = ""
                        continue
                    sample["qa"][idx]["pred"] = output
            else:
                name, ext = os.path.splitext(src_video_path)
                ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
                save_history_path = src_video_path.replace('.mp4', '.json')
                if not os.path.exists(ffmpeg_video_path):
                    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                    logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
                model.load_video(ffmpeg_video_path)
                
                output = ""
                history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
                duration = model.video_duration
                num_frames = int(duration * model.frame_fps)
                query_idx = 0
                answer_idx = 0
                for i in range(num_frames):
                    if query_idx < len(sample["qa"]):
                        qa = sample["qa"][query_idx]
                        s, e = map(float, qa["timestamp"].split("--"))
                        if (s+e)/2 * model.frame_fps <= i:
                            question = qa["question"]
                            model.input_query_stream(question)
                            query_idx += 1
                        
                    model.input_video_stream(i / model.frame_fps)
                    if query:
                        history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                    if response:
                        history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                        output = response
                        if answer_idx < query_idx:
                            sample["qa"][answer_idx]["pred"] = output
                            answer_idx += 1
                    if answer_idx >= len(sample["qa"]): break
                model.reset()
            
            pred_file.write(json.dumps(sample)+"\n")
        
        # proactive alerting
        elif benchmark_name == "pa":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            src_video_path = video_path
            question = sample["question"]
            start, end = sample["answer"]
            

            if isinstance(model, EvalQWen3VL_online):
                model.load_video(src_video_path)
                end = min(end, int(model.video_duration))

                for i in range(1, end+1):
                    output = model.inference_PA(src_video_path, question, question_time=i)
                    if output != '':
                        sample["pred"] = [i]
                        break

            else:
                name, ext = os.path.splitext(src_video_path)
                ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
                save_history_path = src_video_path.replace('.mp4', '.json')
                if not os.path.exists(ffmpeg_video_path):
                    os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                    ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                    logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
                model.load_video(ffmpeg_video_path)
                
                query = sample["question"]
                sample["pred"] = []
                model.input_query_stream(query)
                history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
                duration = model.video_duration
                num_frames = int(duration * model.frame_fps)
                
                for i in range(num_frames):
                    model.input_video_stream(i / model.frame_fps)
                    oom = False
                    try:
                        query, response, response_time = model()
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"OOM at {vid}")
                        oom = True
                    if oom:
                        model.reset()
                        torch.cuda.empty_cache()
                        break
                        
                    if query:
                        history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                    if response:
                        history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                        if int(response_time) != 0:
                            sample["pred"].append(response_time)
                model.reset()
                torch.cuda.empty_cache()

            if "pred" not in sample:
                sample["pred"] = [0]
            pred_file.write(json.dumps(sample)+"\n")
        
        # proactive turntaking
        elif benchmark_name == "pt":
            vid = sample["video"]
            video_path = os.path.join(video_dir, "videos", vid)
            src_video_path = video_path
            name, ext = os.path.splitext(src_video_path)
            ffmpeg_video_path = os.path.join('cache', name + f'_{model.frame_fps}fps_{model.frame_resolution}' + ext)
            # save_history_path = src_video_path.replace('.mp4', '.json')
            if not os.path.exists(ffmpeg_video_path):
                os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                ffmpeg_once(src_video_path, ffmpeg_video_path, fps=model.frame_fps, resolution=model.frame_resolution)
                logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {model.frame_fps} FPS, {model.frame_resolution} Resolution')
            model.load_video(ffmpeg_video_path)
            # model.load_video(src_video_path)
            
            # liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
            convs = sample["conversation"]
            query = ""
            for conv in convs:
                for k, v in conv.items():
                    query += f"{k}\n{v}\n"
            history = {'video_path': src_video_path, 'frame_fps': model.frame_fps, 'conversation': []}
            duration =  model.video_duration
            num_frames = int(duration * model.frame_fps)
            for i in range(num_frames):
                if i == num_chunks - 5:
                    model.input_query_stream(query)
                    query_time = model.video_time
                model.input_video_stream(i / model.frame_fps)
                query, response, response_time = model()
                if query:
                    history['conversation'].append({'role': 'user', 'content': query, 'time': model.video_time})
                if response:
                    history['conversation'].append({'role': 'assistant', 'content': response, 'time': model.video_time})
                    if response_time == query_time:
                        sample["pred"] = 0
                        break
            if "pred" not in sample: sample["pred"] = 1
            pred_file.write(json.dumps(sample)+"\n")
            model.reset()
        
        else:
            raise TypeError(f"INVALID benchmark_name: {benchmark_name}, please select from [plan, speaker, dependency, transition]")

    pred_file.close()

    if skipped_samples:
        with open(debug_file, "w") as f:
            json.dump(skipped_samples, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] {len(skipped_samples)} skipped sample(s) written to {debug_file}")
    

def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        'ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)