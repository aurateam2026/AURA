"""
Realtime Video Audio Capture Client - Streaming Version

支持 Qwen3_VL_online_streaming.py 的流式输入/输出协议。

主要改动:
1. 添加 Type 8 (STREAMING_TOKEN_TYPE) 处理流式 token
2. 添加 streaming_token_queue 队列
3. 添加 /api/poll_streaming_token 接口
4. 保持与原版的向后兼容

协议定义:
- Type 1: VIDEO (C->S)
- Type 2: AUDIO (C->S)
- Type 3: RESPONSE - 完整文本响应 (S->C)
- Type 4: CLEAR_CONTEXT (C->S)
- Type 5: TTS_AUDIO (S->C) - 保留，Streaming 版本暂不使用
- Type 6: START_CAMERA (C->S)
- Type 7: ERROR (S->C) - 服务端错误
- Type 8: STREAMING_TOKEN (S->C) - 流式 token (新增)
"""

import struct
import socket
import threading
import time
import queue
from collections import deque
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 服务端配置
SERVER_HOST = 'hk01dgx050'
SERVER_PORT = 12345

# 视频发送间隔（秒） - 改为连续发送
VIDEO_SEND_INTERVAL = 1.0
# VIDEO_BUFFER_DURATION = 2.0  # 缓存最近 2 秒的视频（共约 4 帧）
# video_buffer = deque()  # 存储 (timestamp, data)
last_video_send_time = 0

# 协议类型
VIDEO_TYPE = b'\x01'
AUDIO_TYPE = b'\x02'
RESPONSE_TYPE = 3  # Integer for unpack check (完整文本响应)
CLEAR_CONTEXT_TYPE = b'\x04'  # 清空上下文
TTS_AUDIO_TYPE = 5  # Integer for unpack check (TTS 音频响应 - WAV)
START_CAMERA_TYPE = b'\x06'  # 开启摄像头（清理文件夹）
ERROR_TYPE = 7  # 服务器错误/拒绝消息
STREAMING_TOKEN_TYPE = 8  # 流式 token
TTS_AUDIO_CHUNK_TYPE = 9  # TTS 音频 chunk (Raw PCM int16) - Step 2 新增

# 全局socket连接
socket_lock = threading.Lock()
global_socket = None
last_connect_attempt = 0
connect_retry_interval = 5
connection_error_logged = False

# 响应队列，用于存储服务端返回的消息
response_queue = queue.Queue()
# TTS 音频队列，用于存储服务端返回的 TTS 音频 (WAV - Type 5)
tts_audio_queue = queue.Queue()
# TTS 音频 chunk 队列 (Raw PCM - Type 9) - Step 2 新增
tts_audio_chunk_queue = queue.Queue()
# 错误队列，用于存储服务端返回的错误消息（如会话被占用）
error_queue = queue.Queue()
# 流式 token 队列
streaming_token_queue = queue.Queue()

# 用户会话锁：确保同时只有一个浏览器用户可以使用系统
user_session_lock = threading.Lock()
current_user_session = None  # 当前占用会话的用户 session_id

def verify_session(session_id: str) -> bool:
    """验证 session_id 是否是当前活动会话。"""
    with user_session_lock:
        return current_user_session is not None and current_user_session == session_id

def get_session_error_response():
    """返回会话验证失败的标准错误响应。"""
    return jsonify({
        'success': False,
        'error': '会话无效或已过期，请重新开启摄像头'
    }), 401

def get_socket():
    """获取或创建socket连接"""
    global global_socket, last_connect_attempt, connection_error_logged
    
    current_time = time.time()
    
    with socket_lock:
        if global_socket is not None:
            return global_socket
        
        # 检查重试间隔
        if current_time - last_connect_attempt < connect_retry_interval:
            return None
        
        last_connect_attempt = current_time
        
        try:
            global_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            global_socket.settimeout(None)  # 设为阻塞模式，由接收线程处理
            global_socket.connect((SERVER_HOST, SERVER_PORT))
            logger.info(f"✓ 已连接到服务端 {SERVER_HOST}:{SERVER_PORT}")
            connection_error_logged = False
            
            # 连接成功后，启动接收线程
            start_receive_thread()
            
            return global_socket
        except Exception as e:
            if not connection_error_logged:
                logger.warning(f"⚠ 服务端 {SERVER_HOST}:{SERVER_PORT} 不可用: {e}")
                logger.info(f"   将每 {connect_retry_interval} 秒重试连接...")
                connection_error_logged = True
            global_socket = None
            return None

def receive_thread_func():
    """后台接收线程"""
    global global_socket
    logger.info("启动后台接收线程")
    
    while True:
        sock = None
        with socket_lock:
            sock = global_socket
        
        if sock is None:
            time.sleep(1)
            continue
            
        try:
            # 读取头部
            header = sock.recv(9)
            if not header:
                logger.warning("服务端关闭了连接")
                with socket_lock:
                    if global_socket == sock:
                        global_socket.close()
                        global_socket = None
                continue
                
            msg_type, msg_len = struct.unpack('>BQ', header)
            
            # 读取内容
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
                
            if msg_type == RESPONSE_TYPE:
                # 完整文本响应
                text_response = data.decode('utf-8')
                logger.info(f"📨 收到完整响应 (长度: {len(text_response)})")
                response_queue.put(text_response)
            
            elif msg_type == ERROR_TYPE:
                # 服务器错误/拒绝消息
                error_msg = data.decode('utf-8')
                logger.warning(f"⚠️ 服务端错误: {error_msg}")
                error_queue.put(error_msg)
                # 关闭连接，因为服务器已拒绝
                with socket_lock:
                    if global_socket == sock:
                        try:
                            global_socket.close()
                        except:
                            pass
                        global_socket = None
                break  # 退出接收循环
            
            elif msg_type == STREAMING_TOKEN_TYPE:
                # 流式 token (新增)
                try:
                    token_data = data.decode('utf-8')
                    logger.debug(f"📝 收到流式 token: {token_data[:50]}...")
                    streaming_token_queue.put(token_data)
                except Exception as e:
                    logger.error(f"解析流式 token 失败: {e}")
            
            elif msg_type == TTS_AUDIO_TYPE:
                # 收到 TTS 音频数据 (句子级流式协议 - WAV)
                try:
                    response_id_len = data[0]
                    response_id = data[1:1+response_id_len].decode('utf-8')
                    # 解析句子序号 (2 bytes each, big-endian)
                    offset = 1 + response_id_len
                    sentence_idx, total_sentences = struct.unpack(">HH", data[offset:offset+4])
                    audio_data = data[offset+4:]
                    logger.info(f"🔊 收到 TTS 句 {sentence_idx + 1}/{total_sentences} (大小: {len(audio_data)} 字节)")
                    tts_audio_queue.put({
                        "response_id": response_id,
                        "sentence_idx": sentence_idx,
                        "total_sentences": total_sentences,
                        "audio_data": audio_data
                    })
                except Exception as parse_e:
                    logger.error(f"解析 TTS 音频协议失败: {parse_e}")
                    # 兼容旧格式
                    try:
                        response_id_len = data[0]
                        response_id = data[1:1+response_id_len].decode('utf-8')
                        audio_data = data[1+response_id_len:]
                        tts_audio_queue.put({
                            "response_id": response_id,
                            "sentence_idx": 0,
                            "total_sentences": 1,
                            "audio_data": audio_data
                        })
                    except:
                        tts_audio_queue.put({
                            "response_id": "",
                            "sentence_idx": 0,
                            "total_sentences": 1,
                            "audio_data": data
                        })
            
            elif msg_type == TTS_AUDIO_CHUNK_TYPE:
                # 收到 TTS 音频 chunk (Raw PCM int16) - Step 2 新增
                # 协议: response_id_len(1) + response_id + sentence_idx(2) + chunk_idx(2) + sample_rate(4) + is_final(1) + pcm_data
                try:
                    response_id_len = data[0]
                    response_id = data[1:1+response_id_len].decode('utf-8')
                    offset = 1 + response_id_len
                    sentence_idx, chunk_idx, sample_rate, is_final = struct.unpack(">HHIB", data[offset:offset+9])
                    pcm_data = data[offset+9:]
                    
                    if is_final:
                        logger.info(f"🔊 收到 TTS chunk [final] sentence={sentence_idx}")
                    else:
                        # 每10个chunk打印一次，避免日志过多
                        if chunk_idx % 10 == 0:
                            logger.info(f"🔊 收到 TTS chunk sentence={sentence_idx} chunk={chunk_idx} ({len(pcm_data)} bytes)")
                    
                    tts_audio_chunk_queue.put({
                        "response_id": response_id,
                        "sentence_idx": sentence_idx,
                        "chunk_idx": chunk_idx,
                        "sample_rate": sample_rate,
                        "is_final": bool(is_final),
                        "pcm_data": pcm_data
                    })
                except Exception as parse_e:
                    logger.error(f"解析 TTS 音频 chunk 协议失败: {parse_e}")
                
        except Exception as e:
            logger.error(f"接收线程错误: {e}")
            with socket_lock:
                if global_socket == sock:
                    try:
                        global_socket.close()
                    except:
                        pass
                    global_socket = None
            time.sleep(1)

_receive_thread_started = False
def start_receive_thread():
    global _receive_thread_started
    if not _receive_thread_started:
        t = threading.Thread(target=receive_thread_func, daemon=True)
        t.start()
        _receive_thread_started = True

def send_data(data_type: bytes, data: bytes):
    """发送数据到服务端 (非阻塞，不等待响应)"""
    global global_socket
    
    try:
        sock = get_socket()
        if sock is None:
            return False
        
        # 构造消息: 类型(1字节) + 长度(8字节) + 数据
        message = data_type + struct.pack('>Q', len(data)) + data
        
        with socket_lock:
            sock.sendall(message)
            logger.info(f"✓ 已发送 {len(data)} 字节 ({'视频' if data_type==VIDEO_TYPE else '音频'})")
            return True
            
    except Exception as e:
        logger.error(f"发送数据失败: {e}")
        with socket_lock:
            if global_socket:
                try:
                    global_socket.close()
                except:
                    pass
                global_socket = None
        return False

@app.route('/')
def index():
    """主页面 - 使用 streaming 版本的模板"""
    return render_template('index_streaming.html', interval=VIDEO_SEND_INTERVAL)

@app.route('/api/video', methods=['POST'])
def receive_video():
    """接收视频帧并根据时间间隔判断是否发送"""
    global last_video_send_time
    try:
        session_id = request.form.get('session_id')
        if not verify_session(session_id):
            return get_session_error_response()
        
        if 'frame' not in request.files:
            return jsonify({'success': False, 'error': '没有视频数据'})
        
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        # 立即检查是否需要发送
        current_time = time.time()
        sent = False
        
        # 简单的限流逻辑：如果距离上次发送超过间隔，则发送
        if current_time - last_video_send_time >= VIDEO_SEND_INTERVAL:
            success = send_data(VIDEO_TYPE, frame_data)
            if success:
                last_video_send_time = current_time
                sent = True
                logger.debug(f"📹 自动发送视频帧 (间隔 {VIDEO_SEND_INTERVAL}s)")
        
        return jsonify({
            'success': True,
            'size': len(frame_data),
            'sent': sent
        })
        
    except Exception as e:
        logger.error(f"处理视频帧失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/audio', methods=['POST'])
def receive_audio():
    """接收音频数据并发送给服务端"""
    try:
        session_id = request.form.get('session_id')
        if not verify_session(session_id):
            return get_session_error_response()
            
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有音频数据'})
            
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        success = send_data(AUDIO_TYPE, audio_data)
        
        if success:
            logger.info(f"🎤 发送音频数据 ({len(audio_data)} bytes)")
        
        return jsonify({
            'success': success,
            'size': len(audio_data)
        })
        
    except Exception as e:
        logger.error(f"处理音频失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/poll_response', methods=['GET'])
def poll_response():
    """轮询获取完整响应"""
    session_id = request.args.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        response = response_queue.get_nowait()
        return jsonify({
            'success': True,
            'response': response
        })
    except queue.Empty:
        return jsonify({
            'success': False,
            'response': None
        })

@app.route('/api/poll_streaming_token', methods=['GET'])
def poll_streaming_token():
    """轮询获取流式 token (新增)"""
    session_id = request.args.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        # 获取所有可用的 token
        tokens = []
        while True:
            try:
                token = streaming_token_queue.get_nowait()
                tokens.append(token)
            except queue.Empty:
                break
        
        if tokens:
            return jsonify({
                'success': True,
                'tokens': tokens
            })
        else:
            return jsonify({
                'success': False,
                'tokens': []
            })
    except Exception as e:
        logger.error(f"获取流式 token 失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/poll_tts_audio', methods=['GET'])
def poll_tts_audio():
    """轮询获取 TTS 音频"""
    session_id = request.args.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        tts_item = tts_audio_queue.get_nowait()
        
        response_id = tts_item.get("response_id", "")
        sentence_idx = tts_item.get("sentence_idx", 0)
        total_sentences = tts_item.get("total_sentences", 1)
        audio_data = tts_item.get("audio_data", b"")
        
        content_type = 'audio/mpeg'
        if audio_data[:4] == b'RIFF':
            content_type = 'audio/wav'
        
        logger.info(f"🔊 返回 TTS 句 {sentence_idx + 1}/{total_sentences}")
        
        return Response(
            audio_data,
            mimetype=content_type,
            headers={
                'Content-Disposition': 'inline; filename="tts_audio.mp3"',
                'X-Audio-Available': 'true',
                'X-Response-Id': response_id,
                'X-Sentence-Idx': str(sentence_idx),
                'X-Total-Sentences': str(total_sentences)
            }
        )
    except queue.Empty:
        return jsonify({
            'success': False,
            'audio': None
        })

@app.route('/api/poll_tts_audio_chunk', methods=['GET'])
def poll_tts_audio_chunk():
    """轮询获取 TTS 音频 chunk (Raw PCM int16) - Step 2 新增
    
    Returns all available chunks at once for efficiency.
    Frontend uses Web Audio API to decode and play PCM data.
    """
    session_id = request.args.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        # 获取所有可用的 chunks
        chunks = []
        while True:
            try:
                chunk = tts_audio_chunk_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break
        
        if chunks:
            # 返回所有 chunks 的 JSON 数组
            # 前端会解析并播放
            logger.info(f"🔊 返回 {len(chunks)} 个 TTS chunk 给前端")
            return jsonify({
                'success': True,
                'chunks': [{
                    'response_id': c['response_id'],
                    'sentence_idx': c['sentence_idx'],
                    'chunk_idx': c['chunk_idx'],
                    'sample_rate': c['sample_rate'],
                    'is_final': c['is_final'],
                    # PCM data 需要 base64 编码
                    'pcm_base64': __import__('base64').b64encode(c['pcm_data']).decode('ascii') if c['pcm_data'] else ''
                } for c in chunks]
            })
        else:
            return jsonify({
                'success': False,
                'chunks': []
            })
    except Exception as e:
        logger.error(f"获取 TTS 音频 chunk 失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取连接状态"""
    connected = global_socket is not None
    return jsonify({
        'connected': connected,
        'server': f"{SERVER_HOST}:{SERVER_PORT}"
    })

@app.route('/api/poll_error', methods=['GET'])
def poll_error():
    """轮询获取服务端错误消息"""
    session_id = request.args.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        error_msg = error_queue.get_nowait()
        return jsonify({
            'success': True,
            'has_error': True,
            'error': error_msg
        })
    except queue.Empty:
        return jsonify({
            'success': True,
            'has_error': False,
            'error': None
        })

@app.route('/api/acquire_session', methods=['POST'])
def acquire_session():
    """获取用户会话锁"""
    global current_user_session
    import uuid
    
    with user_session_lock:
        if current_user_session is None:
            new_session_id = str(uuid.uuid4())[:8]
            current_user_session = new_session_id
            
            # 清空残留的队列数据
            global response_queue, tts_audio_queue, tts_audio_chunk_queue, error_queue, streaming_token_queue
            response_queue = queue.Queue()
            tts_audio_queue = queue.Queue()
            tts_audio_chunk_queue = queue.Queue()
            error_queue = queue.Queue()
            streaming_token_queue = queue.Queue()
            
            logger.info(f"✅ 用户获取会话锁: {new_session_id}，已清空残留队列")
            return jsonify({
                'success': True,
                'session_id': new_session_id
            })
        else:
            logger.warning(f"⚠️ 会话被拒绝: 系统正被 {current_user_session} 使用")
            return jsonify({
                'success': False,
                'error': '系统正在被其他用户使用中，请点击"释放会话"按钮后重试'
            }), 423

@app.route('/api/force_release_session', methods=['POST'])
def force_release_session():
    """强制释放会话锁"""
    global current_user_session
    
    with user_session_lock:
        if current_user_session is not None:
            old_session = current_user_session
            current_user_session = None
            logger.info(f"🔓 会话已释放: {old_session}")
            return jsonify({
                'success': True,
                'message': f'会话已释放'
            })
        else:
            return jsonify({
                'success': True,
                'message': '没有活动会话'
            })

@app.route('/api/session_status', methods=['GET'])
def session_status():
    """获取当前会话状态"""
    with user_session_lock:
        return jsonify({
            'occupied': current_user_session is not None
        })

@app.route('/api/clear_media', methods=['POST'])
def clear_media():
    """清理服务器端的视频和音频文件"""
    if request.is_json:
        session_id = request.json.get('session_id')
    else:
        session_id = request.form.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    import os
    import glob
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    VIDEO_DIR = os.path.join(base_dir, "real_time_captured_video")
    AUDIO_DIR = os.path.join(base_dir, "real_time_captured_audio")
    
    deleted_videos = 0
    deleted_audio = False
    
    try:
        # 清理视频文件
        if os.path.exists(VIDEO_DIR):
            video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
                         glob.glob(os.path.join(VIDEO_DIR, "*.webm")) + \
                         glob.glob(os.path.join(VIDEO_DIR, "*.tmp"))
            for f in video_files:
                try:
                    os.remove(f)
                    deleted_videos += 1
                except Exception as e:
                    logger.warning(f"删除视频文件失败 {f}: {e}")
            
            merged_dir = os.path.join(VIDEO_DIR, "merged")
            if os.path.exists(merged_dir):
                merged_files = glob.glob(os.path.join(merged_dir, "*.mp4"))
                for f in merged_files:
                    try:
                        os.remove(f)
                        deleted_videos += 1
                    except Exception as e:
                        logger.warning(f"删除合并视频失败 {f}: {e}")
        
        # 清理音频文件
        audio_file = os.path.join(AUDIO_DIR, "latest.mp3")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                deleted_audio = True
            except Exception as e:
                logger.warning(f"删除音频文件失败: {e}")
        
        logger.info(f"🗑 已清理媒体文件: {deleted_videos} 个视频, {'1' if deleted_audio else '0'} 个音频")
        
        # 发送清空上下文命令到服务端
        context_cleared = send_data(CLEAR_CONTEXT_TYPE, b'clear')
        
        return jsonify({
            'success': True,
            'deleted_videos': deleted_videos,
            'deleted_audio': deleted_audio,
            'context_cleared': context_cleared
        })
        
    except Exception as e:
        logger.error(f"清理媒体文件失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """开启摄像头时调用，通知服务端清理所有文件夹"""
    if request.is_json:
        session_id = request.json.get('session_id')
    else:
        session_id = request.form.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        logger.info("📷 开启摄像头，通知服务端清理文件夹")
        
        success = send_data(START_CAMERA_TYPE, b'start')
        if success:
            logger.info("✓ 已发送开启摄像头命令")
        else:
            logger.warning("⚠ 开启摄像头命令发送失败")
        
        return jsonify({
            'success': True,
            'command_sent': success
        })
        
    except Exception as e:
        logger.error(f"发送开启摄像头命令失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def main():
    """主函数"""
    import os
    import sys
    
    use_https = '--https' in sys.argv or '-s' in sys.argv
    use_tunnel = '--tunnel' in sys.argv or '-t' in sys.argv
    
    cert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cert.pem')
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'key.pem')
    has_certs = os.path.exists(cert_file) and os.path.exists(key_file)
    
    port = 5003
    
    print("=" * 50)
    print("🎥 实时视频音频捕获客户端 (Streaming 版本)")
    print("=" * 50)
    print(f"📡 服务端地址: {SERVER_HOST}:{SERVER_PORT}")
    print("🔄 支持流式 Token 输出")
    
    tunnel_url = None
    if use_tunnel:
        try:
            from pycloudflared import try_cloudflare
            print("🌐 正在启动 Cloudflare Tunnel...")
            tunnel_url = try_cloudflare(port=port, verbose=False).tunnel
            print(f"✅ Cloudflare Tunnel 已启动!")
            print(f"🔗 公网访问地址: {tunnel_url}")
        except ImportError:
            print("❌ pycloudflared 未安装，请运行: pip install pycloudflared")
            use_tunnel = False
        except Exception as e:
            print(f"❌ Cloudflare Tunnel 启动失败: {e}")
            use_tunnel = False
    
    if use_https and has_certs:
        print("🔒 HTTPS 模式")
        print(f"🌐 访问地址: https://192.168.x.x:{port}")
    elif not use_tunnel:
        print("🌐 HTTP 模式")
        print(f"🌐 本机访问: http://localhost:{port}")
    
    print("=" * 50)
    print("功能说明:")
    print("  1. 视频: 每2秒发送一次")
    print("  2. 音频: 按住麦克风按钮录制，松开后发送")
    print("  3. 流式输出: 实时显示生成的 token")
    print("=" * 50)
    
    if use_https and has_certs:
        ssl_context = (cert_file, key_file)
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, ssl_context=ssl_context)
    else:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()
