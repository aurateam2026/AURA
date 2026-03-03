import base64
import os
import mimetypes
import cv2
import re

def encode_file_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_media_info(path):
    if path.startswith("http://") or path.startswith("https://"):
        # Simple extension guessing
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']:
            return "video", path
        return "image", path

    # Local file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Media file not found: {path}")
    
    # Check for single-frame video and convert to image if necessary - REMOVED to force video treatment
    # ext = os.path.splitext(path)[1].lower()
    # if ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']:
    #     try:
    #         cap = cv2.VideoCapture(path)
    #         if cap.isOpened():
    #             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             if frame_count <= 1:
    #                 ret, frame = cap.read()
    #                 if ret:
    #                     _, buffer = cv2.imencode('.jpg', frame)
    #                     base64_data = base64.b64encode(buffer).decode("utf-8")
    #                     cap.release()
    #                     print(f"Info: Single-frame video detected ({path}), treating as image.")
    #                     return "image", f"data:image/jpeg;base64,{base64_data}"
    #             cap.release()
    #     except Exception as e:
    #         pass
    
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        # Fallback manual check
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            mime_type = "image/jpeg" # Default to jpeg if unsure but looks like image
        elif ext in ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']:
            mime_type = "video/mp4" # Default to mp4 if unsure
        else:
            mime_type = "application/octet-stream"

    base64_data = encode_file_to_base64(path)
    data_url = f"data:{mime_type};base64,{base64_data}"
    
    if mime_type.startswith("video"):
        # 添加调试信息：确认视频被识别为视频
        print(f"[context_manage] Detected VIDEO: {path}, mime_type={mime_type}")
        return "video", data_url
    elif mime_type.startswith("image"):
        print(f"[context_manage] Detected IMAGE: {path}, mime_type={mime_type}")
        return "image", data_url
    else:
        # Default to image or warn
        print(f"[context_manage] Unknown media type, defaulting to image: {path}, mime_type={mime_type}")
        return "image", data_url

def remove_markdown(text):
    """
    Remove Markdown formatting from text.
    """
    # Remove bold/italic (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove headers (# Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove list items (- item, * item) but keep newline structure
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Remove links ([text](url))
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove code blocks (```code```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove inline code (`code`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    return text

class ContextManager:
    def __init__(self, max_text_rounds: int = 30, max_media_rounds: int = 30, similarity_threshold: float = 0.85):
        self.history = []
        self.max_text_rounds = max_text_rounds
        self.max_media_rounds = max_media_rounds
        self.similarity_threshold = similarity_threshold
        self.last_assistant_response = None  # 存储上一次的助手回复

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（0.0 - 1.0）。
        使用简单的字符级别 Jaccard 相似度。
        """
        if not text1 or not text2:
            return 0.0
        
        # 去除空白字符进行比较
        t1 = text1.strip()
        t2 = text2.strip()
        
        # 如果完全相同
        if t1 == t2:
            return 1.0
        
        # 使用 n-gram (bigram) 进行相似度计算
        def get_ngrams(text, n=2):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(t1)
        ngrams2 = get_ngrams(t2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Jaccard 相似度
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0

    def append_user_message(self, prompt: str, media_input: str = None, forced_mode: str = None):
        """
        Appends a user message to the history, handling media processing.
        添加后会自动触发裁剪，确保 context 不会超出限制。
        """
        if media_input:
            if forced_mode:
                media_type = forced_mode
                _, media_url = get_media_info(media_input)
            else:
                media_type, media_url = get_media_info(media_input)
            
            content = []
            
            if media_type == "image":
                content.append({
                    "type": "image_url",
                    "image_url": {"url": media_url}
                })
            elif media_type == "video":
                content.append({
                    "type": "video_url",
                    "video_url": {"url": media_url}
                })

            content.append({"type": "text", "text": prompt})
                
            self.history.append({
                "role": "user",
                "content": content
            })
        else:
            self.history.append({
                "role": "user",
                "content": prompt
            })
        
        # 每次添加 user 消息后都触发裁剪，确保 context 不会无限增长
        self._prune_history()

    def append_assistant_message(self, content: str) -> bool:
        """
        Appends an assistant message to the history and prunes old messages if needed.
        如果当前回复与上一次回复相似度过高，则跳过添加。
        
        Returns:
            bool: True 如果回复被添加，False 如果因相似度过高而跳过
        """
        # 检查与上一次回复的相似度
        if self.last_assistant_response is not None:
            similarity = self._calculate_similarity(content, self.last_assistant_response)
            if similarity >= self.similarity_threshold:
                print(f"[context_manage] 跳过相似回复 (相似度: {similarity:.2%} >= 阈值 {self.similarity_threshold:.2%})")
                return False
        
        # 添加到历史
        self.history.append({
            "role": "assistant",
            "content": content
        })
        
        # 更新上一次回复
        self.last_assistant_response = content
        print(f"[context_manage] 已添加助手回复到上下文")
        
        self._prune_history()
        return True

    def _prune_history(self):
        """
        Prunes history based on separate limits for text-only and multi-modal rounds.
        One 'round' is defined as a User message and subsequent messages until the next User message.
        """
        rounds = [] # List of {'index': int, 'type': 'text'|'media'}
        
        # Identify user rounds
        for i, msg in enumerate(self.history):
            if msg['role'] == 'user':
                content = msg['content']
                is_media = False
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') in ['image_url', 'video_url']:
                            is_media = True
                            break
                
                rounds.append({
                    'index': i,
                    'type': 'media' if is_media else 'text'
                })
        
        # Filter rounds by type
        text_rounds = [r for r in rounds if r['type'] == 'text']
        media_rounds = [r for r in rounds if r['type'] == 'media']
        
        indices_to_remove = set()
        
        # Determine text rounds to remove (FIFO)
        if len(text_rounds) > self.max_text_rounds:
            num_remove = len(text_rounds) - self.max_text_rounds
            for i in range(num_remove):
                start_idx = text_rounds[i]['index']
                # Determine end index for this round (next user message or end of history)
                next_round_idx = len(self.history)
                for r in rounds:
                    if r['index'] > start_idx:
                        next_round_idx = r['index']
                        break
                
                for k in range(start_idx, next_round_idx):
                    indices_to_remove.add(k)

        # Determine media rounds to remove (FIFO)
        if len(media_rounds) > self.max_media_rounds:
            num_remove = len(media_rounds) - self.max_media_rounds
            for i in range(num_remove):
                start_idx = media_rounds[i]['index']
                next_round_idx = len(self.history)
                for r in rounds:
                    if r['index'] > start_idx:
                        next_round_idx = r['index']
                        break
                
                for k in range(start_idx, next_round_idx):
                    indices_to_remove.add(k)
        
        # Remove identified messages
        if indices_to_remove:
            self.history = [msg for i, msg in enumerate(self.history) if i not in indices_to_remove]

    def get_messages(self):
        """
        Returns the current full conversation history.
        """
        return self.history

    def clear_history(self):
        """
        Clears the conversation history.
        """
        self.history = []
        self.last_assistant_response = None  # 同时重置上一次回复记录

# Global instance with default settings
# max_text_rounds and max_media_rounds can be adjusted as needed
# similarity_threshold: 如果回复相似度 >= 该阈值，则跳过添加到上下文
# 注意：每个视频在 VLM 中会占用大量 token（~2000-4000 tokens），
# 所以 max_media_rounds 不能太大，否则会超出模型的 max_model_len (65536)
_global_context = ContextManager(max_text_rounds=15, max_media_rounds=30, similarity_threshold=0.85)

def create_message(prompt: str, media_input: str = None, forced_mode: str = None) -> list:
    """
    Maintains user and model conversation history globally.
    Appends the new user message to the global history and returns the full history.
    """
    _global_context.append_user_message(prompt, media_input, forced_mode)
    return _global_context.get_messages()

def append_response(content: str) -> bool:
    """
    Helper to append assistant response to global context.
    如果回复与上一次相似度过高，则跳过。
    
    Returns:
        bool: True 如果回复被添加，False 如果因相似度过高而跳过
    """
    return _global_context.append_assistant_message(content)

def clear_global_history():
    _global_context.clear_history()

def set_similarity_threshold(threshold: float):
    """
    设置全局 context 的相似度阈值。
    """
    _global_context.similarity_threshold = threshold
    print(f"[context_manage] 相似度阈值已设置为: {threshold:.2%}")

def set_max_rounds(max_text_rounds: int = None, max_media_rounds: int = None):
    """
    设置全局 context 的最大轮次限制。
    
    Args:
        max_text_rounds: 最大文本轮次数（None 表示不修改）
        max_media_rounds: 最大媒体轮次数（None 表示不修改）
    """
    if max_text_rounds is not None:
        _global_context.max_text_rounds = max_text_rounds
    if max_media_rounds is not None:
        _global_context.max_media_rounds = max_media_rounds
    print(f"[context_manage] 最大轮次已设置: text={_global_context.max_text_rounds}, media={_global_context.max_media_rounds}")
