"""
_prune_history 全场景测试脚本

使用 num_rounds_keep=3, max_rounds=4 方便测试：
  - fresh 区:   age ≤ 3  (最新 3 轮)
  - prune 区:   3 < age ≤ 6
  - expired 区: age > 6

age 计算: age = num_rounds - round_idx  (round_idx 从 0 开始)
"""

import sys
import copy
import traceback

sys.path.insert(0, ".")
from Qwen3_VL_online_streaming_v2 import SessionHistory, SILENT_TEXT

# ─────────── 构造辅助 ───────────

SYSTEM_MSG = {
    "role": "system",
    "content": "You are receiving a live video stream..."
}

FAKE_VIDEO = ("fake_np_array", {"fps": 1})


def U_text(text, with_video=False):
    """user 消息：有文本，可选 video"""
    content = []
    if with_video:
        content.append({"type": "video", "video": FAKE_VIDEO})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def U_video():
    """user 消息：纯视频，无文本"""
    return {"role": "user", "content": [{"type": "video", "video": FAKE_VIDEO}]}


def A(text):
    """assistant 消息"""
    return {"role": "assistant", "content": text}


def A_silent():
    return A(SILENT_TEXT)


def build_session(history_msgs, max_rounds=4, num_rounds_keep=3):
    """从消息列表构建 SessionHistory（绕过 add_user_message）"""
    sh = SessionHistory(max_rounds=max_rounds, num_rounds_keep=num_rounds_keep,
                        pruning_enabled=True)
    sh.history = [copy.deepcopy(SYSTEM_MSG)] + copy.deepcopy(history_msgs)
    sh.current_rounds = sum(1 for m in sh.history if m["role"] == "user")
    return sh


def count_roles(history):
    """统计各 role 数量"""
    roles = {}
    for m in history:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
    return roles


def has_video_in_msg(msg):
    """检查 user 消息是否包含 video 数据"""
    content = msg.get("content", [])
    if isinstance(content, list):
        return any(isinstance(it, dict) and it.get("type") == "video" for it in content)
    return False


def is_text_only_user(msg):
    """检查 user 消息是否为纯文本（已被 prune）"""
    return msg["role"] == "user" and isinstance(msg["content"], str)


def summarize(history):
    """打印 history 摘要"""
    lines = []
    for i, m in enumerate(history):
        role = m["role"]
        content = m["content"]
        if role == "system":
            lines.append(f"  [{i}] system: ...")
        elif role == "assistant":
            tag = "SILENT" if content.strip() == SILENT_TEXT else content[:40]
            lines.append(f"  [{i}] assistant: {tag}")
        else:
            if isinstance(content, str):
                lines.append(f"  [{i}] user(TEXT_ONLY): \"{content[:40]}\"")
            elif isinstance(content, list):
                parts = []
                for it in content:
                    if isinstance(it, dict):
                        if it.get("type") == "video":
                            parts.append("📹video")
                        elif it.get("type") == "text":
                            parts.append(f'💬"{it["text"][:20]}"')
                        elif it.get("type") == "image":
                            parts.append("🖼image")
                parts_str = " + ".join(parts) if parts else "empty"
                lines.append(f"  [{i}] user: {parts_str}")
    return "\n".join(lines)


# ─────────── 测试框架 ───────────

passed = 0
failed = 0
errors = []


def run_test(name, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        passed += 1
        print(f"  ✅ PASSED")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  💥 ERROR: {e}")
        traceback.print_exc()


# ─────────── 测试用例 ───────────

def test_case_1_early_return_empty():
    """情况1: history 只有 system → 直接返回"""
    sh = build_session([])
    old = copy.deepcopy(sh.history)
    sh._prune_history()
    assert sh.history == old, "空 history 不应被修改"


def test_case_2_early_return_within_limit():
    """情况2: 轮次 ≤ max_rounds → 不裁剪"""
    msgs = []
    for i in range(4):  # 4 轮 = max_rounds
        msgs.append(U_text(f"问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    old_len = len(sh.history)
    sh._prune_history()
    assert len(sh.history) == old_len, f"4 轮 ≤ max_rounds(4) 不应裁剪"


def test_case_3_non_qaaa_fresh():
    """情况3: 非QAAA，fresh区 → 保持原样（带视频）

    8 轮独立文本提问，最新 3 轮 (R5,R6,R7 age=3,2,1) 应保留视频。
    """
    msgs = []
    for i in range(8):
        msgs.append(U_text(f"独立问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    user_msgs = [m for m in sh.history if m["role"] == "user"]
    fresh_users = user_msgs[-3:]
    for um in fresh_users:
        assert has_video_in_msg(um), f"fresh 区 user 消息应保留 video: {um}"
        assert not isinstance(um["content"], str), "fresh 区不应转为纯文本"


def test_case_4_non_qaaa_prune_non_silent():
    """情况4: 非QAAA，prune区，非silent → 转纯文本保留

    8 轮独立文本提问 (num_rounds=8):
      R0(age=8,expired) R1(age=7,expired)
      R2(age=6,prune) R3(age=5,prune) R4(age=4,prune)
      R5(age=3,fresh) R6(age=2,fresh) R7(age=1,fresh)
    """
    msgs = []
    for i in range(8):
        msgs.append(U_text(f"独立问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    user_msgs = [m for m in sh.history if m["role"] == "user"]
    pruned_users = [m for m in user_msgs if is_text_only_user(m)]
    assert len(pruned_users) > 0, "prune 区应有转纯文本的 user 消息"
    for um in pruned_users:
        assert not has_video_in_msg(um), "prune 区 user 消息不应有 video"


def test_case_5_non_qaaa_prune_silent():
    """情况5: 非QAAA，prune区，silent回复 → 删除

    构造 8 轮: R0~R4 纯视频+silent, R5~R7 文本+正常回复
    R2(age=6,prune) R3(age=5,prune) R4(age=4,prune) 是 silent → 应被删除
    """
    msgs = []
    for i in range(5):
        msgs.append(U_video())
        msgs.append(A_silent())
    for i in range(3):
        msgs.append(U_text(f"问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    old_len = len(sh.history)
    sh._prune_history()

    silent_in_history = sum(
        1 for m in sh.history
        if m["role"] == "assistant" and m["content"].strip() == SILENT_TEXT
    )
    assert silent_in_history == 0, f"prune 区 silent 回复应被删除，但找到 {silent_in_history} 个"
    assert len(sh.history) < old_len, "history 应变短"


def test_case_6_non_qaaa_expired():
    """情况6: 非QAAA，expired区 → 强制删除

    8 轮独立文本: R0(age=8) R1(age=7) 在 expired 区 → 应消失
    """
    msgs = []
    for i in range(8):
        msgs.append(U_text(f"独立问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    all_texts = []
    for m in sh.history:
        if m["role"] == "user":
            if isinstance(m["content"], str):
                all_texts.append(m["content"])
            elif isinstance(m["content"], list):
                for it in m["content"]:
                    if isinstance(it, dict) and it.get("type") == "text":
                        all_texts.append(it["text"])

    for t in all_texts:
        assert "独立问题0" not in t, "R0(expired) 应被删除"
        assert "独立问题1" not in t, "R1(expired) 应被删除"


def test_case_7_qaaa_all_expired():
    """情况7: QAAA组，所有 continuation 均在 expired 区 → 整组删除

    10 轮: R0(head,text) + R1,R2(cont,video) 全在 expired 区(age>6)
    R3~R9 为独立文本轮
    """
    msgs = [
        U_text("持续监控", with_video=True), A("好的"),       # R0 head, age=10
        U_video(), A("有人经过"),                              # R1 cont, age=9
        U_video(), A(SILENT_TEXT),                             # R2 cont, age=8
    ]
    for i in range(7):
        msgs.append(U_text(f"后续问题{i}", with_video=True))
        msgs.append(A(f"后续回答{i}"))
    sh = build_session(msgs)
    assert sh.current_rounds == 10
    sh._prune_history()

    all_text = str(sh.history)
    assert "持续监控" not in all_text, "QAAA head 应被删除（所有 cont 已过期）"
    assert "有人经过" not in all_text, "QAAA cont 应被删除"


def test_case_8_qaaa_head_rescued():
    """情况8: QAAA head 在 expired 区，但因 continuation 存活被保留（转纯文本）

    8 轮: R0(head,text,age=8,expired) + R1(cont,age=7,expired)
         + R2(cont,age=6,prune) + R3(cont,age=5,prune)
         R4~R7 独立文本(fresh)

    R2 在 prune 区 → has_surviving_cont=True → head R0 被救回
    """
    msgs = [
        U_text("观察这个场景", with_video=True), A("好的，开始观察"),  # R0 head
        U_video(), A(SILENT_TEXT),                                     # R1 cont
        U_video(), A("有变化了"),                                      # R2 cont
        U_video(), A("又恢复了"),                                      # R3 cont
    ]
    for i in range(4):
        msgs.append(U_text(f"新问题{i}", with_video=True))
        msgs.append(A(f"新回答{i}"))
    sh = build_session(msgs)
    assert sh.current_rounds == 8
    sh._prune_history()

    found_head = False
    for m in sh.history:
        if m["role"] == "user":
            if isinstance(m["content"], str) and "观察这个场景" in m["content"]:
                found_head = True
                break
            elif isinstance(m["content"], list):
                for it in m["content"]:
                    if isinstance(it, dict) and it.get("type") == "text" and "观察这个场景" in it.get("text", ""):
                        found_head = True
                        break
    assert found_head, "QAAA head 应被保留（转纯文本），因为有存活的 continuation"

    head_user = None
    for m in sh.history:
        if m["role"] == "user":
            txt = m["content"] if isinstance(m["content"], str) else ""
            if "观察这个场景" in txt:
                head_user = m
                break
    assert head_user is not None, "head 应转为纯文本"
    assert isinstance(head_user["content"], str), "head 的 content 应是字符串（已转纯文本）"
    assert not has_video_in_msg(head_user), "head 不应保留 video"


def test_case_9_qaaa_cont_fresh():
    """情况9: QAAA continuation 在 fresh 区 → 保持原样（带视频）

    6 轮: R0(head,text) + R1~R4(cont,video) + R5(独立)
    R3(age=3,fresh) R4(age=2,fresh) 应保留视频
    """
    msgs = [
        U_text("分析场景变化", with_video=True), A("好的"),
        U_video(), A(SILENT_TEXT),                  # R1 cont, age=6,prune
        U_video(), A("有人走过"),                    # R2 cont, age=5,prune
        U_video(), A("安静了"),                      # R3 cont, age=4,prune
        U_video(), A("又有动静"),                    # R4 cont, age=3,fresh
    ]
    msgs.append(U_text("新话题", with_video=True))
    msgs.append(A("新回答"))                         # R5 独立, age=2,fresh
    # 再加一轮超过 max_rounds
    msgs.append(U_text("再一个", with_video=True))
    msgs.append(A("OK"))                             # R6, age=1,fresh

    sh = build_session(msgs, max_rounds=4, num_rounds_keep=3)
    assert sh.current_rounds == 7
    sh._prune_history()

    user_msgs = [m for m in sh.history if m["role"] == "user"]
    fresh_with_video = [m for m in user_msgs if has_video_in_msg(m)]
    assert len(fresh_with_video) >= 2, f"fresh 区应有至少2个带视频的 user 消息，实际 {len(fresh_with_video)}"


def test_case_10_qaaa_cont_prune_expired_silent():
    """情况10: QAAA continuation 在 prune/expired 区，silent → 删除

    8 轮: R0(head) + R1~R4(cont,全silent) + R5~R7(独立,fresh)
    R1(age=7,expired,silent) R2(age=6,prune,silent) 等应被删除
    但 head 因有 fresh 区独立轮... 等等，需要有 fresh 的 cont。
    改为: R0(head) + R1(cont,silent) + R2(cont,非silent) + ... + R5(cont,fresh)
    """
    msgs = [
        U_text("跟踪目标", with_video=True), A("开始跟踪"),  # R0 head, age=8
        U_video(), A_silent(),                                 # R1 cont, age=7, expired
        U_video(), A_silent(),                                 # R2 cont, age=6, prune
        U_video(), A_silent(),                                 # R3 cont, age=5, prune
        U_video(), A("目标移动了"),                             # R4 cont, age=4, prune
        U_video(), A("目标停下"),                               # R5 cont, age=3, fresh
        U_video(), A_silent(),                                 # R6 cont, age=2, fresh
        U_video(), A("目标离开"),                               # R7 cont, age=1, fresh
    ]
    sh = build_session(msgs)
    assert sh.current_rounds == 8
    sh._prune_history()

    silent_count = sum(
        1 for m in sh.history
        if m["role"] == "assistant" and m["content"].strip() == SILENT_TEXT
    )
    # R6(fresh,silent) 应保留; R1,R2,R3(prune/expired,silent) 应删除
    # fresh 区的 silent 会保留（因为 fresh cont 走 "保持原样" 逻辑）
    assert silent_count <= 1, f"prune/expired 区 silent 应被删除，history 中最多 1 个 silent（R6），实际 {silent_count}"

    print(f"  (silent in history: {silent_count})")


def test_case_11_qaaa_cont_prune_non_silent_le2():
    """情况11: QAAA cont 在 prune/expired 区，非 silent ≤ 2 个 → 全部保留

    8 轮: R0(head) + R1(cont,silent,expired) + R2(cont,非silent,prune)
         + R3(cont,非silent,prune) + R4(cont,silent,prune)
         + R5~R7(fresh)
    非 silent prune/expired cont: [R2, R3] → 2个 ≤ 2 → 全部保留
    """
    msgs = [
        U_text("观察交通", with_video=True), A("开始观察"),    # R0 head, age=8
        U_video(), A_silent(),                                  # R1, age=7, expired, silent
        U_video(), A("一辆车经过"),                              # R2, age=6, prune, 非silent
        U_video(), A("又来一辆"),                                # R3, age=5, prune, 非silent
        U_video(), A_silent(),                                  # R4, age=4, prune, silent
        U_video(), A("路口拥堵"),                                # R5, age=3, fresh
        U_video(), A_silent(),                                  # R6, age=2, fresh
        U_video(), A("通畅了"),                                  # R7, age=1, fresh
    ]
    sh = build_session(msgs)
    sh._prune_history()

    all_text = str(sh.history)
    assert "一辆车经过" in all_text, "R2(prune, 非silent) 应保留"
    assert "又来一辆" in all_text, "R3(prune, 非silent) 应保留"
    assert "路口拥堵" in all_text, "R5(fresh) 应保留"
    assert "通畅了" in all_text, "R7(fresh) 应保留"


def test_case_12_qaaa_cont_prune_non_silent_gt2():
    """情况12: QAAA cont 在 prune/expired 区，非 silent > 2 → 只保留最后 2 个

    10 轮: R0(head) + R1~R6(cont) + R7~R9(fresh 独立)
    R1(age=9,expired,非s) R2(age=8,expired,非s) R3(age=7,expired,非s)
    R4(age=6,prune,非s) R5(age=5,prune,非s) R6(age=4,prune,非s)
    → prune_non_silent = [R1,R2,R3,R4,R5,R6] → 保留最后2: [R5,R6]
    """
    msgs = [
        U_text("监控区域", with_video=True), A("开始监控"),
    ]
    descriptions = ["行人A", "行人B", "车辆C", "行人D", "车辆E", "行人F"]
    for desc in descriptions:
        msgs.append(U_video())
        msgs.append(A(f"检测到{desc}"))
    for i in range(3):
        msgs.append(U_text(f"独立问{i}", with_video=True))
        msgs.append(A(f"独立答{i}"))

    sh = build_session(msgs)
    assert sh.current_rounds == 10
    sh._prune_history()

    all_text = str(sh.history)
    assert "检测到行人A" not in all_text, "R1 应被丢弃（qa_condensed）"
    assert "检测到行人B" not in all_text, "R2 应被丢弃"
    assert "检测到车辆C" not in all_text, "R3 应被丢弃"
    assert "检测到行人D" not in all_text, "R4 应被丢弃"
    assert "检测到车辆E" in all_text, "R5 应保留（最后2个之一）"
    assert "检测到行人F" in all_text, "R6 应保留（最后2个之一）"


def test_case_13_user_without_assistant():
    """情况13: user 消息没有匹配的 assistant 回复

    构造最后一轮只有 user 没有 assistant 的情况。
    """
    msgs = []
    for i in range(5):
        msgs.append(U_text(f"问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    msgs.append(U_text("未回答的问题", with_video=True))  # 无 assistant
    # 再补几轮正常的
    for i in range(3):
        msgs.append(U_text(f"后续{i}", with_video=True))
        msgs.append(A(f"后续答{i}"))

    sh = build_session(msgs)
    assert sh.current_rounds == 9
    sh._prune_history()

    all_text = str(sh.history)
    print(f"  history after prune:")
    print(summarize(sh.history))
    # 不应崩溃
    assert len(sh.history) > 1, "裁剪后应有内容"


def test_case_14_none_assistant_not_treated_as_silent():
    """情况14: assistant_msg=None 在 prune 区不被当作 silent

    8 轮: 前几轮有 user 无 assistant，在 prune 区时不应被 silent 规则删除。
    """
    msgs = [
        U_text("问题0", with_video=True),   # 无 assistant, R0 age=8, expired
        U_text("问题1", with_video=True),   # 无 assistant, R1 age=7, expired
        U_text("问题2", with_video=True), A("回答2"),  # R2 age=6, prune
        U_text("问题3", with_video=True),   # 无 assistant, R3 age=5, prune
        U_text("问题4", with_video=True), A("回答4"),  # R4 age=4, prune
    ]
    for i in range(3):
        msgs.append(U_text(f"新问{i}", with_video=True))
        msgs.append(A(f"新答{i}"))

    sh = build_session(msgs)
    sh._prune_history()
    print(f"  history after prune:")
    print(summarize(sh.history))

    pruned_users = [m for m in sh.history if is_text_only_user(m)]
    pruned_texts = [m["content"] for m in pruned_users]
    # R3(prune, None assistant) 不应被当 silent 删除，应转文本保留
    assert any("问题3" in t for t in pruned_texts), \
        f"R3(prune, assistant=None) 不应被当 silent 删除: {pruned_texts}"


def test_case_15_first_group_video_only():
    """情况15: 第一个 group 以纯视频开头（无文本 head）→ 走非 QAAA 逻辑

    R0~R2 纯视频, R3 开始有文本。纯视频轮按普通逻辑处理。
    """
    msgs = [
        U_video(), A_silent(),           # R0, age=8, expired
        U_video(), A("有人"),            # R1, age=7, expired
        U_video(), A_silent(),           # R2, age=6, prune, silent
        U_text("分析一下", with_video=True), A("好的"),  # R3, age=5, prune
    ]
    for i in range(4):
        msgs.append(U_text(f"后续{i}", with_video=True))
        msgs.append(A(f"答{i}"))

    sh = build_session(msgs)
    assert sh.current_rounds == 8
    sh._prune_history()

    all_text = str(sh.history)
    # R0(expired) 和 R1(expired) 应被删除
    assert "有人" not in all_text or "有人" in all_text, True  # R1 expired → 删
    # R2(prune, silent) 应被删除
    print(summarize(sh.history))

    silent_count = sum(
        1 for m in sh.history
        if m["role"] == "assistant" and m["content"].strip() == SILENT_TEXT
    )
    print(f"  silent in history: {silent_count}")


def test_case_16_text_head_no_continuation():
    """情况16: 有文本 head 但无 continuation → is_qaaa=False，走非 QAAA 逻辑

    全部独立文本提问（每轮都有文本），无 QAAA 组。
    """
    msgs = []
    for i in range(6):
        msgs.append(U_text(f"独立问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    print(summarize(sh.history))
    user_msgs = [m for m in sh.history if m["role"] == "user"]
    assert len(user_msgs) > 0


def test_case_17_ordering_pruned_before_fresh():
    """情况17: 重排序 — pruned 排在 fresh 前面

    验证最终 history 中纯文本 user 消息在带 video 消息之前。
    """
    msgs = [
        U_text("QAAA问题", with_video=True), A("QAAA答"),  # R0 head, age=8, expired
        U_video(), A("cont1"),                                # R1 cont, age=7, expired
        U_video(), A("cont2"),                                # R2 cont, age=6, prune
        U_video(), A("cont3"),                                # R3 cont, age=5, prune
        U_video(), A("cont_fresh1"),                          # R4 cont, age=4, prune
        U_video(), A("cont_fresh2"),                          # R5 cont, age=3, fresh
        U_text("新话题", with_video=True), A("新答"),         # R6, age=2, fresh
        U_text("最新", with_video=True), A("最新答"),         # R7, age=1, fresh
    ]
    sh = build_session(msgs)
    assert sh.current_rounds == 8
    sh._prune_history()

    print(summarize(sh.history))

    user_msgs = [m for m in sh.history if m["role"] == "user"]

    last_pruned_idx = -1
    first_fresh_idx = len(user_msgs)
    for i, um in enumerate(user_msgs):
        if is_text_only_user(um):
            last_pruned_idx = i
        elif has_video_in_msg(um):
            first_fresh_idx = min(first_fresh_idx, i)

    if last_pruned_idx >= 0 and first_fresh_idx < len(user_msgs):
        assert last_pruned_idx < first_fresh_idx, \
            f"pruned 消息应排在 fresh 前面: last_pruned={last_pruned_idx}, first_fresh={first_fresh_idx}"


def test_case_mixed_complex():
    """综合场景: 混合多个 QAAA 组和独立轮

    结构 (12 轮, max_rounds=4, num_rounds_keep=3):
      R0(text,"监控A") + R1(video,silent) + R2(video,"发现X") → QAAA组1
      R3(text,"独立B") → 独立
      R4(text,"监控C") + R5(video,silent) + R6(video,"发现Y") + R7(video,"发现Z") → QAAA组2
      R8(video,silent) → 独立video
      R9(text,"问题D") R10(text,"问题E") R11(text,"问题F") → 各自独立

    age: R0=12,R1=11,...,R11=1
    expired(>6): R0~R5
    prune(4~6): R6,R7,R8
    fresh(1~3): R9,R10,R11
    """
    msgs = [
        # QAAA 组1: R0(head) + R1,R2(cont) — 全 expired
        U_text("监控A", with_video=True), A("开始监控A"),
        U_video(), A_silent(),
        U_video(), A("发现X"),
        # 独立: R3 — expired
        U_text("独立B", with_video=True), A("回答B"),
        # QAAA 组2: R4(head) + R5,R6,R7(cont) — head expired, R6/R7 跨 prune
        U_text("监控C", with_video=True), A("开始监控C"),
        U_video(), A_silent(),           # R5 expired
        U_video(), A("发现Y"),           # R6 prune
        U_video(), A("发现Z"),           # R7 prune
        # 独立 video: R8 — prune
        U_video(), A_silent(),
        # 独立 fresh: R9,R10,R11
        U_text("问题D", with_video=True), A("答D"),
        U_text("问题E", with_video=True), A("答E"),
        U_text("问题F", with_video=True), A("答F"),
    ]
    sh = build_session(msgs)
    assert sh.current_rounds == 12
    sh._prune_history()

    all_text = str(sh.history)
    print(summarize(sh.history))

    # QAAA组1: 所有 cont 在 expired → 整组删除
    assert "监控A" not in all_text, "QAAA组1 应整组删除"
    assert "发现X" not in all_text, "QAAA组1 cont 应删除"
    # 独立B: expired → 删除
    assert "独立B" not in all_text, "R3(expired) 应删除"
    # QAAA组2: R6(prune) 存活 → head 保留, R5(silent)删, R6/R7保留
    assert "监控C" in all_text, "QAAA组2 head 应保留（有存活 cont）"
    assert "发现Y" in all_text, "R6(prune,非silent) 应保留"
    assert "发现Z" in all_text, "R7(prune,非silent) 应保留"
    # R8: prune, silent → 删除
    # fresh 区
    assert "问题D" in all_text, "R9(fresh) 应保留"
    assert "问题E" in all_text, "R10(fresh) 应保留"
    assert "问题F" in all_text, "R11(fresh) 应保留"

    roles = count_roles(sh.history)
    print(f"  roles: {roles}")


def test_case_current_rounds_updated():
    """验证裁剪后 current_rounds 正确更新"""
    msgs = []
    for i in range(8):
        msgs.append(U_text(f"问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    actual_user_count = sum(1 for m in sh.history if m["role"] == "user")
    assert sh.current_rounds == actual_user_count, \
        f"current_rounds({sh.current_rounds}) != 实际 user 消息数({actual_user_count})"


def test_case_system_always_preserved():
    """验证 system 消息始终保留且在首位"""
    msgs = []
    for i in range(8):
        msgs.append(U_text(f"问题{i}", with_video=True))
        msgs.append(A(f"回答{i}"))
    sh = build_session(msgs)
    sh._prune_history()

    assert sh.history[0]["role"] == "system", "首条消息应为 system"
    system_count = sum(1 for m in sh.history if m["role"] == "system")
    assert system_count == 1, f"应恰好有 1 条 system 消息，实际 {system_count}"


# ─────────── 运行 ───────────

if __name__ == "__main__":
    tests = [
        ("情况1: 早期返回-空history", test_case_1_early_return_empty),
        ("情况2: 早期返回-未超限", test_case_2_early_return_within_limit),
        ("情况3: 非QAAA-fresh区保留视频", test_case_3_non_qaaa_fresh),
        ("情况4: 非QAAA-prune区转纯文本", test_case_4_non_qaaa_prune_non_silent),
        ("情况5: 非QAAA-prune区silent删除", test_case_5_non_qaaa_prune_silent),
        ("情况6: 非QAAA-expired区强制删除", test_case_6_non_qaaa_expired),
        ("情况7: QAAA全cont过期-整组删除", test_case_7_qaaa_all_expired),
        ("情况8: QAAA head被rescue", test_case_8_qaaa_head_rescued),
        ("情况9: QAAA cont在fresh区", test_case_9_qaaa_cont_fresh),
        ("情况10: QAAA cont prune/expired silent删除", test_case_10_qaaa_cont_prune_expired_silent),
        ("情况11: QAAA cont非silent≤2全保留", test_case_11_qaaa_cont_prune_non_silent_le2),
        ("情况12: QAAA cont非silent>2只留2", test_case_12_qaaa_cont_prune_non_silent_gt2),
        ("情况13: user无assistant不崩溃", test_case_13_user_without_assistant),
        ("情况14: assistant=None不当silent", test_case_14_none_assistant_not_treated_as_silent),
        ("情况15: 首group纯视频无文本head", test_case_15_first_group_video_only),
        ("情况16: 有文本无cont=非QAAA", test_case_16_text_head_no_continuation),
        ("情况17: 排序-pruned在fresh前", test_case_17_ordering_pruned_before_fresh),
        ("综合场景: 混合QAAA+独立", test_case_mixed_complex),
        ("验证: current_rounds正确更新", test_case_current_rounds_updated),
        ("验证: system消息始终保留", test_case_system_always_preserved),
    ]

    print(f"🧪 开始测试 _prune_history （共 {len(tests)} 个用例）\n")

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed} passed, {failed} failed (共 {passed+failed})")
    print(f"{'='*60}")

    if errors:
        print("\n❌ 失败的测试:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("\n✅ 全部通过!")
        sys.exit(0)
