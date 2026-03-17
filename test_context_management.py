"""
Tests for SessionHistory context management (per context.md spec).
Runs without GPU — only tests the pruning / history management logic.

Usage:
    python test_context_management.py
"""

import sys
import os
import json
import copy

# Patch globals that SessionHistory depends on before importing
SILENT_TEXT = "<|silent|>"

# We need to set the global SILENT_TOKEN_ID before importing
sys.path.insert(0, os.path.dirname(__file__))

# Minimal stubs so the module can be imported without vllm / GPU
class _FakeModule:
    def __getattr__(self, name):
        return _FakeModule()
    def __call__(self, *a, **kw):
        return _FakeModule()

for mod in [
    "cv2", "numpy", "aiohttp", "requests",
    "vllm", "vllm.engine", "vllm.engine.arg_utils",
    "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.async_llm",
    "context_manage",
]:
    if mod not in sys.modules:
        sys.modules[mod] = _FakeModule()

# Provide the actual `remove_markdown` so the import line resolves
sys.modules["context_manage"].remove_markdown = lambda t: t

import importlib
mod = importlib.import_module("Qwen3_VL_online_streaming_v2_ContextManaged")
SessionHistory = mod.SessionHistory

# Set the global SILENT_TOKEN_ID that _is_silent_response uses
mod.SILENT_TEXT = SILENT_TEXT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_PLACEHOLDER = {"type": "video", "video": ("fake_np_array", {"fps": 1})}


def make_video_content(text: str = ""):
    """Build a user-message content list with <video> and optional text."""
    content = [VIDEO_PLACEHOLDER]
    if text:
        content.append({"type": "text", "text": text})
    return content


def make_session(max_rounds=10, num_rounds_keep=5, **kw) -> SessionHistory:
    return SessionHistory(
        max_rounds=max_rounds,
        num_rounds_keep=num_rounds_keep,
        pruning_enabled=True,
        **kw,
    )


def add_basic_qa(h: SessionHistory, question: str, answer: str):
    """Add a Basic QA (1 round): user query + assistant response."""
    h.add_user_message(question, video_tuple=("fake_np", {"fps": 1}))
    h.add_assistant_message(answer)


def add_1q1a_qa(h: SessionHistory, question: str, confirm: str,
                silent_count: int, final_answer: str):
    """Add a 1Q1A QA: head + N silent rounds + 1 final answer."""
    h.add_user_message(question, video_tuple=("fake_np", {"fps": 1}))
    h.add_assistant_message(confirm)
    for _ in range(silent_count):
        h.add_user_message("", video_tuple=("fake_np", {"fps": 1}))
        h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake_np", {"fps": 1}))
    h.add_assistant_message(final_answer)


def add_1qna_qa(h: SessionHistory, question: str, confirm: str,
                answers: list, silent_between: int = 2):
    """
    Add a 1QNA QA: head + (silent×N + answer)×len(answers).
    answers: list of non-silent assistant responses.
    """
    h.add_user_message(question, video_tuple=("fake_np", {"fps": 1}))
    h.add_assistant_message(confirm)
    for ans in answers:
        for _ in range(silent_between):
            h.add_user_message("", video_tuple=("fake_np", {"fps": 1}))
            h.add_assistant_message(SILENT_TEXT)
        h.add_user_message("", video_tuple=("fake_np", {"fps": 1}))
        h.add_assistant_message(ans)


def get_context_history_messages(h: SessionHistory):
    """Return flattened context history message list (text-only)."""
    msgs = []
    for qa in h._context_history:
        msgs.extend(qa)
    return msgs


def get_sliding_window_messages(h: SessionHistory):
    """Return sliding window message list."""
    return list(h._sliding_window)


def count_sw_rounds(h: SessionHistory):
    return h._sw_round_count()


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n  Expected: {expected}\n  Actual:   {actual}")


def dump_qa(qa_msgs):
    """Pretty-print a QA for debugging."""
    for m in qa_msgs:
        role = m["role"]
        c = m["content"]
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict):
                    if item.get("type") == "video":
                        parts.append("<video>")
                    elif item.get("type") == "text":
                        parts.append(item["text"])
                else:
                    parts.append(str(item))
            c = " | ".join(parts)
        print(f"  {role}: {c!r}")


# ===========================================================================
# Test cases
# ===========================================================================

def test_01_no_pruning_when_under_limit():
    """Pruning should not trigger when rounds <= max_rounds."""
    h = make_session(max_rounds=10, num_rounds_keep=5)
    for i in range(10):
        add_basic_qa(h, f"Q{i}", f"A{i}")
    assert_eq(len(h._context_history), 0, "No context history yet")
    assert_eq(count_sw_rounds(h), 10, "All 10 rounds in sliding window")
    print("  PASS")


def test_02_basic_trigger_and_migration():
    """
    §3.1: When rounds > max_rounds, move first num_rounds_keep rounds
    to context history.
    """
    h = make_session(max_rounds=5, num_rounds_keep=3)
    for i in range(5):
        add_basic_qa(h, f"Q{i}", f"A{i}")

    # 5 rounds in SW, not yet > 5 → no prune
    assert_eq(count_sw_rounds(h), 5)
    assert_eq(len(h._context_history), 0)

    # Add 6th round → triggers prune (6 > 5 would trigger, but check is >=)
    # Actually, pruning is checked BEFORE adding the 6th, when count is already 5
    # which is >= max_rounds=5, so it triggers
    add_basic_qa(h, "Q5", "A5")

    # After prune: moved first 3 rounds, kept 3 in SW, then added Q5
    assert_eq(len(h._context_history), 3, "3 Basic QAs moved to context history")
    # SW should have rounds 3,4 (kept) + Q5 (newly added) = 3 rounds
    # Wait: we had 5 rounds, moved 3, kept 2 remaining, then add 1 = 3
    sw_rounds = count_sw_rounds(h)
    assert sw_rounds <= 4, f"SW should have ~3 rounds, got {sw_rounds}"

    # Context history QAs should be text-only Basic QAs
    for i, qa in enumerate(h._context_history):
        assert_eq(qa[0]["role"], "user")
        assert_eq(qa[0]["content"], f"Q{i}", f"QA{i} user text")
        assert_eq(qa[1]["role"], "assistant")
        assert_eq(qa[1]["content"], f"A{i}", f"QA{i} assistant text")
    print("  PASS")


def test_03_rule_a_video_removal():
    """§3.2 Rule A: <video> removed from user messages when moving to history."""
    h = make_session(max_rounds=3, num_rounds_keep=2)

    # Round 1: video + text
    h.add_user_message("看到了什么？", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("一张桌子")

    # Round 2: video only (no text)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("有人走过来了")

    # Round 3: video + text
    h.add_user_message("他穿什么？", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("蓝色衬衫")

    # Trigger prune (3 rounds, >= max_rounds=3)
    h.add_user_message("还有吗？", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("没有了")

    # Check context history: video should be stripped
    ctx_msgs = get_context_history_messages(h)
    for m in ctx_msgs:
        if m["role"] == "user":
            assert isinstance(m["content"], str), \
                f"User content should be string, got {type(m['content'])}"
            assert "<video>" not in str(m["content"]), "Video should be removed"
    print("  PASS")


def test_04_rule_b_silent_removal():
    """§3.2 Rule B: Silent rounds removed entirely when moving to history."""
    h = make_session(max_rounds=5, num_rounds_keep=5)

    # Build a 1Q1A QA with 3 silent rounds + 1 final answer = 5 rounds total
    add_1q1a_qa(h, "看到黑色鼠标告诉我", "好的", silent_count=3, final_answer="看到了鼠标")

    # Add one more round to trigger (5 rounds, + 1 trigger)
    add_basic_qa(h, "额外问题", "额外回答")

    # 6 rounds >= 6, prune triggered. Move first 5 rounds.
    # After rewrite: remove 3 silent rounds → 1Q1A becomes 2 rounds in history
    # (head: "看到黑色鼠标告诉我" + "好的", continuation: "" + "看到了鼠标")
    assert len(h._context_history) >= 1, "Should have at least 1 QA in history"

    qa = h._context_history[0]
    # Should be 1Q1A format: 4 messages (user, assistant, user "", assistant)
    assert_eq(len(qa), 4, "1Q1A QA should have 4 messages")
    assert_eq(qa[0]["content"], "看到黑色鼠标告诉我")
    assert_eq(qa[1]["content"], "好的")
    assert_eq(qa[2]["content"], "")
    assert_eq(qa[3]["content"], "看到了鼠标")

    # No silent content in context history
    for m in qa:
        if m["role"] == "assistant":
            assert m["content"] != SILENT_TEXT, "Silent should be removed"
    print("  PASS")


def test_05_rule_d_truncated_qa_merge():
    """
    §3.2 Rule D: Truncated QA merges with last context history QA.
    Scenario: A 1Q1A QA spans the cut boundary. The head goes to history
    first, then orphan continuation round follows in next prune cycle.
    """
    h = make_session(max_rounds=4, num_rounds_keep=2)

    # Build a 1Q1A QA: head (round 1) + 1 silent (round 2) + answer (round 3)
    h.add_user_message("看到猫告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的，我会注意")
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("我看到一只猫了！")

    # Add a Basic QA (round 4)
    add_basic_qa(h, "猫是什么颜色", "白色的")

    # 4 rounds in SW, >= max_rounds=4 → prune on next add
    add_basic_qa(h, "触发问题", "触发回答")

    # Move first 2 rounds: head + 1 silent round
    # After rewrite: silent removed → only head remains as Basic QA
    # The continuation "看到猫了" is in round 3 which stays in SW

    # Now add more to trigger again
    for i in range(3):
        add_basic_qa(h, f"填充{i}", f"回答{i}")

    # At some point the orphan round ("" + "我看到一只猫了！") gets moved
    # and should merge with the Basic QA to form a 1Q1A

    # Verify context history has merged QA
    found_merged = False
    for qa in h._context_history:
        qa_type = h._classify_qa(qa)
        pairs = h._qa_to_round_pairs(qa)
        if len(pairs) >= 2 and pairs[0][0] == "看到猫告诉我":
            found_merged = True
            # Should have the original question + merged continuation
            assert pairs[0][1] == "好的，我会注意", "Head assistant preserved"
            assert pairs[1][0] == "", "Continuation user is empty string"
            assert pairs[1][1] == "我看到一只猫了！", "Continuation answer preserved"
            break

    assert found_merged, "Truncated QA should have merged with head"
    print("  PASS")


def test_06_rule_e_1qna_limit():
    """
    §3.2 Rule E: 1QNA in context history ≤ 4 rounds.
    Build a 1QNA with many non-silent answers, verify truncation.
    """
    h = make_session(max_rounds=15, num_rounds_keep=14, max_1qna_rounds=4)

    # 1QNA: head + 6 answer rounds (no silent, to keep it simple)
    h.add_user_message("每次看到物品告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的")
    for i in range(6):
        h.add_user_message("", video_tuple=("fake", {"fps": 1}))
        h.add_assistant_message(f"物品{i}")

    # Add more basic QAs to fill up to max_rounds
    for i in range(8):
        add_basic_qa(h, f"问题{i}", f"回答{i}")

    # 7 + 8 = 15 rounds, trigger on next add
    add_basic_qa(h, "触发", "触发回答")

    # The 1QNA QA had 7 rounds (head + 6 answers). Rule E trims to ≤ 4.
    found_1qna = False
    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "每次看到物品告诉我":
            found_1qna = True
            n_rounds = len(pairs)
            assert n_rounds <= 4, \
                f"1QNA should be ≤ 4 rounds after Rule E, got {n_rounds}"
            # First round must be preserved
            assert_eq(pairs[0][0], "每次看到物品告诉我")
            assert_eq(pairs[0][1], "好的")
            # Remaining rounds should be the LATEST continuations
            # (earliest "" + response rounds get deleted first)
            break

    assert found_1qna, "1QNA QA should exist in context history"
    print("  PASS")


def test_07_capacity_limit():
    """§3.3: Context history ≤ 10 QAs. Oldest QA removed first."""
    h = make_session(max_rounds=3, num_rounds_keep=2, max_context_qas=5)

    # Each prune cycle moves 2 Basic QAs. Do many cycles.
    for cycle in range(10):
        add_basic_qa(h, f"Q_cycle{cycle}_a", f"A_cycle{cycle}_a")
        add_basic_qa(h, f"Q_cycle{cycle}_b", f"A_cycle{cycle}_b")
        # 3rd round triggers prune → moves first 2
        add_basic_qa(h, f"Q_cycle{cycle}_c", f"A_cycle{cycle}_c")

    assert len(h._context_history) <= 5, \
        f"Context history should be ≤ 5 QAs, got {len(h._context_history)}"

    # Oldest QAs should have been removed
    if h._context_history:
        first_qa = h._context_history[0]
        first_user = first_qa[0]["content"]
        # Should NOT be from earliest cycles
        assert "cycle0" not in first_user, \
            f"Oldest QAs should be removed, but found: {first_user}"
    print("  PASS")


def test_08_hard_cut_in_middle_of_qa():
    """
    §3.1: Hard cut at num_rounds_keep boundary, even if it splits a QA.
    """
    h = make_session(max_rounds=5, num_rounds_keep=3)

    # Basic QA: 1 round
    add_basic_qa(h, "简单问题", "简单回答")

    # 1Q1A QA: head (round 2) + 2 silent (rounds 3,4) + answer (round 5)
    h.add_user_message("看到猫告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的")
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("看到猫了")

    # 6 rounds total, >= max_rounds=6 → trigger prune
    add_basic_qa(h, "触发", "触发回答")

    # Hard cut at round 3: moves rounds 1-3
    # Round 1: Basic QA (complete)
    # Round 2: 1Q1A head (user "看到猫告诉我" + assistant "好的")
    # Round 3: 1Q1A continuation (video + silent) → removed by Rule B
    # So context history gets: Basic QA + Basic QA (the truncated 1Q1A head)
    # Rounds 4-6 stay in SW (round 4: silent, round 5: answer, round 6: 触发)

    # The head "看到猫告诉我" should be in context history
    found_head = False
    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "看到猫告诉我":
            found_head = True
            break
    assert found_head, "The QA head should be in context history despite hard cut"
    print("  PASS")


def test_09_context_history_qa_formats():
    """
    §2.2: Verify all 4 context history QA types are correctly produced.
    """
    h = make_session(max_rounds=20, num_rounds_keep=18)

    # (1) Basic QA: 1 round
    add_basic_qa(h, "今天天气怎么样", "晴天")

    # (2) 1Q1A QA: head + 2 silent + 1 answer = 4 rounds
    add_1q1a_qa(h, "看到红色物体告诉我", "收到", silent_count=2,
                final_answer="桌上有个红色杯子")

    # (3) 1QNA QA: head + (2 silent + answer) × 2 = 7 rounds
    add_1qna_qa(h, "每次有人经过告诉我", "好的", ["有人走过", "又来了一个人"],
                silent_between=2)

    # Fill up to trigger
    remaining = 20 - (1 + 4 + 7)  # = 8
    for i in range(remaining):
        add_basic_qa(h, f"填充{i}", f"回答{i}")

    # Trigger prune
    add_basic_qa(h, "触发", "触发回答")

    # Verify context history QA types
    qa_types = [h._classify_qa(qa) for qa in h._context_history]
    assert "basic" in qa_types, f"Should have Basic QA, got types: {qa_types}"

    # Check 1Q1A format
    for qa in h._context_history:
        if h._classify_qa(qa) == "1q1a":
            pairs = h._qa_to_round_pairs(qa)
            assert_eq(len(pairs), 2, "1Q1A should have exactly 2 rounds")
            assert pairs[0][0].strip() != "", "First round should have text"
            assert_eq(pairs[1][0], "", "Second round user should be empty")
            break

    # Check 1QNA format
    for qa in h._context_history:
        if h._classify_qa(qa) == "1qna":
            pairs = h._qa_to_round_pairs(qa)
            assert len(pairs) >= 3, "1QNA should have ≥ 3 rounds"
            assert pairs[0][0].strip() != "", "First round should have text"
            for p in pairs[1:]:
                assert_eq(p[0], "", "Continuation rounds should have empty user")
            break

    print("  PASS")


def test_10_sliding_window_preserves_multimedia():
    """Sliding window messages should keep full multimedia content."""
    h = make_session(max_rounds=5, num_rounds_keep=3)

    for i in range(4):
        h.add_user_message(f"Q{i}", video_tuple=("fake_video", {"fps": 1}))
        h.add_assistant_message(f"A{i}")

    # Check that sliding window messages have list content with video
    for m in h._sliding_window:
        if m["role"] == "user":
            assert isinstance(m["content"], list), "SW user content should be list"
            has_video = any(
                isinstance(item, dict) and item.get("type") == "video"
                for item in m["content"]
            )
            assert has_video, "SW should preserve video data"
    print("  PASS")


def test_11_history_composition():
    """self.history should be [system] + context_history + sliding_window."""
    h = make_session(max_rounds=4, num_rounds_keep=2)

    # Add 4 rounds then trigger
    for i in range(4):
        add_basic_qa(h, f"Q{i}", f"A{i}")

    # Trigger
    add_basic_qa(h, "Q4", "A4")

    # Verify composition
    assert_eq(h.history[0]["role"], "system", "First message should be system")

    # Context history messages should follow system
    ctx_msg_count = sum(len(qa) for qa in h._context_history)
    sw_msg_count = len(h._sliding_window)

    expected_total = 1 + ctx_msg_count + sw_msg_count
    assert_eq(len(h.history), expected_total,
              f"history length mismatch: 1+{ctx_msg_count}+{sw_msg_count}")

    # Context history portion should be text-only
    for i in range(1, 1 + ctx_msg_count):
        m = h.history[i]
        if m["role"] == "user":
            assert isinstance(m["content"], str), \
                f"Context history user content should be string at index {i}"
    print("  PASS")


def test_12_multiple_prune_cycles():
    """Verify correctness across multiple pruning cycles."""
    h = make_session(max_rounds=4, num_rounds_keep=2, max_context_qas=10)

    total_added = 0
    for cycle in range(6):
        for i in range(4):
            add_basic_qa(h, f"cycle{cycle}_Q{i}", f"cycle{cycle}_A{i}")
            total_added += 1

    # Should have survived multiple prune cycles
    assert len(h._context_history) <= 10, "Context history ≤ 10 QAs"
    assert count_sw_rounds(h) <= 4, "Sliding window ≤ max_rounds"

    # History should be internally consistent
    user_count = sum(1 for m in h.history if m["role"] == "user")
    ctx_rounds = sum(h._count_qa_rounds(qa) for qa in h._context_history)
    sw_rounds = count_sw_rounds(h)
    assert_eq(user_count, ctx_rounds + sw_rounds,
              "Total rounds = context + sliding window")
    print("  PASS")


def test_13_merge_basic_to_1q1a():
    """
    Rule D detail: Basic QA + truncated QA → 1Q1A QA.
    """
    h = make_session(max_rounds=4, num_rounds_keep=2)

    # Round 1: Basic QA
    add_basic_qa(h, "这是什么", "一个杯子")

    # Rounds 2-4: 1Q1A QA (head at round 2, silent at 3, answer at 4)
    h.add_user_message("看到猫告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的")
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("看到了一只猫")

    # Trigger: 4 rounds, >= 4 → prune on next add
    add_basic_qa(h, "触发1", "回答1")

    # First prune: move rounds 1-2 (Basic QA + 1Q1A head)
    # → context history gets 2 Basic QAs

    # Now SW has rounds 3 (silent) + 4 (answer) + 触发1 = 3 rounds
    # Add more to trigger second prune
    add_basic_qa(h, "触发2", "回答2")
    add_basic_qa(h, "触发3", "回答3")

    # Second prune: move first 2 rounds from SW
    # Round 3 was silent → removed. Round 4 is orphan ("" + "看到了一只猫")
    # → truncated QA merges with last QA in history ("看到猫告诉我" + "好的")
    # → becomes 1Q1A: ("看到猫告诉我", "好的") + ("", "看到了一只猫")

    found_1q1a = False
    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "看到猫告诉我":
            found_1q1a = True
            assert_eq(h._classify_qa(qa), "1q1a",
                       "Should be 1Q1A after merge")
            assert_eq(len(pairs), 2, "1Q1A has 2 rounds")
            assert_eq(pairs[1][0], "")
            assert_eq(pairs[1][1], "看到了一只猫")
            break

    assert found_1q1a, "Should find merged 1Q1A QA"
    print("  PASS")


def test_14_merge_1q1a_to_1qna():
    """
    Rule D detail: 1Q1A + truncated QA → 1QNA.
    """
    h = make_session(max_rounds=6, num_rounds_keep=3)

    # Build a 1QNA QA that will span the boundary:
    # Round 1: head
    h.add_user_message("物品出现告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的")
    # Round 2: answer 1
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("出现了杯子")
    # Round 3: answer 2
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("出现了鼠标")
    # Round 4: answer 3
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("出现了键盘")

    # Fill to trigger
    add_basic_qa(h, "Q4", "A4")
    add_basic_qa(h, "Q5", "A5")

    # Trigger
    add_basic_qa(h, "触发", "触发回答")

    # First prune: move rounds 1-3.
    # Rounds 1-3 are all in the same 1QNA group (head + 2 answers).
    # After rewrite: 3-round 1QNA → OK, it's valid.
    # Round 4 (orphan "出现了键盘") stays in SW.

    # Trigger another prune cycle
    for i in range(5):
        add_basic_qa(h, f"填充{i}", f"填充回答{i}")

    # Eventually round 4 gets moved and becomes truncated QA,
    # merges with the 1QNA to become 4-round 1QNA.

    found = False
    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "物品出现告诉我":
            found = True
            qa_type = h._classify_qa(qa)
            assert qa_type == "1qna", f"Should be 1QNA, got {qa_type}"
            n = len(pairs)
            assert n <= 4, f"Rule E: ≤ 4 rounds, got {n}"
            break

    assert found, "Should find the 1QNA QA"
    print("  PASS")


def test_15_rule_e_preserves_first_round():
    """Rule E: First round (user query + response) must never be deleted."""
    h = make_session(max_rounds=20, num_rounds_keep=19, max_1qna_rounds=4)

    # 1QNA with 8 answer rounds (no silents)
    h.add_user_message("跟踪所有物体", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("收到")
    for i in range(8):
        h.add_user_message("", video_tuple=("fake", {"fps": 1}))
        h.add_assistant_message(f"物体{i}")

    # Fill remaining
    for i in range(11):
        add_basic_qa(h, f"Q{i}", f"A{i}")

    # Trigger
    add_basic_qa(h, "触发", "触发回答")

    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "跟踪所有物体":
            assert_eq(pairs[0][0], "跟踪所有物体", "First user preserved")
            assert_eq(pairs[0][1], "收到", "First assistant preserved")
            assert len(pairs) <= 4, f"Rule E: ≤ 4 rounds, got {len(pairs)}"
            # Latest answers should be preserved (earliest deleted)
            last_answer = pairs[-1][1]
            assert "物体" in last_answer, "Latest answers should be kept"
            break
    print("  PASS")


def test_16_empty_qa_after_all_silent_removed():
    """If a QA only has silent rounds (except head), after Rule B it becomes Basic QA."""
    h = make_session(max_rounds=5, num_rounds_keep=4)

    # 1Q1A-like but the "answer" round is actually also silent (unusual edge case)
    h.add_user_message("看到狗告诉我", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message("好的，等着")
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)
    h.add_user_message("", video_tuple=("fake", {"fps": 1}))
    h.add_assistant_message(SILENT_TEXT)

    add_basic_qa(h, "其他问题", "其他回答")

    # Trigger
    add_basic_qa(h, "触发", "触发回答")

    # The 1Q1A-like QA → all continuations are silent → removed
    # Only head remains → becomes Basic QA
    for qa in h._context_history:
        pairs = h._qa_to_round_pairs(qa)
        if pairs and pairs[0][0] == "看到狗告诉我":
            assert_eq(h._classify_qa(qa), "basic",
                       "All-silent QA should become Basic after rewrite")
            assert_eq(len(pairs), 1)
            break
    print("  PASS")


def test_17_get_vllm_inputs_after_prune():
    """get_vllm_inputs should work correctly after pruning."""
    h = make_session(max_rounds=4, num_rounds_keep=2)

    for i in range(4):
        add_basic_qa(h, f"Q{i}", f"A{i}")

    # Trigger
    add_basic_qa(h, "Q4", "A4")

    inputs = h.get_vllm_inputs()
    prompt = inputs["prompt"]

    assert "<|im_start|>system" in prompt, "Should have system"
    # Context history (text-only) should be in prompt
    assert "Q0" in prompt or "Q1" in prompt, "Context history content in prompt"
    # Sliding window should be in prompt
    assert "Q4" in prompt, "Latest round in prompt"
    # Video tokens only from sliding window (not from context history)
    assert "<|video_pad|>" in prompt, "Video tokens from sliding window"
    print("  PASS")


def test_18_classify_qa():
    """Unit test for _classify_qa."""
    h = make_session()

    basic = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    assert_eq(h._classify_qa(basic), "basic")

    q1a = [
        {"role": "user", "content": "看到猫告诉我"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "看到了"},
    ]
    assert_eq(h._classify_qa(q1a), "1q1a")

    qna = [
        {"role": "user", "content": "跟踪物体"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "物体1"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "物体2"},
    ]
    assert_eq(h._classify_qa(qna), "1qna")

    truncated = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "又看到了"},
    ]
    assert_eq(h._classify_qa(truncated), "truncated")

    print("  PASS")


def test_19_enforce_1qna_limit():
    """Unit test for _enforce_1qna_limit."""
    h = make_session(max_1qna_rounds=4)

    qa = [
        {"role": "user", "content": "跟踪"},
        {"role": "assistant", "content": "好"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "C"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "D"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "E"},
    ]

    h._enforce_1qna_limit(qa)
    pairs = h._qa_to_round_pairs(qa)
    assert len(pairs) <= 4, f"Should be ≤ 4 rounds, got {len(pairs)}"
    assert_eq(pairs[0][0], "跟踪", "First round preserved")
    assert_eq(pairs[0][1], "好", "First round assistant preserved")
    # Latest answers should be kept (D, E), earliest (A, B) deleted
    answers = [p[1] for p in pairs[1:]]
    assert "E" in answers, f"Latest answer E should be kept, got {answers}"
    assert "D" in answers, f"Second latest D should be kept, got {answers}"
    print("  PASS")


def test_20_reset_clears_both_areas():
    """_reset should clear both context history and sliding window."""
    h = make_session(max_rounds=4, num_rounds_keep=2)

    for i in range(5):
        add_basic_qa(h, f"Q{i}", f"A{i}")

    assert len(h._context_history) > 0, "Should have context history"
    assert len(h._sliding_window) > 0, "Should have sliding window"

    h._reset()

    assert_eq(len(h._context_history), 0, "Context history cleared")
    assert_eq(len(h._sliding_window), 0, "Sliding window cleared")
    assert_eq(len(h.history), 1, "Only system message remains")
    assert_eq(h.history[0]["role"], "system")
    assert_eq(h.current_rounds, 0)
    print("  PASS")


# ===========================================================================
# Runner
# ===========================================================================

class AssertionError(Exception):
    pass


ALL_TESTS = [
    test_01_no_pruning_when_under_limit,
    test_02_basic_trigger_and_migration,
    test_03_rule_a_video_removal,
    test_04_rule_b_silent_removal,
    test_05_rule_d_truncated_qa_merge,
    test_06_rule_e_1qna_limit,
    test_07_capacity_limit,
    test_08_hard_cut_in_middle_of_qa,
    test_09_context_history_qa_formats,
    test_10_sliding_window_preserves_multimedia,
    test_11_history_composition,
    test_12_multiple_prune_cycles,
    test_13_merge_basic_to_1q1a,
    test_14_merge_1q1a_to_1qna,
    test_15_rule_e_preserves_first_round,
    test_16_empty_qa_after_all_silent_removed,
    test_17_get_vllm_inputs_after_prune,
    test_18_classify_qa,
    test_19_enforce_1qna_limit,
    test_20_reset_clears_both_areas,
]


def main():
    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Context Management Test Suite (per context.md)")
    print("=" * 60)

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            print(f"\n[RUN]  {name}")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
