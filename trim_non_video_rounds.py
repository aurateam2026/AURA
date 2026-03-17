import json
import os

# ============================================================
# 请填写以下变量
INPUT_DIR  = "Streamo_v1/1qNa/9_add_sys_2p"   # 输入 JSON 文件所在目录
OUTPUT_DIR = "Streamo_v1/1qNa/10_trim_non_video_2rounds_2p"   # 处理后 JSON 文件的输出目录
N = 2             # 从后往前保留的"窗口外"非空 user 消息轮数
M = 3             # 每个"窗口外"非空 user 消息后最多保留的 assistant 消息数
# ============================================================


def limit_outside_assistants(convs: list, m: int) -> list:
    """
    对 convs 中每个"窗口外"的非空 user 消息，统计其后（在下一个非空 user 或
    视频帧之前）共有多少条 assistant 消息；若超过 m，只保留最后 m 条
    （同时移除多余的空占位 user 消息）。
    """
    result = []
    i = 0

    while i < len(convs):
        msg = convs[i]
        val = msg.get("value", "")

        # 遇到"窗口外"非空 user 消息 → 收集该组
        if msg.get("from") == "user" and "<video>" not in val and val != "":
            result.append(msg)
            i += 1

            # 收集 (前置空user或None, assistant) 槽位，直到下一个非空user/video user
            slots = []
            while i < len(convs):
                cur     = convs[i]
                cur_val = cur.get("value", "")

                if cur.get("from") == "assistant":
                    # 紧跟在非空user后面的第一个assistant（无前置空user）
                    slots.append((None, cur))
                    i += 1

                elif (cur.get("from") == "user"
                      and cur_val == ""
                      and "<video>" not in cur_val):
                    # 空占位user：期望后面紧跟一个assistant
                    if i + 1 < len(convs) and convs[i + 1].get("from") == "assistant":
                        slots.append((cur, convs[i + 1]))
                        i += 2
                    else:
                        # 孤立空user，直接保留
                        result.append(cur)
                        i += 1

                else:
                    # 下一个非空user或video user → 该组结束
                    break

            # 区分"计数槽位"（无<|silent|>，受 M 限制）
            #   和"静音槽位"（含<|silent|>，视为窗口内，始终保留，不占名额）
            countable_idx = [i for i, (_, asst) in enumerate(slots)
                             if "<|silent|>" not in asst.get("value", "")]
            silent_idx    = [i for i in range(len(slots))
                             if i not in countable_idx]

            if len(countable_idx) > m:
                keep_set = set(countable_idx[-m:]) | set(silent_idx)
            else:
                keep_set = set(range(len(slots)))  # 全部保留

            # 按原顺序重组
            kept_slots = [slots[i] for i in range(len(slots)) if i in keep_set]

            # 第一个保留槽位直接与前面的非空 user 配对，去掉其前置空 user
            if kept_slots and kept_slots[0][0] is not None:
                kept_slots[0] = (None, kept_slots[0][1])

            for (empty_u, asst) in kept_slots:
                if empty_u is not None:
                    result.append(empty_u)
                result.append(asst)

        else:
            result.append(msg)
            i += 1

    return result


def trim_sample(sample: dict, n: int, m: int) -> dict:
    """
    Step1: 从后往前找第 n 个不含 <video> 且不为空的 user 消息，
           保留该消息及其之后的全部内容（system 消息始终保留在开头）。
           若窗口外非空 user 消息不足 n 条，则不截断。

    Step2: 对保留下来的所有"窗口外"非空 user 消息，
           限制其后跟随的 assistant 消息数不超过 m。
    """
    convs = sample.get("conversations", [])

    # 分离 system 消息
    if convs and convs[0].get("from") == "system":
        sys_msg = convs[0]
        rest    = convs[1:]
    else:
        sys_msg = None
        rest    = convs

    # ── Step 1: N 截断 ──────────────────────────────────────
    non_video_user_positions = []
    for i in range(len(rest) - 1, -1, -1):
        msg = rest[i]
        val = msg.get("value", "")
        if msg.get("from") == "user" and "<video>" not in val and val != "":
            non_video_user_positions.append(i)

    if len(non_video_user_positions) >= n:
        cut_idx = non_video_user_positions[n - 1]
        rest    = rest[cut_idx:]

    # ── Step 2: M 限制 ──────────────────────────────────────
    rest = limit_outside_assistants(rest, m)

    # 重组
    new_convs = ([sys_msg] if sys_msg is not None else []) + rest
    new_sample = dict(sample)
    new_sample["conversations"] = new_convs
    return new_sample


def process_file(input_path: str, output_path: str, n: int, m: int) -> None:
    print(f"  读取: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total  = len(data)
    result = []
    for idx, sample in enumerate(data):
        result.append(trim_sample(sample, n, m))
        if (idx + 1) % 5000 == 0 or (idx + 1) == total:
            print(f"    进度: {idx + 1}/{total}", end="\r", flush=True)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"  保存: {output_path}")


def main():
    if not INPUT_DIR or not OUTPUT_DIR:
        raise ValueError("请先填写 INPUT_DIR 和 OUTPUT_DIR 变量")

    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not json_files:
        print("输入目录中未找到 JSON 文件")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"共找到 {len(json_files)} 个 JSON 文件，N={N}，M={M}\n")

    for fname in json_files:
        input_path  = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)
        process_file(input_path, output_path, N, M)

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
