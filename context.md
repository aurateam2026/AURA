# 上下文管理系统定义

## 1. 基本概念

| 术语 | 定义 |
|------|------|
| **对话轮 (Round)** | 一个 user 输入 + 一个 assistant 输出 |
| **QA** | 一个完整的问答单元，包含一个或多个对话轮 |
| **Sliding Window** | 保留完整多模态信息（含 `<video>` 和 `<\|silent\|>`）的近期对话窗口 |
| **超长 Context History** | 移除了视频和静默信息的压缩历史记录 |

---

## 2. QA 类型定义

### 2.1 Sliding Window 中的 QA（三种）

#### (1) Basic QA — 仅 1 个对话轮

**结构**：`(user query, assistant response)`

用户提出一个问题，assistant 直接回答，无后续视频监控。

**示例**：

```json
{"role": "user", "content": ["<video>", {"type": "text", "text": "现在办公室是一种什么样的氛围？"}]},
{"role": "assistant", "content": "现在的氛围比较安静。整个办公室里可以听到持续不断的键盘敲击声，但没有嘈杂的人声，看起来大家正在专注地工作。"}
```

#### (2) 1Q1A QA — 2 个或更多对话轮

**结构**：`(user query, assistant response)` + `("<video>", <|silent|>)` × N + `("<video>", assistant response)`

- **第 1 轮**：user query + assistant response（通常为任务确认/应答）
- **中间轮**：全部为 `"<video>"` + `<|silent|>`（持续监控但无事件触发）
- **最后 1 轮**：`"<video>"` + assistant response（事件触发，给出唯一一次回答）

**示例**：

```json
{"role": "user", "content": ["<video>", {"type": "text", "text": "当你看到一个黑色鼠标的时候，马上告诉我。"}]},
{"role": "assistant", "content": "好的，到时候我会提醒你。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "画面中清晰展示出一只黑色无线鼠标。"}
```

#### (3) 1QNA QA — 3 个或更多对话轮

**结构**：`(user query, assistant response)` + (`("<video>", <|silent|>)` 或 `("<video>", assistant response)`) × N + `("<video>", assistant response)`

- **第 1 轮**：user query + assistant response（通常为任务确认/应答）
- **中间轮**：`"<video>"` + `<|silent|>` 或 `"<video>"` + assistant response（多次事件触发，多次回答）
- **最后 1 轮**：`"<video>"` + assistant response

与 1Q1A 的区别：1QNA 在中间/末尾有 **多次** 非静默的 assistant response（即对同一个问题给出了多次回答）。

**示例**：

```json
{"role": "user", "content": ["<video>", {"type": "text", "text": "每次当镜头中出现一样物品的时候，告诉我是什么物品。"}]},
{"role": "assistant", "content": "已记录，到时候请留意我的提醒。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "一个黑色的包装盒上放着一个白色的无线耳机。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "一个黑色的鼠标出现在了包装盒的左侧。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "摄像头视角向右平移，只看到了耳机和鼠标的一部分。"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "一只手将一个绿色的盒子放到了鼠标和耳机的下面。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "它现在被放在鼠标和耳机的右下方，位置比较靠后。"},
{"role": "user", "content": ["<video>"]}, {"role": "assistant", "content": "<|silent|>"},
{"role": "user", "content": ["<video>"]},
{"role": "assistant", "content": "一个黑色的鼠标垫出现在了桌面上。"}
```

---

### 2.2 超长 Context History 中的 QA（四种）

从 sliding window 移入 context history 时，会经过改写（移除 `<video>` 和 `<|silent|>` 轮），因此格式更精简。

#### (1) Basic QA — 1 个对话轮

**结构**：`user query` + `assistant response`

**示例**：

```json
{"role": "user", "content": "这个鼠标是什么牌子的？"},
{"role": "assistant", "content": "这款鼠标是罗技（Logitech）品牌的，黑色的鼠标表面印有清晰的白色"logitech"字样，它单独放置在白色的桌面上。"}
```

```json
{"role": "user", "content": "当你看到一个白色纸杯的时候，马上就告诉我。"},
{"role": "assistant", "content": "没问题，到时候我会告诉你。"}
```

#### (2) 1Q1A QA — 恰好 2 个对话轮

**结构**：`user query` + `assistant response` + `""` + `assistant response`

**示例**：

```json
{"role": "user", "content": "当你看到一个白色纸杯的时候，马上就告诉我。"},
{"role": "assistant", "content": "没问题，到时候我会告诉你。"},
{"role": "user", "content": ""},
{"role": "assistant", "content": "一个白色的纸杯突然出现在画面中，位于米老鼠造型的笔筒前方。"}
```

#### (3) 1QNA QA — 3 个或更多对话轮

**结构**：`user query` + `assistant response` + (`""` + `assistant response`) × N（N ≥ 2）

**示例**：

```json
{"role": "user", "content": "每当镜头中的物品消失的时候，告诉我消失的是什么，并且告诉我还剩下的物品共有几件。"},
{"role": "assistant", "content": "好的，那到时候我会提醒你。"},
{"role": "user", "content": ""},
{"role": "assistant", "content": "手拿走了放在桌面上的透明签到盒，桌面上只剩下手机和贴纸两件物品。"},
{"role": "user", "content": ""},
{"role": "assistant", "content": "手拿走了手机，桌面完全清空。"}
```

#### (4) 被截断 QA — 1 个对话轮

当 1Q1A/1QNA QA 在 sliding window 中被硬切时，失去原始 user query 的后续轮会以此形式存在。

**结构**：`""` + `assistant response`

**示例**：

```json
{"role": "user", "content": ""},
{"role": "assistant", "content": "它现在被放在鼠标和耳机的右下方，位置比较靠后。"}
```

---

## 3. 上下文管理流程

整个系统分为 **Sliding Window** 和 **超长 Context History** 两部分。

### 3.1 触发条件与轮次迁移

当 sliding window 中的对话轮数量超过 `--max-rounds` 时，执行以下操作：

1. 将 sliding window 中 **前 `max-rounds - num-rounds-keep` 轮** 的对话移入超长 context history。
2. Sliding window 中仅保留 **最后 `num-rounds-keep` 轮** 的对话。
3. 如果 1Q1A 或 1QNA QA 被截断（即切割点落在 QA 中间），则**按轮数硬切**，不做特殊对齐。

### 3.2 移入 Context History 时的改写规则

对话从 sliding window 移入 context history 时，按以下规则依次改写：

#### 规则 A：移除 `<video>` 视频占位符

将 user 消息中的 `<video>` 移除。

| 改写前 | 改写后 |
|--------|--------|
| `{"role": "user", "content": ["<video>"]}` | `{"role": "user", "content": ""}` |
| `{"role": "user", "content": ["<video>", {"type": "text", "text": "这个人看起来多大？"}]}` | `{"role": "user", "content": "这个人看起来多大？"}` |

#### 规则 B：移除静默对话轮

如果 assistant 的回复仅包含 `<|silent|>`，则**整个对话轮**（包括对应的 user 输入和 assistant 回复）都需要被移除。

#### 规则 C：格式校验

改写后的每个 QA 必须符合 **2.2 节定义的四种 context history QA 类型** 之一。

#### 规则 D：与已有 QA 合并

被移入 context history 的**被截断 QA** 可以与 context history 中最后一个 QA 合并：

| Context History 中最后一个 QA | 新移入的内容 | 合并结果 |
|------|------|------|
| Basic QA（1 轮） | 被截断 QA（1 轮） | **1Q1A QA**（2 轮） |
| 1Q1A QA（2 轮） | 被截断 QA（1 轮） | **1QNA QA**（3 轮） |
| 1QNA QA（≥ 3 轮） | 被截断 QA（1 轮） | **1QNA QA**（轮数 +1） |

#### 规则 E：1QNA 轮数上限

超长 context history 中的 1QNA QA 总对话轮数**不能超过 4 轮**。如果超过：

1. 从最早的 `"" + assistant response` 对话轮开始删除。
2. 逐条删除，直到该 QA 总轮数 ≤ 4。
3. **不能删除**第 1 轮的 `user query + assistant response`。

### 3.3 Context History 容量限制

超长 context history 中最多保留 **10 个 QA**。当 QA 数量超过 10 个时：

1. 从**最早的 QA** 开始清除。
2. 每次清除一个**完整的 QA**（一个 QA 可能包含多个对话轮）。
3. 持续清除直到 QA 数量 ≤ 10。
