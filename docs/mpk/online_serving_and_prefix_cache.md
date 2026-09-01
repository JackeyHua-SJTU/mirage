# MPK Online Serving 机制解剖 & Prefix Cache 设计（OIPL）

> 基于 `mpk @ 0cd41ecf`（2026-08-27）。所有行号链接对应该 commit 的本仓库文件，点击可跳转。
> 结论经两轮多智能体对抗验证（机制 54 agent / 设计评审 60 agent），被推翻的初稿断言见 [§10](#10-评审记录被推翻的初稿断言)。

**目录**：[0 总览](#0-总览) · [1 状态字典](#1-状态字典每个变量是干什么的) · [2 交互协议与时序](#2-交互协议时间节点同步性) · [3 执行模型](#3-执行模型谁在什么时候跑-prepare_next_batch) · [4 内存序与 KV 可见性](#4-内存序与-kv-可见性链) · [5 #666 复盘](#5-issue-666-复盘两个方案为什么都卡住) · [6 OIPL 设计](#6-设计oipl所有权反转的页生命周期) · [7 正确性论证](#7-正确性论证) · [8 兼容性](#8-兼容性) · [9 分阶段落地](#9-分阶段落地) · [10 评审记录](#10-评审记录被推翻的初稿断言) · [11 不确定项](#11-遗留不确定项)

---

## 0. 总览

MODE_ONLINE_PINNED 的 online serving 是一个**纯轮询、双向异步**的协议：

- CPU 与 GPU 之间**没有任何一次同步调用**（无 cudaEvent / sync 等待对方；HEAD 如此，#754 会引入一处例外，见 §8）。全部交互经 pinned host memory 上的 lock-free ring + release/acquire 完成。
- GPU 侧全部调度决策（组批、准入、分页、完成检出）由**单个 scheduler 线程**在每次 task-graph 迭代之间串行执行，且该时刻**全 GPU 空转**（由 qwen3 图的链式单叶形状保证的"事实全局屏障"，见 §3）。
- Prefix cache 的可行入口：**attention 从不读 `step`，序列长度/RoPE/因果掩码完全由页表推导**（见 §1.2 `paged_kv_last_page_len_buffer` 与 §5）。给请求预置正确页表 = 命中的充分条件；缺的是三条通道（页导出/导入/归还）和一个所有权规则——这就是 §6 的设计。

调用链一图流：

```
POST /v1/chat/completions        launch_server.py
  └─ LLMEngine.submit            llm_engine.py        (tokenize、分配 rid)
       └─ OnlinePinnedRuntime.submit                  (占 ring 槽、写 inbox、ready=1)
            └─ [pinned 请求 ring] ─────────────► GPU prepare_next_batch Step 4 准入
                                                       │ Step 3 组批/发 token 预算
                                                       │ Step 1 收 output、检出完成
  ◄─ pinned_step[row] 每迭代 release ◄────────────────┘
  ◄─ [pinned 完成 ring] {rid, row, final_step}
  └─ _StreamingMonitor 2ms 轮询 → SSE
```

---

## 1. 状态字典：每个变量是干什么的

理解本节的钥匙是**六个索引空间**。几乎所有已知 bug 和本设计的全部改动，都可以描述为"某个值在空间之间搬运时出错/缺通道"。

### 1.1 六个索引空间

| 空间 | 取值范围（默认配置） | 含义 | 寿命 |
|---|---|---|---|
| **batch slot** `i` | [0, MBR=4) | 本迭代运行批里的座位号。每迭代 Step 3 重新压缩（活跃请求挤到前面），所以**同一请求的 slot 每迭代可能变**。attention task 的 `task_metadata.request_id` 就是它（= `bid.x`，[runtime.cc:343](../../src/kernel/runtime.cc#L343)） | 一次迭代 |
| **buffer row** `row` | [0, total_inflight=4) | 请求的"床位号"：token 缓冲区的行。**请求整个生命周期不变**，完成后回收复用 | 请求生命周期 |
| **ring slot** | cursor & 7 | 两个 pinned ring 的槽号（游标按位与掩码） | 一次 publish→consume |
| **token 位置** | [0, num_tokens≤8) | 本迭代喂给模型的扁平 token 批的下标，边界由 `qo_indptr_buffer` 定义 | 一次迭代 |
| **页槽**（flat） | [0, 批内总页数) | 扁平页表 `paged_kv_indices_buffer` 的下标，边界由 `paged_kv_indptr_buffer` 定义 | 一次迭代 |
| **物理页 id** | [0, MAX_PAGES=16) | KV cache 张量第二维的下标；**一个页 id 命名所有层的同一物理块**（见 1.2） | 进程生命周期 |

另有三个"编号"容易混淆：**rid**（CPU 分配的全局请求号，单调递增永不复用）↔ **row**（可复用的床位）↔ **slot**（每迭代变的座位）。`request_ids` 存 slot→row，`request_rids` 存 slot→rid，`pinned_rid_at_row` 存 row→rid——三个映射缺一不可，因为 CPU 只认 rid、attention 只认 slot、token 缓冲只认 row。

### 1.2 GPU device tensors（CPU 分配、双方可见，`meta_tensors[0..10]`）

分配于 [model_runner.py:153-168](../../python/mirage/engine/model_runner.py#L153-L168)，接线于 [persistent_kernel.cuh:1486-1503](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1486-L1503)。**写者只有 `prepare_next_batch` 与 `init_kernel`，所有计算 task 是纯读者**——这是 §6 设计要保持的核心性质。

| 变量 | 形状 / 索引 | 它是什么 |
|---|---|---|
| `step` | int32[n_req]，按 **row** | 该请求"已处理到哪"：**下一迭代要作为输入喂进模型的第一个 token 的下标**。prefill 期从 0（或 `initial_step`）每迭代最多 +mbt 推进到 prompt_len，decode 期每迭代 +1。Step 1 用 `step + num_tokens` 更新（[:481](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L481), [:494](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L494)）。注意：**attention 不读它**，它只服务于组批 |
| `tokens` | int64[n_req][MAX_SEQ]，按 **row** | 每请求完整 token 序列（prompt + 已生成）的 GPU 权威副本。准入时从 inbox 拷入 prompt（[:615-618](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L615-L618)），Step 1 把上一迭代的输出回写到 `step+j+1` 位置（[:487-491](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L487-L491)）。CPU 最终经 D2H 从这里取回结果 |
| `input_tokens` | int64[mbt][1]，按 **token 位置** | 本迭代喂给模型的扁平输入批：把各 slot 的待处理 token 按 slot 序拼接。每迭代由 Step 3/4 重建（[:573](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L573), [:640](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L640)） |
| `output_tokens` | int64[mbt][1]，按 **token 位置** | 模型（argmax task）为每个输入位置产出的下一个 token。下一次 `prepare_next_batch` 的 Step 1 按**上一迭代的** `qo_indptr` 边界读出、搬进 `tokens`（[:490](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L490)） |
| `new_token_nums` | int32[n_req] | **online 路径的未使用占位**。唯一消费点在 `MPK_SPEC_DECODE`（EAGLE3 offline）下：decode 每步接受的草稿 token 数（[:244](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L244)） |
| `prompt_length` | int32[n_req]，按 **row** | 该请求 prompt 的 token 数。准入写入（[:619](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L619)）；`remaining = prompt_length - step > 0` 即"仍在 prefill"（[:564](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L564)），也参与 done 判定（[:503-506](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L503-L506)：生成的 token 是 EOS **且已过 prompt** 才算完） |
| `qo_indptr_buffer` | int32[MBR+1]，按 **slot** | CSR 式边界数组：slot i 的输入 token 占 `input_tokens[qo_indptr[i] .. qo_indptr[i+1])`。attention 用它找到自己的 query 段，**且靠 `qo_indptr[i]==qo_indptr[i+1]` 判断空槽早退**（[multitoken_paged_attention_4_16.cuh:81-86](../../include/mirage/persistent_kernel/tasks/ampere/multitoken_paged_attention_4_16.cuh#L81-L86)）。哨兵 `[MBR]` = 本迭代总 token 数，argmax 读它 |
| `paged_kv_indptr_buffer` | int32[MBR+1]，按 **slot** | CSR 边界：slot i 的页占 `paged_kv_indices_buffer[indptr[i] .. indptr[i+1])`。哨兵 `[MBR]` = 批内总页数，Step 2 快照用（[:539-541](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L539-L541)） |
| `paged_kv_indices_buffer` | int32[MAX_PAGES]，按 **页槽** | **扁平页表**：按 slot 序拼接的物理页 id。attention 把逻辑位置 p 映射到物理页 `indices[indptr[i] + p/PAGE_SIZE]` |
| `paged_kv_last_page_len_buffer` | int32[MBR]，按 **slot** | slot i **最后一页已占的 token 数**。attention 用 `(num_pages-1)*PAGE_SIZE + last_page_len` 反推序列长度（[multitoken_paged_attention_hopper.cuh:103-107](../../include/mirage/persistent_kernel/tasks/hopper/multitoken_paged_attention_hopper.cuh#L103-L107)）——**这就是"attention 不读 step"的机制**。⚠ online 版用裸 `%` 计算（[:645-646](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L645-L646)），长度恰落页边界时得 0、比 offline 版少了 `(x==0)?PAGE_SIZE:x` 修正（[:337-341](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L337-L341)），默认配置下潜伏 |
| `paged_kv_indices_snapshot` | int32[MAX_PAGES] | **pinned 模式下的死代码**：MODE_OFFLINE/ONLINE 压缩页表时用它做全局内存快照（[:298](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L298), [:344](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L344)），pinned 模式改用 `__shared__` 快照（见 1.4），此 tensor 分配了、接线了、从不读——可复用 |

KV cache 本体：`k_cache`/`v_cache` 各是一个 `(num_layers, max_num_pages, page_size, num_kv_heads, head_dim)` 的 bf16 张量（[qwen3/builder.py:87-107](../../python/mirage/mpk/models/qwen3/builder.py#L87-L107)），逐层切片作为独立 task 输入挂进图（[:341-346](../../python/mirage/mpk/models/qwen3/builder.py#L341-L346)）——所以**一个物理页 id 同时命名全部 36 层的同一块**，导出/导入页 id 一次覆盖所有层。

### 1.3 Pinned 共享通道（`meta_tensors[11..22]`，真正的 CPU↔GPU 接口）

分配于 [model_runner.py:171-182](../../python/mirage/engine/model_runner.py#L171-L182)（`.pin_memory()`，GPU 直接解引用 host 地址）。字段定义在 [runtime_header.h:381-421](../../include/mirage/persistent_kernel/runtime_header.h#L381-L421)。

**请求 ring（CPU 生产 → GPU 消费），容量 8，槽号 = 游标 & 7：**

| 变量 | 它是什么 |
|---|---|
| `pinned_req_ready[8]` | 槽的握手旗：0=空，1=CPU 已发布完整请求。兼做占位判定（CPU 端 submit 见 `==0` 才占槽 [online_pinned_runtime.py:118](../../python/mirage/mpk/online_pinned_runtime.py#L118)）。GPU `ld.acquire.sys` 读（[:603](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L603)）、消费完 `st.release.sys` 清 0（[:624-625](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L624-L625)）。⚠ CPU 写 1 是 plain store 无 release（[:131-134](../../python/mirage/mpk/online_pinned_runtime.py#L131-L134)），靠 x86-TSO 兜底 |
| `pinned_req_request_id[8]` | 槽内请求的 **rid**（见 1.1 三种编号辨析） |
| `pinned_req_prompt_len[8]` | prompt token 数——GPU 据此决定从 inbox 拷多少 |
| `pinned_req_initial_step[8]` | 请求的**起始处理位置**：语义是"前 `initial_step` 个位置的 KV 已就绪，prefill 从这里开始"。今天恒 0（唯一调用方 [llm_engine.py:192](../../python/mirage/engine/llm_engine.py#L192) 不传参）。这是 prefix cache 预留的 hook，但**页通道缺失使它设非 0 就是错的**：Step 4 会给 `[0, initial_step)` 分配全新空页（[:648-651](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L648-L651) 从 `j=0` 起全部新取）然后又跳过这段的 prefill → attention 读垃圾 |
| `pinned_inbox_tokens[8][MAX_SEQ]` | **每槽独立的 prompt 信箱**：CPU 把 token id 写进自己槽的信箱（实为 host-to-host memcpy，[:126-128](../../python/mirage/mpk/online_pinned_runtime.py#L126-L128) 的 CUDA stream 包装是空转），GPU 准入时逐 token 拷进 `tokens[row]`。独立信箱使并发 submit 互不覆盖 |

**完成 ring（GPU 生产 → CPU 消费）：**

| 变量 | 它是什么 |
|---|---|
| `pinned_comp_ready[8]` | 握手旗，GPU 写完 payload 后 `st.release.sys` 置 1（[:515-518](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L515-L518)），CPU 消费后清 0。⚠ GPU 写入前**不检查槽是否空**——CPU 落后 >8 个完成就覆盖丢失（#754 加自旋修此） |
| `pinned_comp_request_id[8]` | 完成请求的 rid |
| `pinned_comp_buffer_row[8]` | 完成请求占的 row——CPU 拿它去 `tokens[row]` 取结果 |
| `pinned_comp_final_step[8]` | 完成时的 `step+num_tokens`：`tokens[row][0..final_step]` 是有效数据（prompt+生成）。CPU 切 `[prompt_len:]` 得输出 |

**其余 pinned 字段：**

| 变量 | 它是什么 |
|---|---|
| `pinned_shutdown[1]` | CPU→GPU 关机旗。GPU **只在空批分支**检查（[:679-685](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L679-L685)）——在飞请求存在时忽略关机 |
| `pinned_step[n_req]`（按 **row**） | GPU 每迭代对每活跃 row `st.release.sys` 其最新 step（[:498](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L498)）——**CPU 唯一的每步进度信号**，streaming 靠轮询它决定读多少新 token。⚠ row 复用时**无人重置**（唯一写者就是 :498），新请求的 monitor 会先看到上一任的 final_step → 跨请求泄漏（#754 修；同时 #754 把它兼做 CPU ack 通道：CPU 写 -1 = "读完了，row 可回收"） |
| `pinned_rid_at_row[n_req]`（按 **row**） | row→rid 反查表：准入写 rid（[:622](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L622)），完成写 -1（[:523](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L523)）。CPU 的 streaming monitor 靠线性扫描它发现"我的 rid 落在哪个 row"（[online_pinned_runtime.py:261-269](../../python/mirage/mpk/online_pinned_runtime.py#L261-L269)）。⚠ 完成即抹 → 快到没被扫到过的请求永远找不到 row（[llm_engine.py:57-65](../../python/mirage/engine/llm_engine.py#L57-L65) 的 Phase-1 `continue` 在完成检查之前） |

### 1.4 gpu_malloc 私有状态（**CPU 完全不可见**，[:1584-1603](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1584-L1603)）

这一栏是 #666 Option 2 "CPU has no knowledge" 的字面含义：不是延迟问题，是**通道不存在**。

| 变量 | 它是什么 |
|---|---|
| `request_ids[MBR]` int16 | **slot→row 映射**（-1=空槽）。⚠ 命名误导：存的不是 rid 是 row |
| `request_rids[MBR]` int32 | slot→rid 映射，完成时把 rid 报告给 CPU 用（[:510](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L510)） |
| `page_queue[MAX_PAGES]` + `page_queue_head/tail` | **空闲物理页 id 的 FIFO 环**。init 全满：`page_queue[i]=i, head=0, tail=MAX`（[:165-168](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L165-L168)）；`tail−head` 恰为空闲页数。head 处弹出分配（[:587-588](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L587-L588), [:649-650](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L649-L650)），tail 处压入回收（[:532-533](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L532-L533)）。⚠ 全文件 head 从不与 tail 比较——**无空池检查**（#747 补 trap） |
| `free_rows[MBR]` + `free_row_top` | **空闲 row 的 LIFO 栈**：准入弹出（[:613](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L613)），完成压回（[:522](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L522)）。LIFO + 同一次调用先 push 后 pop = "行复用 use-after-free"的来源（Step 1 刚释放的 row 在 Step 4 立刻发给新请求并覆写，CPU 还没读走旧数据） |
| `gpu_req_head` / `gpu_comp_tail` | GPU 侧两个 ring 游标（消费请求 ring / 生产完成 ring），单调递增。CPU 有自己的镜像游标（1.5），**双方从不交换游标**——一致性完全靠 ready 旗 |

另有 kernel 局部的 `__shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES]`（[:466](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L466)）：Step 2 把扁平页表快照进共享内存（[:539-545](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L539-L545)），Step 3 就地压缩时从快照读旧页避免自我覆盖——只活一次 `prepare_next_batch` 调用。

### 1.5 CPU 侧状态（[online_pinned_runtime.py](../../python/mirage/mpk/online_pinned_runtime.py) / [llm_engine.py](../../python/mirage/engine/llm_engine.py)）

| 变量 | 它是什么 |
|---|---|
| `_cpu_req_tail` | 请求 ring 生产者游标：下一个要写的槽。与 GPU 的 `gpu_req_head` 是同一个环的两端 |
| `_cpu_comp_head` | 完成 ring 消费者游标 |
| `_cpu_req_ack` | **死变量**：两处赋 0（[:64](../../python/mirage/mpk/online_pinned_runtime.py#L64), [:285](../../python/mirage/mpk/online_pinned_runtime.py#L285)），从不读 |
| `_waiting` | ring 满时的**无界**溢出队列，存 `(rid, token_ids, initial_step)`；`flush_waiting` 每次只搬一个回 ring（[:137-171](../../python/mirage/mpk/online_pinned_runtime.py#L137-L171)） |
| `_completions` | rid → (row, final_step)：drain 后的暂存区，等 `wait_for_request`/monitor 领取后 `release_request` 删除。⚠ 超时路径删 session 不删这里 → 无界泄漏 |
| `_ring_lock` / `_waiting_lock` / `_lock` | 槽预留序列化 / 溢出队列 / `_completions`（RLock）。⚠ submit 与 flush_waiting 之间存在 ABBA 反序（[:116→120](../../python/mirage/mpk/online_pinned_runtime.py#L116-L120) vs [:154→158](../../python/mirage/mpk/online_pinned_runtime.py#L154-L158)） |
| `_next_rid` / `_submit_lock` | rid 分配器。⚠ `+=` 在锁外（[llm_engine.py:187-188](../../python/mirage/engine/llm_engine.py#L187-L188)），并发可撞号 |
| `_StreamingMonitor._sessions` | rid → {queue, row, last_step, deadline}：**单个** 2ms 守护线程服务所有流（[llm_engine.py:100](../../python/mirage/engine/llm_engine.py#L100)），`last_step` 初始化为 `prompt_len-1` 使 prompt 不回吐（[:43](../../python/mirage/engine/llm_engine.py#L43)） |

线程清单：uvicorn 事件循环、非流式 executor 线程、每流一个 `_stream_bridge` 生产线程（[launch_server.py:90](../../python/mirage/engine/launch_server.py#L90)）、monitor（2ms）、drain（0.2ms，[:207](../../python/mirage/mpk/online_pinned_runtime.py#L207)）。"kernel 线程"**不常驻**：`split_worker_scheduler` 硬编码 true（[:1617](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1617)），launch 异步发射 `worker_kernel` + `scheduler_kernel` 两个 kernel（[:1813-1823](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1813-L1823)）后立即返回，megakernel 脱管运行，CPU 从不 join。

### 1.6 实例走查：两个请求的完整生命周期（rid / row / slot / page 一次看懂）

四个编号是四种物理约束各自逼出来的：**rid** 解决"row 会复用，CPU 需要永不重复的名字"；**row** 解决"`tokens[row][*]` 这类大缓冲不能每迭代搬家"；**slot** 解决"attention task 的 grid 编译期定死（`task_metadata.request_id = bid.x`，[runtime.cc:343](../../src/kernel/runtime.cc#L343)），task 只认位置不认身份"；**page** 解决"KV 不能按请求连续预分配"。三张映射表缝合它们：`request_ids[slot]=row`（⚠ 存的是 row 不是 rid）、`request_rids[slot]=rid`、`pinned_rid_at_row[row]=rid`。

为让页数字可见，下例取 `page_size=8`（真实默认 4096）、MBR=4、mbt=8、max_seq=32。初始（[init_kernel](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L143-L186)）：`free_rows=[0,1,2,3](top=4)`（LIFO 床位栈）、`page_queue=[0..15](head=0,tail=16)`（FIFO 空页队列）、`request_ids=[-1,-1,-1,-1]`。

1. **CPU 提交 rid=17（prompt 20 token）**：占请求 ring 一槽（ring slot，第三种"槽"，只活一次投递），token 写该槽 `pinned_inbox_tokens`，发布 `{rid=17, prompt_len=20, initial_step=0}`、`ready=1`。GPU 此刻一无所知——无通知机制，纯等下次轮询。
2. **迭代 N，Step 4 准入**（[:597-655](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L597-L655)）：弹床位 `row=free_rows[--top]`→**row 3**；inbox→`tokens[3][0..19]`，`prompt_length[3]=20`，`step[3]=0`，`pinned_rid_at_row[3]=17`；填**座位 0**：`request_ids[0]=3`、`request_rids[0]=17`；预算 `num_new_tokens=min(20,8)=8`→`input_tokens[0..7]=tokens[3][0..7]`，`qo_indptr=[0,8,8,8,8]`（哨兵 `[4]=8`＝本迭代总 token 数，argmax 直接读）；分页 `ceil(8/8)=1`→FIFO 弹**页 0**，`paged_kv_indices=[0]`，`paged_kv_indptr=[0,1,1,1,1]`。（`last_page_len=8`——A10.1 已修：四处产出统一走 `paged_kv_last_page_len()`，页边界返回 PAGE_SIZE 而非裸 `%` 的 0；此前默认 4096 页时长度到不了边界所以潜伏。）
3. **迭代 N 执行**：绑座位 0 的 attention task 读 `qo_indptr[0..1]` 拿 8 个 query、读页表 `[0]`，把位置 0..7 的 K/V 写进物理页 0（`dst_row = page_id*8 + p%8`；一个页 id 同时命名全部层的同一块）。座位 1/2/3 的 task 见 `qo_indptr[i]==qo_indptr[i+1]` 空转退出。**task 全程不知道 row=3、rid=17 的存在。**
4. **迭代 N+1，Step 1**：`row=request_ids[0]=3`，`step[3]+=8→8`，`st.release.sys(pinned_step[3]=8)`（CPU monitor 每 2ms 轮询它读增量 token）。prefill 每个位置都有 argmax 输出，但写回有 `step+j+1 >= prompt_len` 的门（[:487-489](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L487-L489)）——**第一个生成 token 是处理最后一段 prompt 的那次迭代"顺带"产生的**。Step 3：`remaining=12`，再吃 8；页需求 `ceil(16/8)=2`→旧页从 smem 快照拷回＋新弹页 1。
5. **CPU 提交 rid=18（prompt 5 token）**：迭代 N+1 的准入循环条件 `num_tokens < 8` 不满足（rid17 吃满 8）→**循环体不进，rid18 滞留 ring**。这是"长 prompt 饿死后来者"的最小复现：rid18 的准入延迟由 rid17 的 prompt 长度决定。
6. **迭代 N+2**：rid17 吃 `min(4,8)=4`，预算剩 4→rid18 获准入：**row 2、座位 1**、吃 `min(5,4)=4`、弹页 2。
7. **rid17 完成**（生成到位置 27 遇 EOS）：Step 1 检出 done→写完成 ring `{rid=17, row=3, final_step=27}` 并 release；还床（row 3 压回 `free_rows`、`pinned_rid_at_row[3]=-1`）；还页（压回 `page_queue` tail）；清座位。CPU drain 凭 **row** 去 `tokens[3][0..27]` 取数据、凭 **rid** 在 `_completions` 交给正确等待者。（HEAD 上同一次调用的 Step 4 可立刻复用 row 3 并覆写——#754 修的 use-after-free。）
8. **之后的 Step 3 压缩**：rid18 从座位 1 挪到座位 0，**row 始终是 2**，token/KV 零搬运。绑座位 0 的 task 从此服务 rid18。**这就是 slot 与 row 必须分开的原因：task 绑定是位置性的（编译期），数据存放是身份性的（运行期）。**

高频困惑三答：**(a) 为什么不能一个编号走天下**——用 row 当 slot 会让 CSR 数组出洞且 task 无法靠 `slot ≥ num_reqs` 批量空转；用 slot 当 row 要每迭代搬 tokens+KV；不设 rid 则 row 复用后 CPU 无法区分前后两任租户（`pinned_step` 不重置的跨请求泄漏 bug 正是"误把 row 当身份"的实例）。**(b) attention 为什么连 step 都不读**——它要的三样（本迭代的 query 段、历史 KV 位置、序列长度）全部由 slot 数组给出；这是 prefix cache 的命门：**预置页表＝伪造历史**，不动任何 task。**(c) 页表为什么每迭代重建**——它按 slot 序拼扁平数组，slot 重排页表就得跟着重排（借 `smem_kv_indices` 快照避免就地覆盖）；物理页 id 稳定，动的只是"谁的页列表占扁平数组哪一段"。

---

## 2. 交互协议、时间节点、同步性

一个流式请求的完整时间线（逐点判定同步/异步）：

| # | 事件 | 执行者 | 排序原语 | 延迟性质 |
|---|---|---|---|---|
| 1 | tokenize + 分配 rid | 请求线程 | tokenizer 锁；rid 在锁外（bug） | 阻塞 µs–ms |
| 2 | 占槽 + 写 inbox + 发布 | 请求线程 | `_ring_lock`；**发布是 4 个无序 plain store**（[:131-134](../../python/mirage/mpk/online_pinned_runtime.py#L131-L134)），CPU 侧无 release | 非阻塞 |
| 3 | **GPU 准入**：Step 4 drain ring | scheduler 单线程 | `ld.acquire.sys(ready)` 门控 payload 读 | **轮询，粒度=1 次迭代**。准入延迟 ∈ [0, 1 迭代]；空闲时 GPU 在**全图空跑**（GEMM 不早退），唤醒延迟也是一整次前向（[:678-686](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L678-L686)） |
| 4 | 每步进度 | scheduler | `st.release.sys(pinned_step[row])`（[:498](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L498)） | 每迭代 1 次 |
| 5 | monitor 读增量 token | monitor 线程 | plain 读 + D2D clone + D2H `.tolist()` | 轮询 2ms |
| 6 | 完成发布 | scheduler | payload plain store → `st.release.sys(comp_ready)` | — |
| 7 | CPU drain | drain/monitor/waiter | plain `.item()` 读（x86-TSO 兜底） | 轮询 0.2ms / 2ms / 0.1ms |
| 8 | shutdown | — | plain 写旗；GPU 仅空批时查 | 轮询 |

**要点**：准入、分页、进度/完成发布**全部只发生在迭代间隙**；CPU 的观测节拍是 0.2/2 ms。任何"CPU 参与的决策"天然带 `[0.2ms + ≤1 迭代]` 的陈旧度——这是 §6 设计必须尊重的物理常数。

---

## 3. 执行模型：谁在什么时候跑 `prepare_next_batch`

- **两个常驻 kernel**：`worker_kernel<<<num_workers, 128|256>>>` + `scheduler_kernel<<<num_schedulers, 32>>>` 在两条 non-blocking stream（[:1813-1823](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1813-L1823)）。worker CTA 因 ~201KB 动态共享内存请求而一 SM 一个。
- **task graph 是预发射的**：build 时把所有层间事件改写为 `EVENT_EMPTY`、依赖挪进各 task 的 `dependent_event`（[runtime.cc:942-961](../../src/kernel/runtime.cc#L942-L961)）。运行时只有 3 个事件进 scheduler 队列：TERMINATION、LAUNCH_DEPENDENT_TASKS（广播全部 ~15.7k task）、END_OF_TASK_GRAPH。**层间同步全部是 worker 侧对事件计数器的自旋**（[:938-957](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L938-L957)），scheduler 零参与。
- **单线程调度**：scheduler CTA 的 32 线程里只有 lane 0 进调度体（[:1109](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1109), [:1117](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1117)），其余 31 个直接返回。`prepare_next_batch` 的唯一调用点在 `EVENT_END_OF_TASK_GRAPH` 分支（[:1235-1238](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1235-L1238)），该事件经 [get_rand_sched_id](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L705-L717) 单播给一个 scheduler。
- **它是全局屏障——但由图形状保证**：qwen3 图是链，唯一叶 argmax（`num_triggers=1`，[builder.py:626-631](../../python/mirage/mpk/models/qwen3/builder.py#L626-L631)）传递依赖全部 task；它触发时所有 worker 在取任务处自旋（[:862-874](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L862-L874)）。**换一个多叶图这条性质会悄悄失效**——§6 依赖它，所以要加构建期断言。

---

## 4. 内存序与 KV 可见性链

> **白话导读**：本节回答一个问题——prefix cache 的本质是"请求 B 直接用请求 A 写下的 KV"，那么"A 写完了"这件事 B 凭什么信？GPU 上这不是天然成立的：每个 SM 有自己不互通的 L1（B 可能读到自己缓存里的旧数据，且读 KV 的 `cp.async.ca` 恰恰会把数据留在本 SM 的 L1），编译器和硬件还会乱序。解决靠"快递柜协议"：**release** = 包裹全放好才挂牌，且谁看到牌子就必然看到牌子前的所有包裹；**acquire** = 先看牌再取货；承诺有作用半径（`cta` 本楼层 / `gpu` 整栋楼 / `sys` 连 CPU）。本节把 A 写完 → B 开读之间每一次挂牌/看牌从代码里找出来逐跳检查。**结论：链条是通的**（每跳半径 ≥ gpu），跨请求共享 KV 页机制上安全——这是 §6 敢做的根据；但有三个"但是"（依赖无文档的 release-sequence 规则、三个原子助手缺 `"memory"` 编译器屏障、"迭代结束"只是图形状碰巧给的），逐条见下文。

KV 由 attention 任务自己写（无独立 append 任务）：`paged_k_cache_dmem.at(dst_row, col) = k_smem.at(...)`（[multitoken_paged_attention_4_16.cuh:432-433](../../include/mirage/persistent_kernel/tasks/ampere/multitoken_paged_attention_4_16.cuh#L432-L433)）——全 lane 的 plain 2-byte store，无 fence（唯一的 `__threadfence` 在 [:1186](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1186) 被注释掉）。读历史 KV 用 `cp.async.ca`（[copy_sm80.cuh:51-55](../../include/mirage/persistent_kernel/tasks/common/copy_sm80.cuh#L51-L55)），会驻留读者 SM 的**非一致 L1** → 跨请求页共享**架构上必须**有 release/acquire。

链条（每跳已验证）：

```
A 的 KV plain stores（全 lane）
 → __syncthreads()                          :995   (CTA 屏障并入 lane 0)
 → lane0: atom.add.release.gpu(事件计数器)    :1012
 → [scheduler 链: CAS release :1072 → ld.acquire :1190 → prepare_next_batch
    → st.relaxed + atom.add.release.gpu(worker 队列) :1252/:1258]
 → B 的 worker lane0: ld.acquire :864 → __syncthreads() :878
 → B 的全 lane 读页                                    ✓ ≥ device scope
```

三个必须带走的注意点：
1. 事件计数器处依赖 **release-sequence / RMW 链**（多生产者 `add.release` + 单 acquire 读，教科书成立，源码零说明）。
2. [mpk_atoms.cuh](../../include/mirage/persistent_kernel/mpk_atoms.cuh) 的 `ld_acquire_gpu_u64`([:49-54](../../include/mirage/persistent_kernel/mpk_atoms.cuh#L49-L54))/`ld_relaxed_gpu_u64`/`st_relaxed_gpu_u64` **缺 `"memory"` clobber**（其余六个助手都有）——scheduler 侧使用点 [:1190](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1190) 是真实的编译器重排风险；新代码要么放在 `__syncthreads` 之后、要么先补 clobber。
3. KV 写是 generic-proxy store 才免 fence——将来若改 TMA/cp.async.bulk 写 KV，需要 `fence.proxy.async`。

**对设计最重要的推论**：完成检出发生在屏障点（§3），该请求全部 KV 写已因果先于此刻；导出页 id 经 `st.release.sys` 过 PCIe，再借出时经请求 ring 的 acquire 重新进入上述链——**"完成时导出 → CPU 中转 → 再导入"每跳都有 ≥device scope 的序**。这是 §6 可行的机制学基础。

---

## 5. Issue #666 复盘：两个方案为什么都卡住

- **Option 1（GPU 上做 hash map）**：匹配、refcount、驱逐全要在**单 scheduler lane** 上串行做指针追逐，直接落在每步关键路径；需要 GPU 侧 hashmap/链表与索引内存；换驱逐策略 = 改 kernel 重编译。issue 唯一评论已论证不现实。
- **Option 2（每步导出页 id，CPU 管缓存）**：#666 自己画出死结——CPU 匹配期间 GPU 又产生新页/新 token，匹配基于陈旧映射。根因：**试图对在飞状态做缓存**。且通道本身不存在：完成 ring 只有 `{rid,row,final_step}`，页池是 gpu_malloc 黑箱（§1.4）。#747 的 `kv_event_log` 已是该导出通道的 GPU 内存 debug 原型。
- **隐藏第三案（vLLM 式 CPU 全权分配）**：每迭代同步下发页表 = 杀死 megakernel 自治前提。不考虑。

而 §1.2 的机制事实给出决定性 hook：**attention 从不读 `step`，序列长度完全由页表推导**——给请求预置正确页表就是命中的充分条件。缺的只是三条通道 + 一个所有权规则。现有脚手架为什么"接了也不对"：`initial_step` 一路通到 kernel（[:610](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L610), [:620](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L620), [:634](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L634), [:640](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L640)），但 Step 4 分页从 `j=0` 全新分配（[:648-651](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L648-L651)）；另一半（导出）在 `MODE_ONLINE_TEST` 的 `final_paged_kv_*`（[runtime_header.h:373-380](../../include/mirage/persistent_kernel/runtime_header.h#L373-L380)）——而**没有任何构建路径能定义这个宏**（[persistent_kernel.py:240-253](../../python/mirage/mpk/persistent_kernel.py#L240-L253) 的 mode→flag 映射里没有它）。

---

## 6. 设计：OIPL（所有权反转的页生命周期）

### 6.1 核心规则

> **GPU 是唯一的页分配者**（快路径零改动，页池留在 device）；**CPU 是唯一的页释放者**（哈希索引、refcount、LRU 全在 Python）；**缓存域只覆盖已完成请求的页**。

已完成请求的页 GPU 永不再写、且（释放权移交后）永不自行回收 → CPU 视图对缓存内容**由构造**永不陈旧。#666 Option 2 的死结（匹配期间产生新页）自动消失：在飞状态根本不在缓存域里。这是 #754 row-lease 模式在页维度的补全：#754 lease 了 token row（GPU 完成后保留 row 直到 CPU ack），OIPL lease 页（GPU 完成后导出页直到 CPU 归还）。

v1 收缩（评审结论）：**只缓存 prompt token 覆盖的完整页**。生成 token 的缓存需要跨轮 token-id 精确追踪（detokenize→re-tokenize 不保证恒等），v1 不做。多轮收益仍在：turn k+1 的 prompt 里 turn k 的回答已是 prompt token，完成后即插入，turn k+2 起命中——命中滞后恰一轮（业界标准行为）。

### 6.2 新增 pinned 通道（全在 MODE_ONLINE_PINNED ifdef 内；按 KV group 参数化，v1 实现 group 0）

| 通道 | 字段 | 方向 / 纪律 |
|---|---|---|
| prefix 导入（请求 ring 扩展） | `pinned_req_num_prefix_pages[8]`、`pinned_req_prefix_pages[8][MAX_PAGES_PER_REQ]` | CPU→GPU。payload 全写完 → `store_i32_release(ready=1)`（#754 的 `_publish_request_locked` 纪律）。`MAX_PAGES_PER_REQ = ceil(MAX_SEQ/PAGE_SIZE)` |
| 页导出（完成 ring 扩展） | `pinned_comp_num_pages[8]`、`pinned_comp_pages[8][MAX_PAGES_PER_REQ]` | GPU→CPU。写在 `st.release.sys(comp_ready)` **之前**（复活 `final_paged_kv_*` 语义） |
| 页归还 ring（新） | `pinned_page_return_ring[R]`（R = 2 的幂 ≥ MAX_NUM_PAGES → **结构上不可能溢出**：在外页数 ≤ 总页数）、`pinned_page_return_tail` | CPU→GPU SPSC。CPU 写 id 后 `store_i32_release(tail)`；GPU 私有 head（gpu_malloc）+ `pinned_page_return_head_mirror`（CPU 观测占用/守恒断言） |

### 6.3 GPU 侧改动（全在 `prepare_next_batch` 单线程内，保持"task 皆纯读者"）

1. **Step 0（新）**：`ld_acquire_sys(return_tail)`，从私有 head drain 到 tail，push 进 `page_queue`。**每迭代限额**（如 ≤32 条），避免单线程在屏障点做长串 uncached PCIe 读。
2. **Step 1**：删除 [:528-535](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L528-L535) 的页释放循环；写完成项时先写 `num_pages` + 页 id 列表（源 = `paged_kv_indices_buffer[kv_indptr..]`——Step 1 时该 buffer 仍是上一迭代布局，与被删的释放循环同源，语义一致），再 release `comp_ready`；归还预留余额 `avail_uncommitted += reserved_remaining[row]`。
3. **Step 4（两处关键修订，均来自评审）**：
   - **(a) GPU 预留台账 + 准入门**（修 CPU 水位线的 TOCTOU，见 §10-1）：新增 device 标量 `avail_uncommitted`（init = MAX_NUM_PAGES）与 per-row `reserved_remaining[]`。读出 npp 后：`worst_need = ceil(MAX_SEQ/PAGE_SIZE) − npp; if (avail_uncommitted < worst_need) break;` 否则扣减并记 `reserved_remaining[row] = worst_need`。此后每为该 row 弹一页就 `reserved_remaining[row]−−`（free−− 与 reserved−− 抵消，avail 不变）。**安全不变量完全收拢在单线程内，零竞争**；#747 的 `MPK_REQUIRE_FREE_PAGE` 由此成为真断言；CPU 水位线降级为吞吐启发式。
   - **(b) 槽释放时序**：现行代码在 [:624-625](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L624-L625) 清 ready 后才跑页循环——prefix 页数组必须在清 ready **之前**消费。做法：把 `st_release_sys_i32(ready, 0)` 移到 admit 体末尾（单消费者，无早清必要）。分页两阶段：`j < npp` 从导入数组拷入，`j ≥ npp` 弹新页；`initial_step = npp*PAGE_SIZE`（CPU 保证页对齐且 ≤ prompt_len−1）；last_page_len 用修正式。
4. **Step 6**：写回 return-ring head；发布 head mirror。

### 6.4 CPU 侧：单所有者线程模型（消灭锁层级）

评审证明"新增 cache_lock + 三把既有锁 + #754 的锁"必然成环（§10-4）。修订：**全部缓存与台账操作收拢到一个所有者线程**（现成的 `_drain_loop`）：`submit()` 只把 `(rid, token_ids)` 入队；所有者线程串行做——完成处理（页导出入账 / 去重插入 / 归还）、**发布时匹配**（匹配 + refcount++ + 写 prefix 数组 + publish 是一个原子步骤；请求被真正发布前**不持有任何 refcount / 水位承诺**——同时消灭"`_waiting` 停队请求钉死页池"与"flush 发布陈旧页表"两类问题）、驱逐、水位启发式。代价：准入 +≤0.2ms（drain 周期），相对一次迭代是噪声。

缓存结构：块 = 一个物理页 = PAGE_SIZE 个 prompt token 的全层 KV。键 = 链式哈希 `h_j = H(h_{j-1}, ids[j·PS:(j+1)·PS])`；每块存自己的 PS 个 token id **与父块链接**；**命中时逐块比对每一个匹配块的存储 id 并校验父链**（见 v2.1-b）。反查表 page_id→块。插入只切 `[0, min(prompt_len, final_step))` 内的完整页（见 v2.1-d），键来自 submit 时留存的 prompt ids（**无需 D2H、无 re-tokenize**）。导出入账**无条件执行**（含 #754 abandoned 完成——页记账无条件，会话投递才有条件）。驱逐：LRU + refcount==0 + 叶先于父。reset()/kernel 重启 → 缓存与台账整体失效重建。

**v2.1 修订（2026-09-01，由协议仿真器 [tests/serving_python/oipl_protocol_sim.py](../../tests/serving_python/oipl_protocol_sim.py) 以最小化 trace 逼出，逐条有失败测试背书）：**

- **(0) 发布一致性（P1 翻转阶段确认的硬规则）**：CPU 发布 npp>0 时**必须同时写 `initial_step = npp·PAGE_SIZE`**——kernel 仍从独立的 `pinned_req_initial_step` 通道读取起始位置，两者不一致 = Step 4 跳过 prefill 却没有导入对应页 = attention 读垃圾（§1.3 记录过的既有 hook 陷阱）。P2 落地时在发布点统一派生，长期应让 kernel 侧改为自行派生并退役该字段。
- **(a) 发布时 pin 预算（修活性死锁）**：已发布未准入的请求持有 refcount 却排在 FIFO ring 里，可以把足量缓存页钉死到"ring 头部请求永远凑不齐 worst_need"——账本保证安全但不保证活性（`test_unbounded_publish_pins_deadlock_admission` 复现）。规则：管家维护 `Σ(未准入已发布请求的 npp，不含 ring 头) ≤ TOTAL − ceil(MAX_SEQ/PS)`，超出则对新发布**削减 npp**（可降到 0，宁可少命中不可钉死池子）。CPU 无需新通道即可观测准入：GPU 按 FIFO drain，最老已发布槽的 `ready` 1→0 即是准入信号。
- **(b) 命中校验必须含父链**：仅逐块比对本块 id **不**等价于全前缀比对——链式哈希在深度 j 命中的块可能来自 id 相同、但**前文不同**的另一条链（需要 j−1 层碰撞才会发生，但规则本身必须排除它，否则是静默错误命中而非 miss，S6 失效）。每块存父块引用，命中时沿链校验。仿真器用故意易碰撞的哈希演示了该场景（`scenario_wrong_prefix_hit`）。
- **(c) 水位线在紧池下不是纯启发式**：缓存驻留页 1:1 压低 `avail_uncommitted`，若管家不对"最老已发布请求久等未准入"做反应，准入会在无任何不变量被破坏的情况下无限停摆（GPU 没有"我饿了"的通道）。规则：最老已发布请求连续 N 个管家周期未准入 → 每周期额外驱逐一个可驱逐块。P3 验收因此是**双边的**：准入门触发频率 ≈ 0，**且**无发布请求等待超过 X 周期。
- **(d) 插入范围是 `[0, min(prompt_len, final_step))`**：`prompt_len == MAX_SEQ` 时完成可能发生在 prefill 完成前（final_step < prompt_len），按 `[0, prompt_len)` 会缓存带洞的页。v1 下不可达（npp·PS ≤ prompt_len−1 使此类块不可能被导入），但离"可达"只差一次约束改动，规则先立。
- **(e) 池容量与并发的显式关系**：账本按 worst_need 预留 → 有效准入并发 = ⌊pool / ceil(MAX_SEQ/PS)⌋（B1 的 max_tokens 落地前）。紧池（pool ≈ MBR·ceil(MAX_SEQ/PS)）下缓存**没有**可用余量——池容量公式不是建议而是前提。

---

## 7. 正确性论证

- **S1 无陈旧**：缓存只索引完成页；完成检出在屏障点（前提：链式单叶图——**向 builder 加构建期断言，多叶图须禁用 OIPL**），KV 写因果先于导出；导出→CPU→导入每跳有 sys/gpu scope 的序（§4）。多字 payload 由单旗的 release/acquire 整体覆盖（序语义与 payload 大小无关，无 torn read）。
- **S2 无写共享**：`initial_step = npp·PS` 页对齐 → 导入者首写落在页槽 npp（新页）；只缓存完整冻结页规避同迭代兄弟任务无序问题。同一缓存页出现在同批两请求页表里是纯读共享：Step 2 快照、Step 1 双导出、CPU 双 refcount−− 均自洽。
- **S3 守恒**：`avail_uncommitted = (page_queue_tail − page_queue_head) − Σ reserved_remaining[row]` 是单线程私有量的恒等式。请求页数上界：done 条件（[:503-506](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L503-L506)）使最大写入位置 ≤ MAX_SEQ−1 → 页数 ≤ ceil(MAX_SEQ/PS) = npp + worst_need → `reserved_remaining ≥ 0` 恒成立。
- **S4 无溢出**：完成 ring 占用 ≤ total_inflight（#754 row-lease：完成未 ack 的请求各占一 row）≤ cap——把 `cap ≥ total_inflight` 加进 [`_validate_kernel_compatibility`](../../python/mirage/mpk/persistent_kernel.py#L477-L527)（顺带补上 ring capacity 本就缺失的校验）；归还 ring 容量 ≥ 总页数。
- **S5 退化（P1 故障注入实测后的第三版）**：drain 线程死亡的实际行为是**即时 fail-closed**而非静默停摆——预存的 `_raise_drain_error` 闩锁使后续每个调用者（含 submit）立刻抛错，所有请求 ~0ms 内得到 500；comp-spin 来不及触发（submit 在触碰 ring 前已抛错）。台账**冻结而非损坏**：free 恰好少掉故障前已准入请求持有的页数，pending==0、head==returned、零异常。剩余暴露面 = 可用性（fail-stop），非正确性；看门狗/自愈是后续加固项而非安全前提。请求线程侧的一次性 drain 异常只损失该请求本身（单个 500），守恒不受影响。
- **S6 键正确性**：逐匹配块全量 id 比对**加父链校验**（v2.1-b；只比本块 id 不排除跨链同 id 块）→ 错误命中不可能；最坏情形 = 零收益。模板不稳定只降命中率不伤正确性。

**与 #666 两案对比**：每步关键路径开销 ≈0（vs Option 1 的单线程指针追逐）；一致性构造性消解（vs Option 2 未解决）；策略在 Python 迭代（vs Option 1 重编译）；代价是页回收 +1 次 CPU 往返（≤0.2ms + ≤1 迭代，由 GPU 台账兜底不构成正确性风险）与 v1 并发同前缀突发 miss（v2 用 `pinned_step ≥ (j+1)·PS` 谓词做在飞发布——该谓词已与 KV 写 release 排序）。

**收益量化**：MPK 的 prefill 速率被 `mbt=8` 钉死在 8 token/迭代，且长 prompt 的贪心预算会饿死同批其他请求（[:565-570](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L565-L570)）。命中 H token = 省 H/8 次**全图迭代**，同时把预算还给其他请求的 decode——**在 MPK 上 prefix cache 同时是 TTFT 优化和公平性优化**，边际价值高于 vLLM/SGLang。

---

## 8. 兼容性

| 对象 | 关系 |
|---|---|
| **PR #754**（row lease） | **硬前提且同构**。叠加点：导出写先于 comp_ready release（同其纪律）；prefix 发布走 `_publish_request_locked`；新字段全用其 `load_i32_acquire`/`store_i32_release` C 助手（顺带修 ARM 主机的 plain-store 洞）；abandoned 完成的页记账无条件执行。文本冲突：都改 Step 1——先合 #754 再叠 OIPL |
| **PR #747**（unified KV pool） | **前提**（运行时 page size、2D smem 快照、`MPK_REQUIRE_FREE_PAGE`）**且需参数化**：三条通道与台账按 KV group 加一维，v1 实现同构 KV（qwen3 单 group）。其 `kv_event_log` 可视为本协议的 debug 前身。#747 与 #754 改同段 init 代码，需先后手工合并 |
| A10 分页几何修复 | **硬前提**：① online 版 last_page_len 裸 `%` 补零修正；② Ampere 向量化页表载入补 `first_page_pos`（[multitoken_paged_attention_4_16.cuh:110-113](../../include/mirage/persistent_kernel/tasks/ampere/multitoken_paged_attention_4_16.cuh#L110-L113) 漏加，标量尾循环 [:122](../../include/mirage/persistent_kernel/tasks/ampere/multitoken_paged_attention_4_16.cuh#L122) 是对的）——**且补了之后 `base + first_page_pos*4` 不再保证 16B 对齐，必须做标量头对齐**；③ `static_assert(PAGE_SIZE % KV_TILE_SIZE == 0)`。PAGE_SIZE 4096→64 时 `MAX_NUM_PAGES`/indices buffer/smem 上界同步放大，池容量公式 `pool ≥ MBR·ceil(MAX_SEQ/PS) + 缓存余量` 由 #747 kv_planner 承载校验 |
| 多轮对话（A3） | **硬前提**：现路径唯一共享前缀是 30-token 硬编码 system prompt（[tokenizer_manager.py:15-28](../../python/mirage/engine/tokenizer_manager.py#L15-L28)）< 1 页——不修多轮，命中率结构性为 0 |
| Chunked prefill (#689) | 正交兼容：`remaining = prompt_len − initial_step`，命中只是把 chunk 起点前移 |
| Offline / spec decode / TP | 全部改动在 MODE_ONLINE_PINNED ifdef 内，offline 零影响；draft KV 独立池（#684）不进缓存域；TP>1 需 admit 决策跨 rank 确定性一致，v1 单卡 |
| Codegen 三重簿记 | `assert(meta_tensors.size()==23)`（[:1483](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L1483)）+ 两份手工列表（[persistent_kernel.py:3003-3028](../../python/mirage/mpk/persistent_kernel.py#L3003-L3028) 与 [:3126-3145](../../python/mirage/mpk/persistent_kernel.py#L3126-L3145)）同步改——漏掉第二份会静默破坏 `load_mpk` kernel 复用 |

---

## 9. 分阶段落地（每步独立可测、可单独合并）

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | 合 #754 → 合 #747（同段冲突序贯处理）→ A10 三修 + PS=64 对齐修 → 多轮对话 → [mpk_atoms.cuh](../../include/mirage/persistent_kernel/mpk_atoms.cuh) 补 `"memory"` clobber | 各项独立有价值 |
| **P1** | 所有权反转（无缓存）：三条通道 + Step 0/1/4/6 + CPU 直通归还 + **GPU 预留台账/准入门**（它是所有权反转本身的安全条件，随 P1 进） | #756 harness 全绿；守恒三方对账（mirror_head/导出/归还）；吞吐差 < 噪声。⚠ P1 非严格行为等价：页回收多一轮 CPU 往返，极限吞吐下准入可能后移一迭代——写进预期 |
| **P2** | 缓存 + 命中（prompt-only）：所有者线程、发布时匹配、去重插入 | hit rate、TTFT p50/p99、迭代时间无回归；**模板前缀稳定性单测在此定论** |
| **P3** | 驱逐 + 水位启发式 + 压力（池占满/驱逐风暴/所有者线程 kill 注入）+ v2.1-a/c 两条活性规则 | **双边验收**：GPU 准入门触发频率 ≈ 0（否则池配小了），且无已发布请求等待超过 X 个管家周期（否则活性规则失效） |
| **P4** | v2 在飞发布：`pinned_row_page_table` 增量发布 + `step ≥ (j+1)·PS` 谓词，覆盖并发同前缀突发 | — |
| **P5** | 远期：导入通道复用为 P/D 分离 decode 侧恢复原语之一（还需 prompt token 搬运与 row 状态协议，**非充分**） | — |

---

## 10. 评审记录：被推翻的初稿断言

设计初稿经 6 视角 60-agent 对抗评审，以下断言被推翻并已在上文修正（诚实清单）：

1. **"CPU 水位线保证 GPU 永不撞空池" —— 错。** 归还 ring 与请求 ring 是两条无序通道，GPU 在同一次 `prepare_next_batch` 里先读前者（Step 0）后读后者（Step 4）；CPU 在两次读之间"驱逐一页 → 记账为可用 → 发布新请求"是 drain 线程一个函数内微秒级连续动作，且**缓存稳态下每次准入恰好靠刚归还的页融资——竞争是常态路径**。#747 下后果是 `__trap()` 全 kernel 死亡。→ 修正：GPU 单线程预留台账 + 准入门（§6.3a）。
2. **Step 4 在槽释放后读 prefix 数组 —— 错。** [:624-625](../../include/mirage/persistent_kernel/persistent_kernel.cuh#L624-L625) 清 ready 后 CPU 即可复用槽。→ ready-clear 移到 admit 体末尾。
3. **submit 时匹配 + refcount++，可停 `_waiting` —— 错。** 停队请求钉死页池可致永久准入死锁；`flush_waiting` 不发布 prefix 数组 → **静默错误输出**；水位不在 flush 点重查。→ 发布时匹配 + 单所有者线程 + 停队零持有（§6.4）。
4. **cache_lock 置顶的锁序 —— 错。** 与 #754 的 `_lock` 组成 ABBA。→ 锁消灭于单所有者线程。
5. **"CPU 死亡不影响运行中请求" —— 夸大。** #754 comp 自旋下 drain 死透可 wedge GPU。→ S5 改写为诚实退化。
6. **v1 缓存生成 token 的块 —— 不可靠。** detokenize→re-tokenize 不保证 id 恒等，且完成路径需额外 D2H。→ v1 prompt-only。
7. **abandoned 完成"照常处理"（未写明）—— 漏洞。** #754 的 `_abandoned` 分支绕过页记账 → 永久泄漏。→ 页记账无条件、会话投递有条件。

机制部分（§1–§4）的 96 条断言中 37 条在验证中被修正/精化、0 条被整体推翻；文中已全部采用修正后版本。

## 11. 遗留不确定项

1. ~~Qwen3 模板前缀稳定性~~ **已实测解决（2026-09-01，真实 tokenizer 实验）**：`enable_thinking=True`（默认）下跨轮 token 前缀**完全稳定**——历史 assistant 内容含 `<think>` 也稳定（模板每次渲染都同样剥除）。仅硬开关 `enable_thinking=False` 破坏前缀，且只差 generation-prompt 尾部强插的 4 个 token（`<think>\n\n</think>`）。P2 规则：**插入时把 gen-prompt 尾巴排除在可缓存区外**（尾长启动时渲染一次即得）。`/no_think` 软开关不影响结构。残余：实验用 transformers 5.16.1，仓库钉 4.57.1——P2 带 in-repo 单测复确认。
2. **PS=64 下 Step 2 快照与 Step 0 drain 的单线程开销**——预估 µs 级（批内 ≤256 页），需 `MPK_ENABLE_PROFILING` 实测；超标则把快照挪 global（可复用死掉的 `paged_kv_indices_snapshot`）。
3. **"链式单叶图"前提的持久性**——spec-decode/MoE 图可能多叶；已列构建期断言，多叶时 OIPL 须禁用或改逐任务计数。
4. **#754 尚未合并且 CI 未绿**——本文对其依赖基于 patch 文本而非运行验证。
