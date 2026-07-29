# Qwen3-Omni 开源架构与当前项目差异报告

日期：2026-07-29
本地基线：`391d0186cb225f41f8e47aa600c36da0bc6b21e4`
结论状态：研究与修复方案，不宣称当前项目兼容官方 checkpoint

## 1. 结论摘要

官方 Qwen3-Omni 已开放 Instruct、Thinking 和 Captioner 权重，也已在
Transformers 中提供完整的推理结构。当前项目则是一个具备自定义 Thinker、
MoE/TP 实验和 Stage-2 训练基础设施的研究原型。两者的差异不是少量配置项，
而是覆盖媒体编码、序列融合、位置编码、Thinker、Talker、codec、缓存和训练
阶段的系统性差异。

本分支已经完成一轮 P0 正确性修复，包括：

- 原型和官方参考依赖环境隔离；
- Stage-2 无标签推理；
- Stage-2 配置归一化、校验和 step cadence；
- 多模态 token ID 由 tokenizer 解析并原子校验；
- 训练和推理共享严格媒体加载语义；
- 标准与 TP MoE top-k 路由对齐；
- Transformers 4/5 checkpoint helper 导入兼容。

这些修复使当前原型成为稳定的后续改造基线，但没有改变以下核心事实：

- 图像和音频仍各自压缩为一个 token；
- 没有视频输入、音视频时间交错或 TM-RoPE；
- 自定义 Thinker 结构和官方 30B-A3B 不一致；
- Talker、MTP 和 Code2Wav 仍未实现；
- 没有生成缓存或真正的流式推理；
- 官方训练所需的数据、完整 recipe 和 codec encoder 也未全部公开。

因此建议先建立官方 Qwen3-Omni golden oracle，再分阶段替换当前占位模块。
在 state-dict、处理结果和数值差分均通过前，只能称为
“Qwen3-Omni-style prototype”，不能称为官方结构复现。

## 2. 资料范围与证据等级

报告按以下等级描述事实：

- **论文明确**：Qwen3-Omni 技术报告直接说明。
- **公开配置/源码**：官方 checkpoint 配置、tokenizer 或 Transformers 实现。
- **本地代码事实**：当前基线代码和测试可以直接验证。
- **合理推断**：由公开结构推导，但不是论文原文承诺。
- **未公开**：没有足够资料支持精确复现。

主要一手资料：

- [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765v1)
- [QwenLM/Qwen3-Omni，固定提交](https://github.com/QwenLM/Qwen3-Omni/tree/e4235853125589c789f06a2dd83e9f4126df5e9d)
- [Qwen3-Omni-30B-A3B-Instruct，固定 revision](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/tree/26291f793822fb6be9555850f06dfe95f2d7e695)
- [Transformers v5.2.0 Qwen3-Omni 实现](https://github.com/huggingface/transformers/tree/v5.2.0/src/transformers/models/qwen3_omni_moe)
- [官方模型 collection](https://huggingface.co/collections/Qwen/qwen3-omni-68d100a86cd0906843ceccbe)

## 3. 官方开放模型边界

| 模型 | 开放内容 | 输入到输出 | 主要组件 |
|---|---|---|---|
| `Qwen3-Omni-30B-A3B-Instruct` | 权重、配置、tokenizer | 文本/图像/音频/视频到文本和语音 | Thinker、Talker、MTP、Code2Wav |
| `Qwen3-Omni-30B-A3B-Thinking` | 权重、配置、tokenizer | 任意输入模态到推理文本 | Thinker |
| `Qwen3-Omni-30B-A3B-Captioner` | 权重、配置、tokenizer | 音频到详细文本描述 | Thinker，配置与 Thinking 相同 |
| `Qwen3-Omni-30B-A3B-Base` | 仅论文讨论 | 无公开 checkpoint | 未公开 |
| `Qwen3-Omni-Flash` | API | 商业服务 | 精确结构和权重未公开 |

官方 GitHub 仓库主要提供模型说明、示例、Docker 和 demo。可执行模型实现实际
位于 Transformers 和推理框架中；完整预训练代码没有随权重开放。

## 4. 官方 Qwen3-Omni 架构

### 4.1 总体数据流

```text
文本 ── byte-level BPE ────────────────────────────────┐
音频 ── 16 kHz/128 Mel ── AuT ── projector ───────────┤
图像/视频 ── Qwen3-VL ViT ── merger/DeepStack ───────┤
                                                       ▼
                              TM-RoPE + Thinker 30B-A3B
                                                       │
                         ┌─────────────────────────────┴───────────┐
                         ▼                                         ▼
                       文本                              Instruct 的 Talker
                                                                 │
                                                          主 codebook
                                                                 │
                                                        5-layer MTP
                                                                 │
                                                        其余 15 路 code
                                                                 │
                                                           Code2Wav
                                                                 ▼
                                                            24 kHz 波形
```

融合采用媒体 embedding 替换媒体 placeholder，而不是 cross-attention，也不是
把所有媒体固定放在文本前。视频和音频根据真实时间戳交错。

### 4.2 Tokenizer

**论文明确：**

- Qwen byte-level BPE；
- 151,643 个 regular tokens。

**公开配置/源码：**

- Thinker embedding vocabulary 为 152,064；
- `<|image_pad|>`、`<|video_pad|>`、`<|audio_pad|>` 等特殊 token 位于
  regular vocabulary 之后；
- tokenizer metadata、模型配置和实际训练长度存在不同口径：tokenizer 可写
  131,072，Thinker 配置为 65,536，论文长上下文阶段训练到 32,768。

当前项目已经改为从 tokenizer 解析多模态 token ID，并禁止配置中的裸数字成为
第二权威源。这一方向正确，但尚未实现官方 processor 所要求的 placeholder
展开、grid 和 position ID 构造。

### 4.3 AuT 音频编码器

**论文明确：**

- 输入重采样至 16 kHz；
- 128-channel Mel，25 ms window、10 ms hop；
- 三层 stride-2 Conv2D，总下采样 8 倍；
- 输出约 12.5 Hz，即每个 audio token 对应约 80 ms；
- AuT 从零训练，使用约 2,000 万小时监督音频。

**公开配置/源码：**

- 32 层 Transformer；
- `d_model=1280`、20 heads、FFN 5120；
- projector 为 `1280 → 1280 → GELU → 2048`；
- 支持动态 attention window 和长音频推理路径。

### 4.4 视觉与视频编码器

**论文明确：**

- 采用 Qwen3-VL vision encoder，初始化自 SigLIP2-So400m；
- 同一编码器处理图像和视频；
- 视频使用动态 FPS，并按实际 timestamp 与音频对齐。

**公开配置/源码：**

- 27 个 ViT block；
- hidden 1152、16 heads、FFN 4304；
- Conv3D patch 为 `2×16×16`；
- spatial merge size 为 2；
- 输出通过 merger 映射到 Thinker hidden 2048；
- ViT 第 8、16、24 层的 DeepStack 特征注入 Thinker 最前面的三层。

### 4.5 TM-RoPE 与序列融合

TM-RoPE 使用 temporal、height、width 三轴。官方 checkpoint 的 rotary
section 分配为 `24/20/20`，`rope_theta=1,000,000`。

- 文本和音频的三轴 position ID 相同；
- 图像具有固定 temporal ID 和变化的 height/width ID；
- 视频 temporal ID 按真实时间增加；
- 音频与视频共享约 80 ms 的时间网格；
- 后一模态的位置从前一模态最大位置加一开始；
- `use_audio_in_video=true` 时，音频和视频 token 按 timestamp 交错。

### 4.6 Thinker 30B-A3B

| 项目 | 官方配置 |
|---|---:|
| decoder layers | 48 |
| hidden size | 2048 |
| Q / KV heads | 32 / 4 |
| head dim | 128 |
| routed experts | 128 |
| experts per token | 8 |
| expert FFN | 768 |
| shared expert | 无 |
| MoE frequency | 每层 |
| top-k renormalization | 是 |
| router aux coefficient | 0.001 |
| max positions | 65,536 |
| RMSNorm epsilon | `1e-6` |

Thinker 使用 causal GQA、Q/K RMSNorm 和 SwiGLU 风格专家。官方结构不包含当前
项目的简化 DeltaNet 或自定义 head-wise output gate。

### 4.7 Talker、MTP 与 Code2Wav

仅 Instruct checkpoint 包含完整语音输出链路。

Talker：

- 20 层、hidden 1024；
- 16 Q heads、2 KV heads；
- 128 experts、top-6；
- routed expert FFN 384、shared expert FFN 768；
- 接收 Thinker token embedding 和中间 hidden representation。

MTP：

- 5 层 dense causal Transformer；
- 预测主 codebook 之外的 15 个 residual codebooks；
- residual codebook vocabulary 为 2048。

Code2Wav：

- 使用 16 路 RVQ codebooks；
- 每秒 12.5 codec frames；
- 公开实现包含 sliding-attention Transformer frontend 和 causal
  convolution decoder；
- 每个 frame 产生 1,920 个 waveform samples，得到 24 kHz 波形。

完整 audio-to-RVQ codec encoder、codec 训练数据和 loss 没有公开，因此
“重新训练同一个 codec”不可精确复现；开放 checkpoint 的推理和微调仍可作为
golden oracle。

## 5. 官方训练流程与不可复现边界

### 5.1 预训练

论文公开三阶段：

1. **Encoder Alignment**
   - LLM 初始化自 Qwen3 并冻结；
   - vision 和 audio 分开训练；
   - 先训练 adapter，再训练 encoder。
2. **General**
   - 解冻全部参数；
   - 使用约 2T 的多模态 token；
   - 同时包含文本、图像、视频、音频和音视频数据。
3. **Long Context**
   - 最大长度从 8K 扩展至 32K；
   - 增加长音频和长视频比例。

### 5.2 后训练

Thinker 使用 SFT、strong-to-weak distillation 和 GSPO。Talker 使用
continued pretraining、DPO 和 speaker fine-tuning。

### 5.3 未公开

以下内容不足以支持官方等价复训：

- 完整数据集、许可、清洗和去重；
- 优化器、学习率、batch、训练步数和算力；
- 各阶段精确 sampling ratio 和复合 loss 权重；
- AuT decoder 和 codec 的完整训练 recipe；
- 论文延迟实验的精确硬件与部署拓扑。

因此本项目可以复现开放结构、加载开放权重并重建小规模训练阶段，但不能宣称
复现了官方预训练结果。

## 6. 当前项目架构

当前项目的主要实现位于：

- [模型配置](../../src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py)
- [自定义 Thinker](../../src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py)
- [多模态 wrapper](../../src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py)
- [Stage-2 训练入口](../../src/qwen3_omni_pretrain/training/trainer_thinker.py)
- [Stage-2 推理入口](../../src/qwen3_omni_pretrain/cli_infer_thinker.py)

### 6.1 已经可靠的基础能力

- target-only causal LM supervision；
- padding 和多模态 attention mask；
- tokenizer 驱动的特殊 token ID；
- 严格/可审计的媒体加载；
- 标准和 TP MoE 的 top-k 路由一致性；
- 非有限 loss/logits/gradient 的分布式 fail-fast；
- Stage-2 配置归一化、checkpoint resume 和 step cadence；
- 独立的原型/官方参考依赖环境。

这些能力应保留并迁移到后续架构，而不是在重写时丢弃。

### 6.2 仍是占位的架构

[SimpleVisionEncoder](../../src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py#L35)
将 224×224 整图 flatten 后用一个 Linear 映射成单个 token。

[SimpleAudioEncoder](../../src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py#L54)
将最多 32,000 个 raw waveform samples 直接映射成单个 token。

多模态 wrapper 固定构造：

```text
[vision summary token, audio summary token, text tokens]
```

它没有：

- 可变长 patch/audio sequence；
- placeholder replacement；
- 视频或多段媒体；
- 音视频 timestamp interleave；
- T/H/W position IDs；
- DeepStack；
- chunked encoder cache。

### 6.3 Thinker 不对应官方 30B-A3B

[本地 30B YAML](../../configs/model/qwen3_omni_30b_moe.yaml)使用：

- hidden 6656；
- 64 Q heads、64 KV heads；
- max positions 4096；
- `rope_theta=10000` 和 partial linear RoPE；
- 简化 DeltaNet；
- 自定义 attention output gate；
- 非官方的 MoE 层和专家配置。

它与官方 hidden 2048、32Q/4KV、128 experts/top-8 的 30B-A3B 不兼容。
文件名中的“30b”也没有由测试证明等于真实总参数。

### 6.4 Talker 与流式推理缺失

Talker 和 Code2Wav 目前只有 config dataclass；训练器和专用 loss 仍为空。
文本 greedy decode 每步重新计算完整增长序列，也会重复执行媒体 encoder。
没有 `past_key_values`、GDN state、Talker state 或 codec streaming cache。

### 6.5 分布式缺口

Stage-1 具备多种分布式路径，但 Stage-2 当前明确拒绝 DDP、Accelerate、
DeepSpeed 和 TP 入口。这是安全的 fail-fast 行为，但尚未满足大规模多模态训练
要求。

## 7. 差异矩阵

| 子系统 | 官方 Qwen3-Omni | 当前项目 | 影响 |
|---|---|---|---|
| 模型身份 | 官方 `qwen3_omni_moe` schema | 自定义 schema 使用相同 model type | AutoConfig 和 checkpoint 误识别风险 |
| Tokenizer | 152,064 embedding vocab | 已对齐 vocab，并由 tokenizer 解析媒体 ID | ID 基础已修复，processor 仍缺 |
| Vision | 27L ViT、patch sequence、DeepStack | 整图单 token | 丢失空间结构与视频能力 |
| Audio | AuT、12.5 Hz sequence | 两秒 waveform 单 token | 丢失时间结构和长音频 |
| Fusion | placeholder replacement、AV interleave | 固定两 token 前缀 | 不支持多段媒体或同步 |
| Position | TM-RoPE T/H/W | 1D partial RoPE | 无时间和空间建模 |
| Thinker | 48L、2048、128/top-8 MoE | 自定义尺寸、DeltaNet 和 gate | state dict 与数值不兼容 |
| Talker | 20L MoE | 仅配置 | 无语音生成 |
| MTP/codec | 16 code groups + Code2Wav | 仅配置 | 无 codec token 或 waveform |
| Cache | chunk prefill、生成状态 | 每步全量重算 | 无实时性，复杂度高 |
| 训练阶段 | alignment/general/long-context | text Stage-1 + 简单 Stage-2 | freeze 和数据语义不一致 |
| 后训练 | distillation/GSPO/DPO | 无 | 无对话、推理和语音对齐 |
| 评测 | 分模态与流式评测 | evaluation 包为空 | 无质量和延迟基线 |

## 8. 修复与完善方案

### A0：模型身份与官方 oracle

1. 为本地原型使用独立 `model_type`，避免与官方 schema 冲突。
2. 固定官方 checkpoint、Transformers 和 tokenizer revision。
3. 建立只读 golden oracle：
   - processor 输出；
   - token/grid/mask/position IDs；
   - Thinker logits；
   - Talker code shape；
   - Code2Wav output shape。
4. 把当前 YAML 标记为实验配置，不再用文件名暗示官方参数规模。

退出条件：

- 官方 config 由官方类无警告加载；
- 本地 config 不会被官方 AutoConfig 误识别；
- 参考测试不下载权重也能做 config/schema smoke；
- 权重测试通过显式标记按需运行。

### A1：序列保真的媒体输入

优先包装或复用官方 processor/encoder 接口，不继续扩展 one-token adapter。

1. 引入 `MediaSequence` 契约，包含 embeddings、mask、grid 和 timestamp。
2. 图像/视频保留 patch sequence。
3. 音频使用 Mel + Conv2D + Transformer sequence。
4. 融合改成 placeholder replacement。
5. 缺失媒体不生成伪 summary token。

退出条件：

- 1 秒音频产生约 12–13 个有效 audio tokens；
- 图像和视频输出可变长 patch sequence；
- media shuffle/ablation 会显著改变合成任务结果；
- processor 输出与官方 oracle 的 shape、mask 和位置一致。

### A2：TM-RoPE、视频与音视频交错

1. 建立独立 `PositionBuilder`。
2. 实现 T/H/W position IDs。
3. 动态 FPS 和真实 timestamp 映射到统一时间网格。
4. 音视频 token 按时间交错。
5. 多模态片段之间位置连续且无冲突。

退出条件：

- 同一事件的音频和视频 token 映射到相同 temporal bin；
- 不同图像 patch 具有正确 height/width ID；
- 跨模态位置单调、连续；
- chunked 与 offline 序列构造一致。

### A3：官方结构 Thinker

建立与实验 Thinker 隔离的官方 profile：

- 官方层数、hidden、GQA 和 head dim；
- Q/K RMSNorm；
- routed-only SwiGLU experts；
- 128 experts/top-8；
- 官方 aux loss 和 top-k renormalization；
- state-dict key 命名与官方实现对齐。

退出条件：

- 小输入 state key/shape 完整匹配；
- 官方 checkpoint 无意外 missing/unexpected keys；
- FP32 logits 与官方 oracle 在约定误差内；
- cached 和 uncached decode 输出一致。

### A4：Talker、MTP、Code2Wav 与流式状态

先以官方开放权重链路为 oracle：

1. Talker 接收正确的 Thinker embedding/hidden。
2. 主 codebook 自回归，MTP 补齐 residual codebooks。
3. Code2Wav 采用因果分块解码。
4. Thinker、Talker 和 codec 分别持有会话状态。
5. 媒体 encoder 只在 prefill 阶段运行一次。

退出条件：

- codebook mask 和 CE 目标正确；
- chunked/whole waveform 边界一致；
- 无 future leakage；
- 首帧、跨 chunk 连续性和中断恢复测试通过。

### A5：论文阶段式训练与分布式

1. Encoder Alignment：冻结 LLM，先 projector 后 encoder。
2. General：全参多模态训练。
3. Long Context：分阶段扩长并提高长媒体比例。
4. 为 Stage-2 增加经过测试的 DDP/Accelerate/DeepSpeed 路径。
5. 后训练模块单独实现 SFT、distillation、RL 和 Talker preference training。

退出条件：

- 每阶段 freeze/unfreeze 有自动断言；
- tiny dataset 可过拟合；
- 固定 seed resume 重现后续 loss；
- 分布式采样无重复/遗漏；
- 新模态加入后，文本和旧模态回归不超过预设阈值。

## 9. 最终判断

当前项目已经完成“稳定研究原型”的 P0 基础工作，但距离官方 Qwen3-Omni 的
架构忠实复现仍有明确且较大的差距。最稳妥的路线是：

1. 保留 legacy prototype 作为回归基线；
2. 建立官方 golden oracle；
3. 先解决媒体序列、融合和位置；
4. 再解决官方 Thinker 和生成缓存；
5. 最后接入 Talker/codec 和阶段式训练。

这条路线能够利用开放 checkpoint 做结构与数值验证，也能明确隔离官方没有开放
的训练细节，避免把自主设计误标为官方复现。
