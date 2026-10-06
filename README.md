# IMDB：BERT、SCL、R-Drop 与 LoRA

之前我做过 IMDB 的传统文本分类和预训练模型微调。这次沿用同一任务，把重点放在老师提出的SCL、R-Drop，以及 Unsloth 包装的 LoRA 上。

我先用 BERT 建立交叉熵基线，再分别加入 SCL 和 R-Drop，最后比较普通 PEFT LoRA 与 Unsloth LoRA。五组正式实验都使用 `bert-base-uncased`，在同一份训练/验证划分上完成训练，并生成 Kaggle 提交文件。

## 本次结果

| 实验 | 验证 Accuracy | 验证 ROC-AUC | Kaggle ROC-AUC | 最佳 epoch | 可训练参数 | 峰值显存 GiB |
|---|---:|---:|---:|---:|---:|---:|
| BERT + CE | 91.68% | 0.97239 | 0.97434 | 3 | 109,483,778 | 3.050 |
| BERT + CE + SCL | 91.88% | 0.97339 | 0.97566 | 2 | 109,483,778 | 3.149 |
| BERT + R-Drop | 92.42% | 0.97465 | 0.97637 | 3 | 109,483,778 | 4.788 |
| BERT + PEFT LoRA | 88.38% | 0.95376 | 0.95699 | 3 | 591,362 | 1.779 |
| BERT + Unsloth LoRA | 88.64% | 0.95461 | 0.95724 | 3 | 591,362 | 1.777 |

验证指标取验证 Accuracy 最好的 checkpoint。Kaggle 分数是提交后的 ROC-AUC，不是分类准确率。完整数值在 [comparison.csv](results/comparison.csv)，提交成绩在 [kaggle_scores.csv](results/kaggle_scores.csv)。成绩来自提交页面的记录，原始截图保存在本地。

这一次 R-Drop 的验证 Accuracy 比 CE 高 0.74 个百分点，SCL 高 0.20 个百分点。LoRA 的可训练参数和显存明显减少，但当前设置下准确率也下降了。Unsloth 与普通 LoRA 的耗时相近，本次没有观察到明确的加速效果。

这些是单随机种子的结果，不能据此判断统计显著性。完整微调与 LoRA 分别比较。普通 PEFT 和 Unsloth 所用的依赖版本不同，二者只能作为本次环境下的工程比较。

## 和老师要求的对应

| 要求 | 这次完成的内容 | 位置 |
|---|---|---|
| 理解 SCL 的原理和实现 | 明确 anchor、正负样本、温度、输入形状和 CE+SCL 总损失 | [方法笔记](docs/方法笔记.md)、[losses.py](src/losses.py) |
| 通过继承修改 Hugging Face 训练流程 | 继承 Trainer，重写 `compute_loss()`，保留分类 logits 与统一验证方式 | [research_trainer.py](src/research_trainer.py) |
| 尝试 R-Drop | 同一输入两次 Dropout forward，平均 CE 加双向 KL | [bert_rdrop](results/bert_rdrop/summary.json) |
| 尝试 Unsloth 包装的 LoRA | 完成三轮正式训练，与普通 PEFT LoRA 比较 | [bert_unsloth_lora](results/bert_unsloth_lora/summary.json) |
| 自定义损失与 Unsloth/LoRA 衔接 | Unsloth + LoRA + SCL 完成 30 步短步测试，SCL 项有实际数值 | [smoke_tests.json](results/smoke_tests.json) |

Unsloth + LoRA + SCL 目前只验证了流程，没有完成全量三轮实验。我把这一项单独记录，没有放进五组正式结果里。论文方法的理解整理在笔记中，不把这次配置称为论文的完整复现。

## 实验设置

- 官方 `word2vec-nlp-tutorial` 数据：25,000 条带标签评论，分层划分为 20,000 条训练、5,000 条验证，随机种子 42。
- 基座固定到同一个 Hugging Face commit，最大长度 256，batch 16，3 epochs，学习率 `2e-5`。
- 单张 RTX 4090 D 24GB，FP32 参数、BF16 混合精度。每个训练 batch 包含 8 条正面、8 条负面评论，梯度累积为 1。
- SCL：`alpha=0.2`、`temperature=0.07`，使用最后一层 `[CLS]` 表示，单视图同标签正样本。
- R-Drop：双向 KL 取平均，`beta=1.0`。LoRA：`r=16`、`alpha=32`，作用于 `query/value`，同时训练分类头。
- Unsloth 编译关闭，没有使用 4-bit 量化。验证损失统一为一次确定性 forward 的 CE。

## 文件与复现

```text
configs/       五组正式配置、短步测试配置、固定划分
src/           训练、自定义损失、模型加载和分类推理
scripts/       数据准备、按配置运行、结果绘图、提交检查
tests/         损失、采样、checkpoint、CSV 和离线流程测试
results/       每组指标、逐轮记录、验证预测、环境和图
submission/    五份可提交的 CSV
docs/          方法笔记、实验分析、运行说明和问题记录
requirements/  两套训练环境的依赖
```

从数据准备到训练、模型重载和生成提交文件的命令见 [运行说明](docs/运行说明.md)。结果解释见 [实验记录](docs/实验记录.md)，实现上的注意事项见 [问题记录](docs/问题记录.md)。

这份仓库保留本次实验需要的代码和小文件，不包含原始评论数据、完整模型权重、云端缓存和旧任务材料。已有训练权重保存在本地备份中。仓库里的结果来自已完成的运行记录，本次整理没有重新训练模型，也没有改变提交文件的概率值。公开记录只统一了文件路径和实验名称。

## 参考资料

- Khosla et al. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)。
- Liang et al. [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)。
- Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft)、[Unsloth](https://github.com/unslothai/unsloth)。
- 数据与在线评测：[Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/competitions/word2vec-nlp-tutorial)。

实现思路来自老师提供的示例与上述论文、框架。我在此基础上统一了划分和训练流程，增加了分项损失、结果记录、严格加载与提交检查。
