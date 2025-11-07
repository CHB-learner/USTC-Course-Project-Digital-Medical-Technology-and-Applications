IMCS-V2-MRG：中文医疗报告生成（本地 vLLM，Qwen/Llama）

本项目在 IMCS-V2-MRG 数据集上评测两种本地部署的指令模型（Qwen2.5-7B-Instruct 与 Llama 指令版）的结构化病历生成能力。
评测采用字级 ROUGE-1 / ROUGE-2 / ROUGE-L，并取三者算术平均为总分。

1. 文件结构与职责
medbench_run/
├─ scripts/
│  ├─ start_qwen.sh        # 启动 Qwen vLLM OpenAI 兼容服务（示例：127.0.0.1:8001）
│  └─ start_llama.sh       # 启动 Llama vLLM OpenAI 兼容服务（示例：127.0.0.1:8002）
│
├─ run_infer_imcs_mrg.py   # 在 IMCS-V2-MRG 上做推理：对话 → 6段病历文本（主诉/现病史/...）
│
├─ convert_for_eval.py     # 将 *.jsonl 推理结果解析为评测脚本所需的 {id: {六字段}} JSON
├─ eval.py                 # 评测脚本（字级 ROUGE-1/2/L；也可快速打印平均分）
│
├─ outputs/                # 结果与中间文件
│  ├─ imcs_mrg_test_qwen.jsonl      # Qwen 推理原始结果（逐行：id + pred_report）
│  ├─ imcs_mrg_test_llama.jsonl     # Llama 推理原始结果
│  ├─ pred_for_eval_qwen.json       # 转换后：{id: {主诉:..., 现病史:..., ...}}
│  └─ pred_for_eval_llama.json
│
├─ results.txt             # 记录一次完整评测的关键数值（可提交/存档）
└─ test.json               # 示例/占位（与数据集无直接绑定）


说明

服务端：两个 .sh 脚本分别常驻启动 Qwen / Llama（vLLM OpenAI 兼容端点）。

客户端：run_infer_imcs_mrg.py 调用本地端点生成“六段式病历”；convert_for_eval.py 只做格式转换；eval.py 计算 ROUGE。

outputs：保留原始 .jsonl 与转换后的 pred_for_eval_*.json，方便复查与复算。

2. 数据与任务

数据集：IMCS-V2-MRG（中文门诊多轮对话 → 结构化病历 6 段：主诉/现病史/辅助检查/既往史/诊断/建议）。

切分：dev 和/或 test 具金标可评测；IMCS-V2-MRG_test.json 为无金标提交集（只做生成）。

输出格式：强制 6 个中文小节标题；避免缺段/串栏。

3. 环境与部署（本地 3090×8，可选多卡）

启动模型服务（tmux 两个窗口分别执行）：

# Qwen (示例：GPU 0,1；端口 8001；对外模型名 qwen)
bash scripts/start_qwen.sh

# Llama (示例：GPU 2,3；端口 8002；对外模型名 llama)
bash scripts/start_llama.sh


--served-model-name 决定推理时 model 字段（此处为 qwen、llama）。
若遇到 outlines/pyairports 依赖告警，可按需安装或在启动脚本禁用 guided decoding（本项目默认常规生成即可）。

4. 一键复现实验
4.1 推理
cd medbench_run

# Qwen（test 或 dev 任选其一）
python run_infer_imcs_mrg.py --which qwen  --split test --max_tokens 512

# Llama
python run_infer_imcs_mrg.py --which llama --split test --max_tokens 512

4.2 转换为评测输入格式
# 解析 pred_report → {id: {主诉:..., 现病史:..., ...}}
python convert_for_eval.py --in_jsonl outputs/imcs_mrg_test_qwen.jsonl  --out_json outputs/pred_for_eval_qwen.json
python convert_for_eval.py --in_jsonl outputs/imcs_mrg_test_llama.jsonl --out_json outputs/pred_for_eval_llama.json

4.3 评测（字级 ROUGE）
# gold_path 指向带金标的 dev/test 文件
python eval.py --gold_path /data/haoranma/data/IMCS-V2-MRG/IMCS-V2_test.json \
               --pred_path outputs/pred_for_eval_qwen.json

python eval.py --gold_path /data/haoranma/data/IMCS-V2-MRG/IMCS-V2_test.json \
               --pred_path outputs/pred_for_eval_llama.json

5. 结果
模型	ROUGE-1	ROUGE-2	ROUGE-L	平均分 (=三项算术平均)
Qwen2.5-7B-Instruct	0.4620	0.2803	0.3883	0.3769
Llama 指令版	0.4382	0.2517	0.3752	0.3550

总体：Qwen 全面领先，平均分较 Llama +6.16%。

细节：优势主要体现在 ROUGE-2（短语级一致性、中文搭配复现）。

注：若在 test_input（无金标）上运行，评测将返回 N/A，仅用于提交预测。

6. 设计要点与实践经验

提示词工程：中文“六段式”模板能显著提高结构稳定性；建议设置每段 2–4 句，禁止新增/缺失标题。

解码超参：temperature=0.2、max_tokens=512 在本任务较稳；可网格搜索长度/温度以平衡信息完整性与啰嗦度。

评测口径：中文 字级 ROUGE 较能反映模板化输出的相似度；平均分作为简明总分用于横向比较。

工程化：推理/转换/评测分离；结果统一落盘到 outputs/ 便于复查与复算。

GPU 利用：在 vLLM 中可按显存调节 --tensor-parallel-size、--max-num-seqs、--gpu-memory-utilization 达到吞吐与稳定性的平衡。

7. 常见问题（FAQ）

评测显示 N/A：所用切分无金标（如 IMCS-V2-MRG_test.json）。请改用 dev 或带金标的 test。

model not found (404)：推理时 model 必须等于服务器 --served-model-name（如 qwen / llama）。

依赖告警（outlines/pyairports）：安装 outlines==0.0.46 与 pyairports==0.0.1 或在启动脚本禁用 guided decoding。

速度/显存：降低 --max_tokens（如 256/384），或在 vLLM 降 --max-num-seqs；必要时分批推理。

8. 许可与致谢

数据集与模型：遵循各自发布条款与许可。

用途：学术研究与课程作业复现。
