# -*- coding: utf-8 -*-
import os, json, argparse, time, re
from openai import OpenAI

# ===== 修改这里：你的数据目录 =====
DATA_DIR_DEFAULT = "./IMCS-V2-MRG"

SECS = ["主诉","现病史","辅助检查","既往史","诊断","建议"]

PROMPT_ZH = """你是中文临床医生助理。给你一段“患者-医生”的完整对话（含患者自述）。请阅读后生成一份**结构化医疗报告**，严格用以下6个中文小节标题，且每个小节写2-4句，禁止新增或缺少标题：
主诉：
现病史：
辅助检查：
既往史：
诊断：
建议：

请仅输出上述6行小节及其内容，不要输出其它说明。

【对话】：
{dialog}
"""

def build_dialogue(ex):
    parts = []
    if ex.get("self-report"):
        parts.append(f"患者：{ex['self-report']}")
    # 常见字段名：dialogue 是列表，元素含 speaker/sentence
    for turn in ex.get("dialogue", []):
        spk = turn.get("speaker","")
        utt = turn.get("sentence","")
        parts.append(f"{spk}：{utt}")
    return "\n".join(parts)

def canonicalize_report(text: str) -> str:
    text = text.strip().replace(":", "：")
    blocks = {}
    for sec in SECS:
        m = re.search(rf"{sec}：([\s\S]*?)(?=(?:^|\n)(?:{'|'.join(SECS)})：|$)", text)
        blocks[sec] = (m.group(1).strip() if m else "")
    return "\n".join(f"{sec}：{blocks[sec]}".rstrip() for sec in SECS)

def load_split(data_dir: str, split: str):
    """
    文件映射：
      train      -> IMCS-V2_train.json
      dev        -> IMCS-V2_dev.json
      test       -> IMCS-V2_test.json        （有标签，可评测）
      test_input -> IMCS-V2-MRG_test.json    （无标签，仅生成）
    JSON 可能是 {id: {...}} 或 [ {...} ]，两种都兼容。
    """
    name_map = {
        "train": "IMCS-V2_train.json",
        "dev": "IMCS-V2_dev.json",
        "test": "IMCS-V2_test.json",
        "test_input": "IMCS-V2-MRG_test.json",
    }
    path = os.path.join(data_dir, name_map[split])
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    if isinstance(raw, dict):
        for k, ex in raw.items():
            items.append({"id": k, **ex})
    elif isinstance(raw, list):
        # 若列表里本身含 id 字段则复用；否则用顺序号
        for i, ex in enumerate(raw):
            ex = dict(ex)
            ex.setdefault("id", ex.get("example_id", str(i)))
            items.append(ex)
    else:
        raise ValueError(f"Unsupported JSON root type: {type(raw)}")

    # 统一 gold 报告的线性化（评测时用）
    for ex in items:
        refs = ex.get("report", [])
        gold_reports = []
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, dict):
                    gold_reports.append("\n".join([f"{sec}：{r.get(sec,'').strip()}" for sec in SECS]))
        ex["gold_reports_linear"] = gold_reports  # 可能为空（test_input）
    return items

def run(endpoint: str, model_name: str, data_dir: str, split: str, out_path: str,
        max_tokens: int = 512, temperature: float = 0.2):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = load_split(data_dir, split)
    total = len(ds)                 #  总样本数
    client = OpenAI(base_url=endpoint, api_key="EMPTY")

    t0 = time.time()                #  起始时间
    n = 0                           #  已完成计数
    lat_sum = 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            dialog = build_dialogue(ex)
            prompt = PROMPT_ZH.format(dialog=dialog)

            s = time.time()
            resp = client.chat.completions.create(
                model=model_name,   # "qwen" 或 "llama"（与 /v1/models 的 id 一致）
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            lat_sum += (time.time() - s)
            text = resp.choices[0].message.content.strip()
            pred = canonicalize_report(text)

            rec = {
                "id": ex["id"],
                "pred_report": pred,
                "gold_reports": ex.get("gold_reports_linear", []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            if n % 20 == 0 or n == total:   # 每 20 条提示一次
                elapsed = time.time() - t0
                avg = n / max(elapsed, 1e-9)
                eta = (total - n) / max(avg, 1e-9)
                print(f"[{n}/{total}] {n/total*100:5.1f}% | avg {avg:.2f}/s | ETA {eta/60:.1f} min", flush=True)

    stats = {
        "samples": n,
        "wall_time_sec": round(time.time()-t0,3),
        "avg_req_latency_sec": round(lat_sum/max(n,1),3),
        "throughput_req_per_sec": round(n/(time.time()-t0+1e-9),3),
    }
    print(f"[DONE] saved -> {out_path}")
    print("[RUNTIME]", stats)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["qwen","llama"], required=True)
    ap.add_argument("--data_dir", default=DATA_DIR_DEFAULT)
    ap.add_argument("--split", choices=["train","dev","test","test_input"], default="dev")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    out_path = os.path.join(args.out_dir, f"imcs_mrg_{args.split}_{args.which}.jsonl")
    if args.which == "qwen":
        run("http://127.0.0.1:8001/v1", "qwen", args.data_dir, args.split, out_path,
            max_tokens=args.max_tokens, temperature=args.temperature)
    else:
        run("http://127.0.0.1:8002/v1", "llama", args.data_dir, args.split, out_path,
            max_tokens=args.max_tokens, temperature=args.temperature)
