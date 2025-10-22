#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse

SECS = ["主诉", "现病史", "辅助检查", "既往史", "诊断", "建议"]
PAT = re.compile(rf"({'|'.join(SECS)})：([\s\S]*?)(?=\n(?:{'|'.join(SECS)})：|$)")

def parse_pred_report(text: str):
    """把整段 '主诉：...\\n现病史：...\\n...' 解析成 6 个字段。"""
    text = (text or "").replace(":", "：").strip()
    out = {k: "" for k in SECS}
    for sec, val in PAT.findall(text):
        out[sec] = val.strip()
    return out

def main():
    ap = argparse.ArgumentParser(description="IMCS-MRG 预测结果 -> 评测输入格式 转换器（最小版）")
    ap.add_argument("--in_jsonl", required=True, help="输入：推理产生的 JSONL（含 id、pred_report）")
    ap.add_argument("--out_json", required=True, help="输出：评测脚本需要的 JSON（{id: {六字段}}）")
    args = ap.parse_args()

    preds = {}
    cnt = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("id"))
            rep = parse_pred_report(obj.get("pred_report", ""))
            preds[rid] = rep
            cnt += 1

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"[OK] converted {cnt} items -> {args.out_json}")

if __name__ == "__main__":
    main()
