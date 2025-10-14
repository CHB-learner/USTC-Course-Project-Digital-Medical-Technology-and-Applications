# filename: ner_min_bert_cn.py
# pip install -U transformers datasets seqeval accelerate

import json, argparse
from typing import List, Dict
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer, pipeline)
from seqeval.metrics import f1_score, precision_score, recall_score

def read_jsonl(path): 
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f)

def infer_types(train_items):
    s=set()
    for ex in train_items:
        for ent in ex.get("entities", []):
            s.add(ent["type"])
    return sorted(s)

def make_labels(types):
    labels=["O"]
    for t in types: labels += [f"B-{t}", f"I-{t}"]
    id2label={i:l for i,l in enumerate(labels)}
    label2id={l:i for i,l in id2label.items()}
    return labels, id2label, label2id

def char_bio(text, ents):
    L=len(text); y=["O"]*L
    for e in ents or []:
        s,eidx,t=e["start_idx"], e["end_idx"], e["type"]
        if 0<=s<eidx<=L:
            y[s]=f"B-{t}"
            for k in range(s+1,eidx): y[k]=f"I-{t}"
    return y

def align(tokenizer, texts, entities, label2id, max_len):
    enc = tokenizer(texts, truncation=True, max_length=max_len, return_offsets_mapping=True)
    all_labels=[]
    for offs, txt, ents in zip(enc["offset_mapping"], texts, entities):
        bio = char_bio(txt, ents)
        labs=[]
        for s,e in offs:
            if s==e: labs.append(-100)       # special tokens
            else:    labs.append(label2id.get(bio[s], label2id["O"]))
        all_labels.append(labs)
    enc.pop("offset_mapping")
    enc["labels"]=all_labels
    enc["text"]=texts
    return Dataset.from_dict(enc)

def metrics_builder(id2label):
    def comp(p):
        preds = np.argmax(p.predictions, axis=-1)
        y_true=[]; y_pred=[]
        for pl, tl in zip(preds, p.label_ids):
            t_str=[]; p_str=[]
            for p_i, t_i in zip(pl, tl):
                if t_i==-100: continue
                t_str.append(id2label[int(t_i)])
                p_str.append(id2label[int(p_i)])
            y_true.append(t_str); y_pred.append(p_str)
        return {
            "precision": precision_score(y_true,y_pred),
            "recall":    recall_score(y_true,y_pred),
            "f1":        f1_score(y_true,y_pred)
        }
    return comp



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_file", default="Project_1/data/CMeEE-V2/CMeEE-V2_train.json")
    ap.add_argument("--dev_file",   default="Project_1/data/CMeEE-V2/CMeEE-V2_dev.json")
    ap.add_argument("--test_file",  default="Project_1/data/CMeEE-V2/CMeEE-V2_test.json")
    ap.add_argument("--model_name", default="google-bert/bert-base-chinese")
    ap.add_argument("--out_dir",    default="ner_bert_cn")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz",    type=int, default=16)
    ap.add_argument("--lr",     type=float, default=3e-5)
    ap.add_argument("--max_len",type=int, default=512)
    # 新增：默认训练；可选 predict / train_predict
    ap.add_argument("--mode", choices=["train", "predict", "train_predict"], default="train")
    args=ap.parse_args()

    train_items = read_jsonl(args.train_file)
    dev_items   = read_jsonl(args.dev_file)
    test_items  = read_jsonl(args.test_file)

    types = infer_types(train_items)
    labels,id2label,label2id = make_labels(types)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # build datasets
    train_ds = align(tok, [x["text"] for x in train_items],
                          [x.get("entities",[]) for x in train_items],
                          label2id, args.max_len)
    dev_ds   = align(tok, [x["text"] for x in dev_items],
                          [x.get("entities",[]) for x in dev_items],
                          label2id, args.max_len)

    # model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    def do_train():
        collator = DataCollatorForTokenClassification(tokenizer=tok)
        # 仅用旧版也支持的参数
        targs = TrainingArguments(
            output_dir=args.out_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            num_train_epochs=args.epochs,
            logging_steps=50,
            save_steps=500,      # 简单按步保存
            # 去掉 evaluation_strategy / save_strategy / load_best_model_at_end / metric_for_best_model
            # 有的旧版也没有 report_to；若再报错，就删掉下面这一行
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=metrics_builder(id2label),
        )
        trainer.train()
        # 手动评估与保存
        print("Eval metrics:", trainer.evaluate(eval_dataset=dev_ds))
        trainer.save_model(args.out_dir)
        tok.save_pretrained(args.out_dir)


    def do_predict():
        from transformers import pipeline
        nlp = pipeline(
            "token-classification",
            model=args.out_dir,     # 用训练产物
            tokenizer=args.out_dir,
            aggregation_strategy="simple"
        )
        results=[]
        for ex in test_items:
            text = ex["text"]
            outs = nlp(text)
            ents=[]
            for o in outs:
                ents.append({
                    "start_idx": o["start"],
                    "end_idx":   o["end"],
                    "type":      o["entity_group"],
                    "entity":    text[o["start"]:o["end"]]
                })
            results.append({"text": text, "entities": ents})
        with open(f"{args.out_dir}/test_predictions.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.out_dir}/test_predictions.json")

    if args.mode == "train":
        do_train()
    elif args.mode == "predict":
        do_predict()
    else:  # train_predict
        do_train()
        do_predict()


if __name__ == "__main__":
    main()


