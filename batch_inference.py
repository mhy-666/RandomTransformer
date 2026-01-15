#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Inference Script
遍历指定文件夹下的所有实验，对每个final_model进行评估
只评估WikiText-103 Perplexity
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset


def evaluate_wikitext103_ppl(model, tokenizer, device, max_length=1024):
    """
    评估WikiText-103 Perplexity
    使用滑动窗口方式计算完整perplexity
    """
    
    model.eval()
    
    # 将所有文本拼接
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = max_length
    stride = max_length
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        # print(input_ids)
        # print(target_ids)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    ppl_value = ppl.item()
    
    print(f"WikiText-103 - Perplexity: {ppl_value:.2f}")
    return ppl_value


def find_all_final_models(base_dir):
    """在base_dir下递归查找所有的final_model文件夹"""
    base_path = Path(base_dir)
    final_models = []
    
    print(f"Searching for final_model folders in: {base_dir}")
    
    for model_dir in base_path.rglob("final_model"):
        if model_dir.is_dir():
            rel_path = model_dir.relative_to(base_path)
            exp_name = str(rel_path.parent) if rel_path.parent != Path('.') else rel_path.name
            final_models.append((exp_name, str(model_dir)))
    
    print(f"Found {len(final_models)} final_model folders")
    return sorted(final_models)


def load_model_and_tokenizer(model_path, device="cuda"):
    """加载模型和tokenizer"""
    print(f"Loading model from: {model_path}")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def evaluate_single_model(exp_name, model_path, args):
    """评估单个模型"""
    print("=" * 80)
    print(f"Evaluating: {exp_name}")
    print("=" * 80)
    
    model, tokenizer = load_model_and_tokenizer(model_path, args.device)
    if model is None:
        return None
    
    try:
        ppl = evaluate_wikitext103_ppl(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_length=args.max_length
        )
        
        results = {
            'experiment_name': exp_name,
            'model_path': model_path,
            'wikitext103_ppl': ppl
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Batch inference on all final models (WikiText-103 only)")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing experiment folders")
    parser.add_argument("--output_file", type=str, default="all_results.csv",
                       help="Output CSV file name (will be saved in base_dir)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=1024)
    
    args = parser.parse_args()
    
    
    # 查找所有模型
    final_models = find_all_final_models(args.base_dir)
    
    if not final_models:
        print("No final_model folders found!")
        return
    
    # 评估所有模型
    all_results = []
    
    for exp_name, model_path in tqdm(final_models, desc="Evaluating models"):
        result = evaluate_single_model(exp_name, model_path, args)
        if result is not None:
            all_results.append(result)
            
            # 实时保存单个结果
            single_result_path = os.path.join(args.base_dir, f"{exp_name.replace('/', '_')}_results.json")
            with open(single_result_path, 'w') as f:
                json.dump(result, f, indent=2)
    
    # 保存汇总结果
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 按perplexity排序（从低到高）
        df = df.sort_values('wikitext103_ppl')
        
        # 保存CSV
        csv_path = os.path.join(args.base_dir, args.output_file)
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {csv_path}")
        print(f"{'='*80}")
        
        # 保存JSON
        json_path = os.path.join(args.base_dir, args.output_file.replace('.csv', '.json'))
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON results saved to: {json_path}")
        
        # 打印统计
        print("\n" + "="*80)
        print("WikiText-103 Perplexity Summary")
        print("="*80)
        print(f"Best (lowest) PPL:  {df['wikitext103_ppl'].min():.2f}")
        print(f"Worst (highest) PPL: {df['wikitext103_ppl'].max():.2f}")
        print(f"Mean PPL:           {df['wikitext103_ppl'].mean():.2f}")
        print(f"Median PPL:         {df['wikitext103_ppl'].median():.2f}")
        
        # 打印最佳模型
        print("\n" + "="*80)
        print("Top 10 Models by WikiText-103 Perplexity (lower is better)")
        print("="*80)
        top10 = df.head(10)[['experiment_name', 'wikitext103_ppl']]
        print(top10.to_string(index=False))
        
        print("\n" + "="*80)
        print("Bottom 5 Models (highest perplexity)")
        print("="*80)
        bottom5 = df.tail(5)[['experiment_name', 'wikitext103_ppl']]
        print(bottom5.to_string(index=False))
        
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
