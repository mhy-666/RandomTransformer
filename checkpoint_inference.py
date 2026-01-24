#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint Evolution Inference Script
遍历单个实验文件夹下的所有checkpoint，评估不同训练步数的性能
只评估WikiText-103 Perplexity
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

def evaluate_wikitext103_ppl(model, tokenizer, device, max_length=1024):
    """评估WikiText-103 Perplexity"""
    print("\n[3/4] Evaluating WikiText-103...")
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    encodings = tokenizer("".join(test["text"]), return_tensors="pt")
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
    
    print(f"WikiText-103 PPL: {ppl_value:.2f}")
    return ppl_value


def find_all_checkpoints(exp_dir, total_steps=None):
    """在实验文件夹下查找所有的checkpoint"""
    exp_path = Path(exp_dir)
    checkpoints = []
    
    print(f"Searching for checkpoints in: {exp_dir}")
    
    for ckpt_dir in exp_path.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            match = re.search(r'checkpoint-(\d+)', ckpt_dir.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, str(ckpt_dir)))
    
    final_model_path = exp_path / "final_model"
    if final_model_path.exists() and final_model_path.is_dir():
        final_step = total_steps if total_steps is not None else 999999
        checkpoints.append((final_step, str(final_model_path)))
    
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints")
    for step, path in checkpoints:
        print(f"  Step {step}: {Path(path).name}")
    
    return checkpoints


def load_model_and_tokenizer(model_path, device="cuda"):
    """加载模型和tokenizer"""
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None


def evaluate_checkpoint(step, ckpt_path, args):
    """评估单个checkpoint"""
    print(f"\n{'='*80}")
    print(f"Evaluating checkpoint at step {step}")
    print(f"{'='*80}")
    
    model, tokenizer = load_model_and_tokenizer(ckpt_path, args.device)
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
            'step': step,
            'checkpoint_path': ckpt_path,
            'wikitext103_ppl': ppl
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating checkpoint at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        del model
        torch.cuda.empty_cache()


def plot_perplexity(df, output_dir):
    """绘制perplexity曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['step'], df['wikitext103_ppl'], marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('WikiText-103 Perplexity', fontsize=12)
    ax.set_title('WikiText-103 Perplexity Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标记最优点
    best_idx = df['wikitext103_ppl'].idxmin()
    best_step = df.loc[best_idx, 'step']
    best_ppl = df.loc[best_idx, 'wikitext103_ppl']
    ax.scatter([best_step], [best_ppl], color='green', s=200, 
              marker='*', zorder=5, label=f'Best: {best_ppl:.2f} @ step {best_step}')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'checkpoint_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints (WikiText-103 only)")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--output_name", type=str, default="checkpoint_evolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=999999)
    
    args = parser.parse_args()

    
    # 查找checkpoints
    checkpoints = find_all_checkpoints(args.exp_dir, args.total_steps)
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    checkpoints = [(step, path) for step, path in checkpoints 
                   if args.start_step <= step <= args.end_step]
    
    print(f"\nWill evaluate {len(checkpoints)} checkpoints")
    
    # 评估所有checkpoint
    all_results = []
    
    for step, ckpt_path in checkpoints:
        result = evaluate_checkpoint(step, ckpt_path, args)
        if result is not None:
            all_results.append(result)
            
            temp_csv = os.path.join(args.exp_dir, f"{args.output_name}_temp.csv")
            pd.DataFrame(all_results).to_csv(temp_csv, index=False)
    
    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values('step')
        
        csv_path = os.path.join(args.exp_dir, f"{args.output_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {csv_path}")
        
        json_path = os.path.join(args.exp_dir, f"{args.output_name}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON results saved to: {json_path}")
        
        print(f"\n{'='*80}")
        print("Checkpoint Performance Summary")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        best_idx = df['wikitext103_ppl'].idxmin()
        print(f"\n{'='*80}")
        print("Best Checkpoint")
        print(f"{'='*80}")
        print(f"Step: {df.loc[best_idx, 'step']}, PPL: {df.loc[best_idx, 'wikitext103_ppl']:.2f}")
        
        if not args.skip_plot:
            try:
                plot_perplexity(df, args.exp_dir)
            except Exception as e:
                print(f"Error plotting: {e}")
        
        temp_csv = os.path.join(args.exp_dir, f"{args.output_name}_temp.csv")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
