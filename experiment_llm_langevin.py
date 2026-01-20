#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import random
import numpy as np
import wandb 
import os  
import torch.nn as nn 
from tqdm.auto import tqdm
import argparse
import json 
from sklearn.decomposition import PCA
import sys
import logging
from itertools import chain
from transformers.testing_utils import CaptureLogger
import transformers
from random_backprop_enhanced import setup_random_backprop_experiment
from random_forward_enhanced import setup_random_forward_experiment 
from transformers.trainer_callback import TrainerCallback
from langevin_baseline import setup_langevin_baseline, DisableLangevinCallback

# 创建logger
logger = logging.getLogger(__name__)

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='transformer')
parser.add_argument('--model_size', type=str, default='gpt2',
                    choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--weight_frozen', type=int, default=1,
                    help='-1: no training (zero-shot); 0: full finetune; 1: freeze attn+mlp; '
                         '2: freeze attn only; 3: freeze mlp only; 100: freeze attn+mlp+pos')
# 新增参数：QKV单位矩阵初始化
parser.add_argument('--qkv_identity_init', type=str, default='none',
                    choices=['none', 'all', 'shallow', 'mixed'],
                    help='QKV identity initialization strategy: '
                         'none: no special init (default); '
                         'all: initialize all layers with identity; '
                         'shallow: initialize first 2 layers only; '
                         'mixed: initialize with alpha*I + beta*random')
parser.add_argument('--identity_alpha', type=float, default=0.8,
                    help='Weight for identity component in mixed initialization (0-1)')
parser.add_argument('--identity_beta', type=float, default=0.2,
                    help='Weight for random component in mixed initialization (0-1)')
parser.add_argument('--per_device_train_batch_size', type=int, default=4)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--train_samples', type=int, default=1000) 
parser.add_argument('--eval_nq_samples', type=int, default=500)
parser.add_argument('--eval_lambada_samples', type=int, default=1000)
parser.add_argument('--eval_wikitext_samples', type=int, default=100)
parser.add_argument('--project_name', type=str, default='gpt2_frozen_comprehensive')
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./outputs')
parser.add_argument('--skip_training', action='store_true', help='Skip training (for zero-shot eval)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--freeze_layers', type=str, default='', help='如low:0-5,high:6-11,custom:2,4,7')
parser.add_argument('--embedding_swap', type=str, default='', help='如path/to/embA.pth,path/to/embB.pth')
parser.add_argument('--embedding_mix', type=float, default=0.5, help='embedding混合比例')
parser.add_argument('--layer_ablate', type=int, default=-1, help='消融层号')
parser.add_argument('--layer_patch', type=int, default=-1, help='patch层号')
parser.add_argument('--pca_layer', type=int, default=-1, help='PCA可视化层号')
parser.add_argument('--from_scratch', action='store_true', 
                   help='Initialize model from scratch')
parser.add_argument('--max_steps', type=int, default=10000, 
                   help='training total iterations')
parser.add_argument('--freeze_qkv_components', type=str, default='all',
                   choices=['all', 'none', 'q', 'k', 'v', 'qk', 'qv', 'kv'],
                   help='Which QKV components to freeze (e.g., "qk" freezes Q and K, keeps V trainable)')
parser.add_argument('--qkv_init_components', type=str, default='all',
                   help='Which QKV components to apply special initialization. Format: "q:identity,k:identity,v:random" or "qk:identity,v:random" or "all:identity"')
parser.add_argument('--share_attention_weights', action='store_true',
                   help='Share attention weights across all layers')
parser.add_argument('--share_mlp_weights', action='store_true',
                   help='Share MLP weights across all layers')
parser.add_argument('--max_length', type=int, default=256, help='max_length')
parser.add_argument('--random_backprop_strategy', type=str, default='none',
                    choices=['none', 'full_random', 'low_rank_projection'],
                    help='Random backpropagation strategy')
parser.add_argument('--projection_type', type=str, default='random',
                    choices=['random', 'rotation'],
                    help='Projection matrix type for low_rank_projection')
parser.add_argument('--projection_rank', type=int, default=None,
                    help='Rank of projection matrix (None = full rank)')
parser.add_argument('--resample_every_batch', action='store_true',
                    help='Resample random parameters every batch')
parser.add_argument('--load_pretrained_path', type=str, default=None,
                    help='Path to load pretrained model checkpoint from previous stage')
# parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
#                     help='Training stage: 1=train attn, 2=train mlp')
parser.add_argument('--freeze_strategy_override', type=str, default=None,
                    choices=['freeze_mlp_train_attn', 'freeze_attn_train_mlp', 'freeze_embeddings_train_all'],
                    help='Override freeze strategy for multi-stage training')
parser.add_argument('--apply_random_backprop_to_layers', type=str, default='all',
                   choices=['all', 'attn', 'mlp', 'attn_mlp'],
                   help='Which layers to apply random backprop even in full training')
parser.add_argument('--disable_random_bp_at_ratio', type=float, default=1.0,
                   help='Disable random backprop after this ratio of training (e.g., 0.9 for last 10%)')

parser.add_argument('--random_forward_strategy', type=str, default='none',
                    choices=['none', 'additive_noise', 'full_random'],
                    help='Random forward propagation strategy')
parser.add_argument('--forward_noise_scale', type=float, default=0.02,
                    help='Noise scale for additive_noise strategy')
parser.add_argument('--forward_projection_type', type=str, default='random',
                    choices=['random', 'rotation'],
                    help='Projection matrix type for full_random strategy')
parser.add_argument('--forward_projection_rank', type=int, default=None,
                    help='Rank of projection matrix for full_random (None = full rank)')
parser.add_argument('--forward_resample_every_batch', action='store_true',
                    help='Resample random weights/noise every batch for forward')
parser.add_argument('--apply_random_forward_to_layers', type=str, default='all',
                    choices=['all', 'attn', 'mlp', 'attn_mlp'],
                    help='Which layers to apply random forward')
parser.add_argument('--disable_random_forward_at_ratio', type=float, default=1.0,
                    help='Disable random forward after this ratio of training (e.g., 0.9)')

parser.add_argument('--use_langevin_baseline', action='store_true',
                    help='Use Langevin dynamics noise instead of random backprop')
parser.add_argument('--langevin_noise_scale', type=float, default=0.01,
                    help='Noise scale for Langevin dynamics (η)')
parser.add_argument('--langevin_temperature', type=float, default=1.0,
                    help='Temperature parameter for Langevin sampling')
parser.add_argument('--langevin_precond', action='store_true',
                    help='Use preconditioner (RMSprop-style) for Langevin noise')
parser.add_argument('--langevin_apply_to', type=str, default='embedding',
                    choices=['all', 'embedding', 'attn', 'mlp'],
                    help='Which layers to apply Langevin noise')
parser.add_argument('--disable_langevin_at_ratio', type=float, default=1.0,
                    help='Disable Langevin noise after this ratio of training')


args = parser.parse_args()  
print("weight decay")
print(args.weight_decay)
# ============ 设置随机种子 ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ============ 初始化 wandb ============
freeze_strategy_names = {
    # 原始策略
    -1: "zeroshot",
    0: "full_finetune",
    1: "freeze_attn_mlp",              # Freeze Attn+MLP, train Emb
    2: "freeze_attn_only",             # Freeze Attn, train MLP+Emb
    3: "freeze_mlp_only",              # Freeze MLP, train Attn+Emb
    100: "freeze_attn_mlp_pos",        # Freeze Attn+MLP+Pos, train wte+lm_head
    
    # 对称策略 (与原始策略sum=300)
    297: "train_mlp_only",             # Symmetric to 3: Freeze Attn+Emb, train MLP only
    298: "train_attn_only",            # Symmetric to 2: Freeze MLP+Emb, train Attn only
    299: "train_attn_mlp_only",        # Symmetric to 1: Freeze Emb, train Attn+MLP
}


run_name = args.run_name or f"{freeze_strategy_names.get(args.weight_frozen, 'custom')}_{args.model_size}_seed{args.seed}"
if args.qkv_identity_init != 'none':
    run_name += f"_qkv_{args.qkv_identity_init}"
    
wandb.init(project=args.project_name, name=run_name, config=vars(args))

# ============ 初始化 tokenizer ============
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
print("什么是eos token")

# ============ 数据集加载函数 ============
def load_natural_questions_data(split='validation', max_samples=500):
    """加载 Natural Questions 数据集（使用 SQuAD 作为替代）"""
    print(f"Loading Natural Questions ({split}, max_samples={max_samples})...")
    try:
        dataset = load_dataset('squad', split='validation' if split == 'validation' else 'train', keep_in_memory=True)
        processed_data = []
        for idx, example in enumerate(dataset):
            if idx >= max_samples:
                break
            question = example['question']
            answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ""
            context = example['context']
            if answer:
                processed_data.append({
                    'question': question,
                    'answer': answer,
                    'context': context[:500]
                })
        print(f"Loaded {len(processed_data)} NQ samples")
        return processed_data
    except Exception as e:
        print(f"Error loading NQ: {e}")
        return []

def load_lambada_data(split='test', max_samples=1000):
    """加载 LAMBADA 数据集"""
    print(f"Loading LAMBADA ({split}, max_samples={max_samples})...")
    try:
        dataset = load_dataset('lambada', split=split, keep_in_memory=True)
        processed_data = []
        for idx, example in enumerate(dataset):
            if idx >= max_samples:
                break
            text = example['text']
            words = text.split()
            if len(words) > 1:
                context = ' '.join(words[:-1])
                target = words[-1]
                processed_data.append({'context': context, 'target': target})
        print(f"Loaded {len(processed_data)} LAMBADA samples")
        return processed_data
    except Exception as e:
        print(f"Error loading LAMBADA: {e}")
        return []

def load_wikitext2_data(split='test', max_samples=None):
    """加载 WikiText-2 数据集"""
    print(f"Loading WikiText-2 ({split}, max_samples={max_samples})...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, keep_in_memory=True)
        texts = []
        for idx, example in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            if len(example['text'].strip()) > 0:
                texts.append(example['text'])
        print(f"Loaded {len(texts)} WikiText-2 samples")
        return texts
    except Exception as e:
        print(f"Error loading WikiText-2: {e}")
        return []

def load_wsc_data(split='test'):
    """加载 Winograd Schema Challenge 数据集"""
    print(f"Loading WSC ({split})...")
    try:
        dataset = load_dataset('super_glue', 'wsc', split=split, keep_in_memory=True)
        processed_data = []
        for example in dataset:
            processed_data.append({
                'text': example['text'],
                'span1': example['span1_text'],
                'span2': example['span2_text'],
                'label': example['label']
            })
        print(f"Loaded {len(processed_data)} WSC samples")
        return processed_data
    except Exception as e:
        print(f"Error loading WSC: {e}")
        return []

# ============ 评估函数 ============
def evaluate_natural_questions(model, tokenizer, device, data, max_length=512):
    """评估 Natural Questions - Exact Match"""
    if len(data) == 0:
        return 0.0
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in tqdm(data, desc="Evaluating NQ", leave=False):
            question = example['question']
            answer = example['answer'].lower().strip()
            prompt = f"Question: {question}\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            
            try:
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=inputs['input_ids'].shape[1] + 20,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_answer = generated_text[len(prompt):].strip().lower()
                
                if generated_answer and (answer in generated_answer or generated_answer in answer):
                    correct += 1
            except Exception as e:
                continue
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Natural Questions - Exact Match: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_lambada(model, tokenizer, device, data, max_length=512):
    """评估 LAMBADA - Accuracy"""
    if len(data) == 0:
        return 0.0
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in tqdm(data, desc="Evaluating LAMBADA", leave=False):
            context = example['context']
            target = example['target'].lower().strip()
            
            inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            
            try:
                outputs = model(inputs['input_ids'])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                next_token_logits = logits[0, -1, :]
                predicted_token_id = torch.argmax(next_token_logits).item()
                predicted_token = tokenizer.decode([predicted_token_id]).strip().lower()
                
                if predicted_token == target:
                    correct += 1
            except Exception as e:
                continue
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"LAMBADA - Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_wikitext2_ppl(model, tokenizer, device, texts, max_length=512):
    """评估 WikiText-2 - Perplexity"""
    if len(texts) == 0:
        return float('inf')
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Evaluating WikiText-2 PPL", leave=False):
            encodings = tokenizer(text, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            if input_ids.shape[1] < 2:
                continue

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            # print(attention_mask)
            try:
                outputs = model(input_ids, labels=labels)
                # print(input_ids.shape)
                # print(outputs['logits'].shape)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                num_valid_tokens = (labels != -100).sum().item()
                # print(loss)
                total_loss += loss.item() * num_valid_tokens
                total_tokens += num_valid_tokens
            except Exception as e:
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    # print(avg_loss)
    perplexity = np.exp(avg_loss)
    print(f"WikiText-2 - Perplexity: {perplexity:.2f}")
    return perplexity

def evaluate_wsc(model, tokenizer, device, data, max_length=512):
    """评估 Winograd Schema Challenge - Accuracy"""
    if len(data) == 0:
        return 0.0
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in tqdm(data, desc="Evaluating WSC", leave=False):
            text = example['text']
            span1 = example['span1']
            span2 = example['span2']
            label = example['label']
            
            text_with_span1 = text.replace(span2, span1)
            
            inputs_original = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            inputs_replaced = tokenizer(text_with_span1, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            
            try:
                outputs_original = model(inputs_original['input_ids'], labels=inputs_original['input_ids'])
                outputs_replaced = model(inputs_replaced['input_ids'], labels=inputs_replaced['input_ids'])
                
                loss_original = outputs_original.loss if hasattr(outputs_original, 'loss') else outputs_original['loss']
                loss_replaced = outputs_replaced.loss if hasattr(outputs_replaced, 'loss') else outputs_replaced['loss']
                
                predicted_label = 1 if loss_replaced < loss_original else 0
                
                if predicted_label == label:
                    correct += 1
            except Exception as e:
                continue
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"WSC - Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

# ============ 综合评估函数 ============
def run_all_evaluations(model, tokenizer, device, args, phase=""):
    """运行所有评估任务"""
    results = {}
    
    print("\n" + "="*70)
    print(f"Evaluation Phase: {phase}")
    print("="*70 + "\n")
    
    # Natural Questions
    print("[1/4] Evaluating Natural Questions...")
    nq_data = load_natural_questions_data(max_samples=args.eval_nq_samples)
    if len(nq_data) > 0:
        nq_acc = evaluate_natural_questions(model, tokenizer, device, nq_data)
        results['natural_questions_acc'] = nq_acc
        wandb.log({f'{phase}/natural_questions_acc': nq_acc})
    
    # LAMBADA
    print("\n[2/4] Evaluating LAMBADA...")
    lambada_data = load_lambada_data(max_samples=args.eval_lambada_samples)
    if len(lambada_data) > 0:
        lambada_acc = evaluate_lambada(model, tokenizer, device, lambada_data)
        results['lambada_acc'] = lambada_acc
        wandb.log({f'{phase}/lambada_acc': lambada_acc})
    
    # WikiText-103
    print("\n[3/4] Evaluating WikiText-103...")
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = args.max_length
    stride = args.max_length
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

    results['wikitext103_ppl'] = ppl.item()
    wandb.log({f'{phase}/wikitext103_ppl': ppl})
    # wikitext_data = load_wikitext2_data(max_samples=args.eval_wikitext_samples)
    # if len(wikitext_data) > 0:
    #     wikitext2_ppl = evaluate_wikitext2_ppl(model, tokenizer, device, wikitext_data)
    #     results['wikitext2_ppl'] = wikitext2_ppl
    #     wandb.log({f'{phase}/wikitext2_ppl': wikitext2_ppl})
    
    # WSC
    print("\n[4/4] Evaluating Winograd Schema Challenge...")
    wsc_data = load_wsc_data()
    if len(wsc_data) > 0:
        wsc_acc = evaluate_wsc(model, tokenizer, device, wsc_data)
        results['wsc_acc'] = wsc_acc
        wandb.log({f'{phase}/wsc_acc': wsc_acc})
    
    print("\n" + "="*70)
    print("Evaluation Results Summary:")
    print("="*70)
    for task, score in results.items():
        print(f"  {task:30s}: {score:.4f}")
    print("="*70 + "\n")
    
    return results

class DisableRandomBackpropCallback(TrainerCallback):
    """在训练特定步数后关闭random backprop的回调"""

    def __init__(self, random_bp_manager, disable_ratio, max_steps):
        """
        Args:
            random_bp_manager: RandomBackpropManager实例
            disable_ratio: 在总步数的这个比例后关闭 (0.9 = 90%)
            max_steps: 总训练步数
        """
        self.random_bp_manager = random_bp_manager
        self.disable_step = int(max_steps * disable_ratio)
        self.disabled = False

        print("=" * 70)
        print(f"DisableRandomBackpropCallback Initialized")
        print(f"  Will disable at step: {self.disable_step} / {max_steps} ({disable_ratio*100:.0f}%)")
        print("=" * 70)

    def on_step_begin(self, args, state, control, **kwargs):
        """在每个训练步开始时检查是否需要关闭random backprop"""
        if not self.disabled and state.global_step >= self.disable_step:
            print("\n" + "=" * 70)
            print(f"🔄 TRAINING MILESTONE: Step {state.global_step}/{args.max_steps}")
            print(f"   Disabling Random Backpropagation")
            print(f"   Switching to Standard Transformer Training")
            print("=" * 70 + "\n")

            self.random_bp_manager.disable()
            self.disabled = True

            # 记录到wandb
            wandb.log({
                'random_backprop_disabled': True,
                'disable_step': state.global_step,
                'training_phase': 'standard'
            })

class DisableRandomForwardCallback(TrainerCallback):
    """在训练特定步数后关闭 random forward 的回调"""
    
    def __init__(self, random_forward_manager, disable_ratio, max_steps):
        """
        Args:
            random_forward_manager: RandomForwardManager 实例
            disable_ratio: 在总步数的这个比例后关闭 (0.9 = 90%)
            max_steps: 总训练步数
        """
        self.random_forward_manager = random_forward_manager
        self.disable_step = int(max_steps * disable_ratio)
        self.disabled = False
        
        print("=" * 70)
        print(f"DisableRandomForwardCallback Initialized")
        print(f"  Will disable at step: {self.disable_step} / {max_steps} ({disable_ratio*100:.0f}%)")
        print("=" * 70)
    
    def on_step_begin(self, args, state, control, **kwargs):
        """在每个训练步开始时检查是否需要关闭 random forward"""
        if not self.disabled and state.global_step >= self.disable_step:
            print("\n" + "=" * 70)
            print(f"🔄 TRAINING MILESTONE: Step {state.global_step}/{args.max_steps}")
            print(f"  Disabling Random Forward Propagation")
            print(f"  Switching to Standard Forward Pass")
            print("=" * 70 + "\n")
            
            self.random_forward_manager.disable()
            self.disabled = True
            
            # 记录到 wandb
            wandb.log({
                'random_forward_disabled': True,
                'disable_step': state.global_step,
                'training_phase': 'standard_forward'
            })


# ============ QKV组件管理工具函数 ============
def parse_qkv_components(component_str):
    """
    解析QKV组件字符串
    
    输入示例:
        'all' -> ['q', 'k', 'v']
        'qk' -> ['q', 'k']
        'q' -> ['q']
    
    返回: list of components
    """
    if component_str == 'all':
        return ['q', 'k', 'v']
    elif component_str == 'none':
        return []
    else:
        components = []
        if 'q' in component_str:
            components.append('q')
        if 'k' in component_str:
            components.append('k')
        if 'v' in component_str:
            components.append('v')
        return components


def parse_qkv_init_config(init_str):
    """
    解析QKV初始化配置字符串
    
    输入示例:
        'all:identity' -> {'q': 'identity', 'k': 'identity', 'v': 'identity'}
        'qk:identity,v:random' -> {'q': 'identity', 'k': 'identity', 'v': 'random'}
        'q:identity,k:random,v:mixed' -> {'q': 'identity', 'k': 'random', 'v': 'mixed'}
    
    返回: dict mapping component to init type
    """
    config = {}
    
    if ':' not in init_str:
        # 默认所有都用同样的策略
        init_type = init_str if init_str in ['identity', 'random', 'mixed'] else 'random'
        return {'q': init_type, 'k': init_type, 'v': init_type}
    
    for part in init_str.split(','):
        components, init_type = part.split(':')
        components = components.strip()
        init_type = init_type.strip()
        
        if components == 'all':
            config = {'q': init_type, 'k': init_type, 'v': init_type}
        elif components in ['qk', 'kq']:
            config['q'] = init_type
            config['k'] = init_type
        elif components in ['qv', 'vq']:
            config['q'] = init_type
            config['v'] = init_type
        elif components in ['kv', 'vk']:
            config['k'] = init_type
            config['v'] = init_type
        else:
            # 单个组件
            for c in components:
                if c in ['q', 'k', 'v']:
                    config[c] = init_type
    
    # 确保所有组件都有配置
    for c in ['q', 'k', 'v']:
        if c not in config:
            config[c] = 'random'
    
    return config
# ============ 随机初始化函数（支持QKV细粒度控制） ============
def reinitialize_layers(model, layer_names, freeze_qkv_components='all', use_xavier=True):
    """
    对指定的层进行随机初始化,支持对QKV的细粒度控制
    参数:
        model: GPT2LMHeadModel
        layer_names: 需要重新初始化的层名称列表
        freeze_qkv_components: 'all', 'qk', 'q', 'k', 'v', 'qv', 'kv', 'none'
        use_xavier: 是否使用Xavier初始化
    """
    print("\n" + "="*70)
    print("Reinitializing layers (removing pretrained weights)...")
    print(f"QKV freeze components: {freeze_qkv_components}")
    print("="*70)
    
    qkv_components_to_freeze = parse_qkv_components(freeze_qkv_components)
    reinitialized_count = 0
    
    for name, param in model.named_parameters():
        should_reinit = any(layer_name in name for layer_name in layer_names)
        
        if should_reinit:
            # 特殊处理 QKV 的 c_attn 层
            if 'attn.c_attn.weight' in name:
                # 对 QKV 进行选择性初始化
                with torch.no_grad():
                    embed_dim = param.shape[0]
                    for i, component in enumerate(['q', 'k', 'v']):
                        if component in qkv_components_to_freeze:
                            start_idx = i * embed_dim
                            end_idx = (i + 1) * embed_dim
                            # 只初始化要冻结的组件
                            if use_xavier:
                                nn.init.xavier_normal_(param[:, start_idx:end_idx])
                            else:
                                nn.init.normal_(param[:, start_idx:end_idx], mean=0.0, std=0.02)
                            print(f"  [REINIT-{component.upper()}] {name:55s} columns {start_idx}:{end_idx}")
                    reinitialized_count += 1
            
            elif 'attn.c_attn.bias' in name:
                # bias处理
                with torch.no_grad():
                    embed_dim = param.shape[0] // 3
                    for i, component in enumerate(['q', 'k', 'v']):
                        if component in qkv_components_to_freeze:
                            start_idx = i * embed_dim
                            end_idx = (i + 1) * embed_dim
                            param[start_idx:end_idx].zero_()
                    reinitialized_count += 1
                    print(f"  [REINIT-BIAS] {name:55s}")
            
            # ===== 修改点: 添加 c_proj 的重新初始化 =====
            elif 'attn.c_proj.weight' in name or 'attn.c_proj.bias' in name:
                with torch.no_grad():
                    if 'weight' in name:
                        if use_xavier:
                            nn.init.xavier_normal_(param)
                        else:
                            nn.init.normal_(param, mean=0.0, std=0.02)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                reinitialized_count += 1
                print(f"  [REINIT-PROJ] {name:55s} {tuple(param.shape)}")
            
            else:
                # 非QKV/c_proj层的正常初始化
                with torch.no_grad():
                    if 'weight' in name:
                        if len(param.shape) >= 2:
                            if use_xavier:
                                nn.init.xavier_normal_(param)
                            else:
                                nn.init.normal_(param, mean=0.0, std=0.02)
                        else:
                            nn.init.normal_(param, mean=0.0, std=0.02)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                reinitialized_count += 1
                print(f"  [REINIT] {name:60s} {tuple(param.shape)}")
    
    print("="*70)
    print(f"Reinitialized {reinitialized_count} parameters")
    print("="*70 + "\n")
    
    return reinitialized_count


# ============ QKV特殊初始化函数（支持细粒度控制） ============
def initialize_qkv_with_identity(model, init_config, strategy='all', alpha=0.8, beta=0.2):
    """
    对QKV应用细粒度的初始化策略
    
    参数:
        model: GPT2LMHeadModel
        init_config: QKV初始化配置字典，例如:
            {'q': 'identity', 'k': 'identity', 'v': 'random'}
            {'q': 'mixed', 'k': 'mixed', 'v': 'identity'}
        strategy: 应用到哪些层 ('all', 'shallow', 'none')
        alpha: 混合初始化中单位矩阵的权重
        beta: 混合初始化中随机部分的权重
    """
    if strategy == 'none':
        print("\n[QKV Init] Strategy is 'none', keeping random initialization")
        return 0
    
    print("\n" + "="*70)
    print(f"Applying fine-grained QKV initialization (layer strategy: {strategy})")
    print(f"Component-wise init config: {init_config}")
    if any(v == 'mixed' for v in init_config.values()):
        print(f"  Alpha (identity weight): {alpha}")
        print(f"  Beta (random weight): {beta}")
    print("="*70)
    
    num_layers = len(model.transformer.h)
    initialized_count = 0
    
    for layer_idx, block in enumerate(model.transformer.h):
        # 确定是否对当前层应用初始化
        should_initialize = False
        if strategy == 'all':
            should_initialize = True
        elif strategy == 'shallow':
            should_initialize = (layer_idx < 2)
        
        if not should_initialize:
            continue
        
        attn = block.attn
        c_attn_weight = attn.c_attn.weight
        c_attn_bias = attn.c_attn.bias
        
        embed_dim = c_attn_weight.shape[0]
        
        with torch.no_grad():
            for i, component in enumerate(['q', 'k', 'v']):
                start_idx = i * embed_dim
                end_idx = (i + 1) * embed_dim
                
                init_type = init_config.get(component, 'random')
                
                if init_type == 'identity':
                    # 完全单位矩阵
                    identity_matrix = torch.eye(
                        embed_dim,
                        device=c_attn_weight.device,
                        dtype=c_attn_weight.dtype
                    )
                    c_attn_weight[:, start_idx:end_idx] = identity_matrix
                    print(f"  Layer {layer_idx:2d} {component.upper()}: identity matrix")
                
                elif init_type == 'mixed':
                    # 混合初始化
                    current_random = c_attn_weight[:, start_idx:end_idx].clone()
                    identity_part = alpha * torch.eye(
                        embed_dim,
                        device=c_attn_weight.device,
                        dtype=c_attn_weight.dtype
                    )
                    random_part = beta * current_random
                    c_attn_weight[:, start_idx:end_idx] = identity_part + random_part
                    print(f"  Layer {layer_idx:2d} {component.upper()}: mixed (α={alpha}, β={beta})")
                
                elif init_type == 'random':
                    # 保持随机初始化（不做修改）
                    print(f"  Layer {layer_idx:2d} {component.upper()}: keeping random")
                
                else:
                    print(f"  Layer {layer_idx:2d} {component.upper()}: unknown init type '{init_type}', keeping random")
            
            # bias初始化为0
            c_attn_bias.zero_()
        
        initialized_count += 1
    
    print("="*70)
    print(f"Applied special initialization to {initialized_count}/{num_layers} attention layers")
    print("="*70 + "\n")
    
    return initialized_count


# ============ 冻结权重策略（支持QKV细粒度控制） ============
# def apply_freeze_strategy(model, strategy, freeze_qkv_components='all'):
#     """
#     确定哪些层需要被冻结,支持QKV细粒度控制
#     参数:
#         freeze_qkv_components: 'all', 'qk', 'q', 'k', 'v', etc.
#     """
#     print("\n" + "="*70)
#     print(f"Determining freeze strategy: {freeze_strategy_names.get(strategy, 'custom')}")
#     print(f"QKV component control: {freeze_qkv_components}")
#     print("="*70)
    
#     layers_to_freeze = []
#     qkv_components = parse_qkv_components(freeze_qkv_components)
    
#     for name, param in model.named_parameters():
#         should_freeze = False
        
#         # 特殊处理 attention 层
#         if 'attn' in name:
#             if strategy in [1, 2, 100]:  # 需要freeze attention的策略
#                 # 对于 c_attn，我们需要记录它需要部分冻结
#                 if 'attn.c_attn' in name:
#                     # 标记为需要特殊处理
#                     should_freeze = True
#                 # ===== 修改点: 添加 c_proj 的冻结 =====
#                 elif 'attn.c_proj' in name:
#                     # c_proj也需要冻结
#                     should_freeze = True
#                 else:
#                     # 其他attention组件完全冻结
#                     should_freeze = True
        
#         # 其他层的冻结逻辑
#         elif strategy == -1:  # Zero-shot
#             should_freeze = True
#         elif strategy == 1:  # Freeze Attention + MLP
#             if not any(x in name for x in ['wte', 'wpe', 'lm_head']):
#                 should_freeze = True
#         elif strategy == 3:  # Freeze MLP only
#             if 'mlp' in name:
#                 should_freeze = True
#         elif strategy == 100:  # Freeze Attention + MLP + Pos
#             if not any(x in name for x in ['wte', 'lm_head']):
#                 should_freeze = True
        
#         if should_freeze:
#             layers_to_freeze.append(name)
    
#     print(f"Identified {len(layers_to_freeze)} parameters to be frozen")
#     if qkv_components != ['q', 'k', 'v']:
#         print(f"  Note: For c_attn layers, only {qkv_components} will be frozen")
#     print(f"  Note: c_proj (attention output projection) will also be frozen")  # 新增提示
#     print("="*70 + "\n")
    
#     return layers_to_freeze, qkv_components
def apply_freeze_strategy(model, strategy, freeze_qkv_components='all'):
    """
    Enhanced freeze strategy system with symmetric options
    
    Original strategies (0-100):
        -1: Zero-shot (no training)
        0:  Full finetune (train all)
        1:  Freeze Attention + MLP (train embeddings only)
        2:  Freeze Attention only (train MLP + embeddings)
        3:  Freeze MLP only (train Attention + embeddings)
        100: Freeze Attention + MLP + Pos (train word embeddings + lm_head only)
    
    Symmetric strategies (200-300, sum to 300 with originals):
        297: Train MLP only (freeze Attention + embeddings) [对称于3, sum=300]
        298: Train Attention only (freeze MLP + embeddings) [对称于2, sum=300]
        299: Train Attention + MLP (freeze embeddings) [对称于1, sum=300]
        200: Freeze all (对称于100, sum=300)
    """
    print("=" * 70)
    print(f"Determining freeze strategy: {strategy}")
    print(f"QKV component control: {freeze_qkv_components}")
    print("=" * 70)
    
    layers_to_freeze = []
    qkv_components = parse_qkv_components(freeze_qkv_components)
    
    # 确定哪些策略需要freeze attention
    strategies_freeze_attn = [1, 2, 100, 297, -1]
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # ========== 特殊处理 attention 层 ==========
        if 'attn' in name:
            if strategy in strategies_freeze_attn:
                if 'attn.c_attn' in name:
                    # c_attn (QKV projection) 需要特殊处理（可能部分冻结）
                    should_freeze = True
                elif 'attn.c_proj' in name:
                    # c_proj (output projection) 也需要冻结
                    should_freeze = True
                else:
                    # 其他attention组件完全冻结
                    should_freeze = True 
        
        # ========== Original strategies (0-100) ==========
        if strategy == -1:  # Zero-shot
            should_freeze = True
            
        elif strategy == 0:  # Full finetune
            should_freeze = False
            
        elif strategy == 1:  # Freeze Attention + MLP, train embeddings
            if 'mlp' in name:  # 也要freeze MLP
                should_freeze = True
            # attention已经在上面处理了
                
        elif strategy == 2:  # Freeze Attention, train MLP + embeddings
            # attention已经在上面处理了
            # MLP和embeddings不freeze
            pass
                
        elif strategy == 3:  # Freeze MLP, train Attention + embeddings
            if 'mlp' in name:
                should_freeze = True
                
        elif strategy == 100:  # Freeze Attention + MLP + Pos
            if 'mlp' in name or 'wpe' in name:
                should_freeze = True
            # attention已经在上面处理了
        
        # ========== Symmetric strategies (200-300) ==========
        elif strategy == 297:  # Train MLP only (对称于3)
            # Freeze: Attention + embeddings
            # attention已经在上面处理了
            if 'wte' in name or 'wpe' in name or 'lm_head' in name:
                should_freeze = True
            # Train: mlp only
            
        elif strategy == 298:  # Train Attention only (对称于2)
            # Freeze: MLP + embeddings
            if 'mlp' in name or 'wte' in name or 'wpe' in name or 'lm_head' in name:
                should_freeze = True
            # Train: attn only
            
        elif strategy == 299:  # Train Attention + MLP (对称于1)
            # Freeze: embeddings only
            if any(x in name for x in ['wte', 'wpe', 'lm_head']):
                should_freeze = True
            # Train: attn + mlp
        
        if should_freeze:
            layers_to_freeze.append(name)
    
    # Print strategy info
    strategy_info = get_strategy_description(strategy)
    print(f"Strategy {strategy}: {strategy_info['name']}")
    print(f"  Frozen: {strategy_info['frozen']}")
    print(f"  Trainable: {strategy_info['trainable']}")
    
    # Show symmetric relationship
    if strategy in [297, 298, 299, 200]:
        symmetric = 300 - strategy
        print(f"  ⚠️  Symmetric to strategy {symmetric} (sum = 300)")
    elif strategy in [1, 2, 3, 100]:
        symmetric = 300 - strategy
        if symmetric in [297, 298, 299, 200]:
            print(f"  ℹ️  Has symmetric strategy {symmetric} (sum = 300)")
    
    print(f"Total parameters to freeze: {len(layers_to_freeze)}")
    
    if qkv_components != ['q', 'k', 'v']:
        print(f"Note: For c_attn layers, only {qkv_components} will be frozen")
        print(f"Note: c_proj (attention output projection) will also be frozen")
    print("=" * 70)
    
    return layers_to_freeze, qkv_components

def get_strategy_description(strategy):
    """Return description for each strategy"""
    descriptions = {
        -1: {
            'name': 'Zero-shot',
            'frozen': 'All parameters',
            'trainable': 'None'
        },
        0: {
            'name': 'Full finetune',
            'frozen': 'None',
            'trainable': 'All parameters'
        },
        1: {
            'name': 'Freeze Attention + MLP',
            'frozen': 'Attention + MLP',
            'trainable': 'Embeddings (wte, wpe, lm_head)'
        },
        2: {
            'name': 'Freeze Attention only',
            'frozen': 'Attention',
            'trainable': 'MLP + Embeddings'
        },
        3: {
            'name': 'Freeze MLP only',
            'frozen': 'MLP',
            'trainable': 'Attention + Embeddings'
        },
        100: {
            'name': 'Freeze Attention + MLP + Position',
            'frozen': 'Attention + MLP + wpe',
            'trainable': 'Word embeddings (wte, lm_head)'
        },
        # Symmetric strategies
        297: {
            'name': 'Train MLP only (Symmetric to 3)',
            'frozen': 'Attention + Embeddings',
            'trainable': 'MLP only'
        },
        298: {
            'name': 'Train Attention only (Symmetric to 2)',
            'frozen': 'MLP + Embeddings',
            'trainable': 'Attention only'
        },
        299: {
            'name': 'Train Attention + MLP (Symmetric to 1)',
            'frozen': 'Embeddings',
            'trainable': 'Attention + MLP'
        },
    }
    
    return descriptions.get(strategy, {
        'name': 'Unknown strategy',
        'frozen': 'Unknown',
        'trainable': 'Unknown'
    })


def tie_layer_weights(model, component='attention'):
    """
    强制所有transformer层共享同一套权重（attention或MLP）
    注意：此函数假设权重已经通过reinitialize_layers进行了初始化
    
    参数:
    model: GPT2LMHeadModel
    component: 'attention' 或 'mlp'
    """
    print("\n" + "="*70)
    print(f"Tying {component} weights across all {len(model.transformer.h)} layers")
    print("="*70)
    
    num_layers = len(model.transformer.h)
    
    if component == 'attention':
        # 使用第0层的attention作为共享权重的源（已经被reinitialize_layers初始化过）
        source_attn = model.transformer.h[0].attn
        
        print(f"  Using Layer 0 attention as source (already initialized by reinitialize_layers)")
        
        # 将所有其他层的attention指向第0层（共享权重）
        for layer_idx in range(1, num_layers):
            model.transformer.h[layer_idx].attn = source_attn
            print(f"  Layer {layer_idx}: tied to Layer 0")
        
        # 冻结共享的attention权重
        for param in source_attn.parameters():
            param.requires_grad = False
        
        print(f"✓ All {num_layers} layers now share the same attention weights (frozen)")
        
    elif component == 'mlp':
        # 使用第0层的MLP作为共享权重的源（已经被reinitialize_layers初始化过）
        source_mlp = model.transformer.h[0].mlp
        
        print(f"  Using Layer 0 MLP as source (already initialized by reinitialize_layers)")
        
        # 将所有其他层的MLP指向第0层
        for layer_idx in range(1, num_layers):
            model.transformer.h[layer_idx].mlp = source_mlp
            print(f"  Layer {layer_idx}: tied to Layer 0")
        
        # 冻结共享的MLP权重
        for param in source_mlp.parameters():
            param.requires_grad = False
        
        print(f"✓ All {num_layers} layers now share the same MLP weights (frozen)")
    
    else:
        raise ValueError(f"Unknown component: {component}. Use 'attention' or 'mlp'")
    
    print("="*70 + "\n")
    return model


def execute_freeze(model, layers_to_freeze, qkv_components=['q', 'k', 'v']):
    """
    执行实际的冻结操作，支持QKV细粒度控制
    """
    num_frozen = 0
    num_trainable = 0
    
    print("\n" + "="*70)
    print("Executing freeze operations...")
    print("="*70)
    
    for name, param in model.named_parameters():
        if name in layers_to_freeze:
            # 特殊处理 c_attn
            if 'attn.c_attn' in name:
                # 对于权重，需要部分冻结
                if 'weight' in name:
                    embed_dim = param.shape[0]
                    
                    # 创建一个mask来标记哪些参数可训练
                    param.requires_grad = False  # 先全部冻结
                    
                    # 统计冻结和可训练的参数数量
                    for i, component in enumerate(['q', 'k', 'v']):
                        component_params = embed_dim * embed_dim
                        if component in qkv_components:
                            num_frozen += component_params
                            print(f"  [FROZEN-{component.upper()}] {name:55s} component {component}")
                        else:
                            num_trainable += component_params
                            print(f"  [TRAIN-{component.upper()}] {name:55s} component {component}")
                    
                    # 注意：PyTorch不支持部分参数冻结，需要特殊处理
                    # 如果有任何组件需要训练，整个参数需要设为可训练
                    if len(qkv_components) < 3:
                        param.requires_grad = True
                        print(f"  [WARNING] {name} set to trainable (partial freeze not directly supported)")
                        print(f"            Need to use optimizer param_groups or hooks for partial update")
                
                elif 'bias' in name:
                    # bias也需要部分处理
                    param.requires_grad = (len(qkv_components) < 3)
                    if param.requires_grad:
                        num_trainable += param.numel()
                    else:
                        num_frozen += param.numel()
            else:
                # 普通冻结
                param.requires_grad = False
                num_frozen += param.numel()
                print(f"  [FROZEN] {name:60s}")
        else:
            param.requires_grad = True
            num_trainable += param.numel()
            print(f"  [TRAINABLE] {name:60s} {tuple(param.shape)}")
    
    total_params = num_trainable + num_frozen
    print("="*70)
    print(f"Trainable: {num_trainable:,} ({num_trainable/total_params*100:.2f}%)")
    print(f"Frozen: {num_frozen:,} ({num_frozen/total_params*100:.2f}%)")
    print(f"Total: {total_params:,}")
    print("="*70 + "\n")
    
    return num_trainable, num_frozen


# ============ 自定义优化器（支持QKV部分参数更新） ============
def create_optimizer_with_qkv_control(model, layers_to_freeze, qkv_components, lr, weight_decay):
    """
    创建优化器，支持QKV组件的细粒度控制
    """
    # 如果需要部分更新QKV，使用参数分组
    if len(qkv_components) < 3 and len(qkv_components) > 0:
        print("\n" + "="*70)
        print("Creating optimizer with QKV component control...")
        print("="*70)
        
        # 需要手动处理QKV的梯度更新
        # 这里创建一个hook来在反向传播后清零frozen组件的梯度
        
        def create_qkv_mask_hook(qkv_components_to_freeze):
            """创建hook来mask掉frozen QKV组件的梯度"""
            def hook(grad):
                if grad is None:
                    return None
                
                embed_dim = grad.shape[0]
                masked_grad = grad.clone()
                
                for i, component in enumerate(['q', 'k', 'v']):
                    if component in qkv_components_to_freeze:
                        start_idx = i * embed_dim
                        end_idx = (i + 1) * embed_dim
                        masked_grad[:, start_idx:end_idx] = 0
                
                return masked_grad
            return hook
        
        # 注册hooks
        for name, param in model.named_parameters():
            if 'attn.c_attn.weight' in name and name in layers_to_freeze:
                param.register_hook(create_qkv_mask_hook(qkv_components))
                print(f"  Registered gradient mask hook for {name}")
        
        print("="*70 + "\n")
    
    # 创建标准优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    
    return optimizer


# ============ 主流程（完整版） ============
print("\n" + "="*70)
print("MODEL INITIALIZATION AND FREEZING PIPELINE")
print("="*70)

# 步骤1: 加载模型
print(f"\n[Step 1] Loading/Initializing GPT-2 model ({args.model_size})...")
if args.load_pretrained_path:
    print("=" * 70)
    print(f"Loading pretrained model from: {args.load_pretrained_path}")
    print("=" * 70)
    model = GPT2LMHeadModel.from_pretrained(args.load_pretrained_path)
    print(f"✓ Successfully loaded model")
    model.to(args.device)
elif args.from_scratch:
    # config = GPT2Config.from_pretrained(args.model_size)
    config = GPT2Config(
        vocab_size=50257,
        n_positions=args.max_length,  # 使用你参数里的 max_length
        n_ctx=args.max_length,
        n_embd=768,           # 关键修改
        n_layer=12,            # 关键修改
        n_head=12,             # 关键修改
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # resid_pdrop=0.35,  # 残差连接 Dropout
        # embd_pdrop=0.35,   # Embedding Dropout
        # attn_pdrop=0.35,   # Attention Dropout
    )
    model = GPT2LMHeadModel(config)
    print(f"✓ Initialized {args.model_size} from scratch")
else:
    model = GPT2LMHeadModel.from_pretrained(args.model_size)
    print(f"✓ Loaded pretrained {args.model_size}")

model = model.to(args.device)

# 步骤2: 确定要冻结的层和QKV组件
print(f"\n[Step 2] Determining freeze strategy...")
layers_to_freeze, qkv_components_to_freeze = apply_freeze_strategy(
    model, 
    args.weight_frozen,
    freeze_qkv_components=args.freeze_qkv_components
)
print(f"✓ Freeze config: {len(layers_to_freeze)} layers, QKV components: {qkv_components_to_freeze}")

# 步骤3: 随机初始化要冻结的层
if len(layers_to_freeze) > 0 and args.from_scratch and not args.load_pretrained_path:
    print(f"\n[Step 3] Reinitializing frozen layers...")
    num_reinit = reinitialize_layers(
        model, 
        layers_to_freeze,
        freeze_qkv_components=args.freeze_qkv_components,
        use_xavier=True
    )
    print(f"✓ Reinitialized {num_reinit} layers")
    
    wandb.log({
        'num_reinitialized_layers': num_reinit,
        'reinit_method': 'xavier_normal',
        'freeze_qkv_components': args.freeze_qkv_components
    })
else:
    print(f"\n[Step 3] Skipping reinitialization")

# 步骤4: 对QKV应用特殊初始化
if args.qkv_init_components and args.qkv_init_components != 'none':
    has_frozen_attn = any('attn' in name for name in layers_to_freeze)
    
    if has_frozen_attn:
        print(f"\n[Step 4] Applying fine-grained QKV initialization...")
        
        # 解析初始化配置
        qkv_init_config = parse_qkv_init_config(args.qkv_init_components)
        print(f"  Parsed config: {qkv_init_config}")
        
        num_qkv_init = initialize_qkv_with_identity(
            model,
            init_config=qkv_init_config,
            strategy=args.qkv_identity_init,
            alpha=args.identity_alpha,
            beta=args.identity_beta
        )
        print(f"✓ Applied special initialization to {num_qkv_init} layers")
        
        wandb.log({
            'qkv_init_config': qkv_init_config,
            'qkv_init_layers': num_qkv_init
        })
    else:
        print(f"\n[Step 4] Skipping QKV init (no frozen attention)")
else:
    print(f"\n[Step 4] Skipping QKV special initialization")

# Step 4.5: 权重共享（新增 - 在执行freeze之前）
if args.share_attention_weights or args.share_mlp_weights:
    print(f"\n[Step 4.5] Applying weight sharing...")
    
    if args.share_attention_weights:
        print("  Sharing attention weights across all layers...")
        model = tie_layer_weights(model, component='attention')
        wandb.log({'attention_weights_shared': True})
    
    if args.share_mlp_weights:
        print("  Sharing MLP weights across all layers...")
        model = tie_layer_weights(model, component='mlp')
        wandb.log({'mlp_weights_shared': True})
    
    print(f"✓ Weight sharing completed")

# 步骤5: 执行冻结
print(f"\n[Step 5] Freezing parameters...")
num_trainable, num_frozen = execute_freeze(model, layers_to_freeze, qkv_components_to_freeze)
print(f"✓ Frozen {num_frozen:,} parameters, {num_trainable:,} trainable")

wandb.log({
    'num_trainable': num_trainable,
    'num_frozen': num_frozen,
    'freeze_strategy': args.weight_frozen,
    'freeze_qkv_components': args.freeze_qkv_components
})

# 步骤6: 
if args.random_backprop_strategy != 'none':
    print("\n" + "=" * 70)
    print("Setting up Random Backpropagation Experiment")
    print("=" * 70)

    # 确定要应用random backprop的层
    if args.weight_frozen == 0:  # 全参数训练模式
        print(f"Mode: Full Parameter Training with Random Backprop")
        print(f"Applying random backprop to: {args.apply_random_backprop_to_layers}")

        # 构建层名称列表
        if args.apply_random_backprop_to_layers == 'attn':
            random_bp_layers = [f"transformer.h.{i}.attn" for i in range(model.config.n_layer)]
        elif args.apply_random_backprop_to_layers == 'mlp':
            random_bp_layers = [f"transformer.h.{i}.mlp" for i in range(model.config.n_layer)]
        elif args.apply_random_backprop_to_layers == 'attn_mlp':
            random_bp_layers = []
            for i in range(model.config.n_layer):
                random_bp_layers.append(f"transformer.h.{i}.attn")
                random_bp_layers.append(f"transformer.h.{i}.mlp")
        else:  # 'all'
            random_bp_layers = [f"transformer.h.{i}" for i in range(model.config.n_layer)]

        allow_trainable = True
        print(f"Total layers for random backprop: {len(random_bp_layers)}")

    else:  # 冻结参数模式（原有逻辑）
        print(f"Mode: Frozen Parameter Training with Random Backprop")
        random_bp_layers = layers_to_freeze
        allow_trainable = False

    # 设置随机反向传播
    random_bp_manager = setup_random_backprop_experiment(
        model=model,
        frozen_layer_names=random_bp_layers,
        strategy=args.random_backprop_strategy,
        projection_type=args.projection_type,
        projection_rank=args.projection_rank,
        resample_every_batch=args.resample_every_batch,
        random_seed=args.seed,
        device=args.device,
        allow_trainable=allow_trainable,  # 关键参数
    )

    # 记录到wandb
    wandb.log({
        "random_backprop_strategy": args.random_backprop_strategy,
        "projection_type": args.projection_type,
        "projection_rank": args.projection_rank,
        "resample_every_batch": args.resample_every_batch,
        "apply_to_layers": args.apply_random_backprop_to_layers,
        "allow_trainable": allow_trainable,
        "num_random_bp_layers": len(random_bp_layers),
    })

    print("=" * 70 + "\n")

else:
    random_bp_manager = None

if args.random_forward_strategy != 'none':
    print("\n" + "=" * 70)
    print("Setting up Random Forward Propagation")
    print("=" * 70)
    
    # 根据配置确定目标层
    random_forward_layers = []
    
    if args.apply_random_forward_to_layers == 'all':
        # 所有 transformer 层
        random_forward_layers = ['h.']  # 匹配所有 h.0, h.1, ... 层
    elif args.apply_random_forward_to_layers == 'attn':
        random_forward_layers = ['attn.c_attn', 'attn.c_proj']
    elif args.apply_random_forward_to_layers == 'mlp':
        random_forward_layers = ['mlp']
    elif args.apply_random_forward_to_layers == 'attn_mlp':
        random_forward_layers = ['attn', 'mlp']
    
    # 检查是否对可训练参数应用
    allow_trainable = (args.weight_frozen == 0 or 
                        args.apply_random_forward_to_layers in ['attn', 'mlp', 'attn_mlp'])
    
    # 设置 random forward manager
    random_forward_manager = setup_random_forward_experiment(
        model=model,
        target_layer_names=random_forward_layers,
        strategy=args.random_forward_strategy,  # 'full_random'
        noise_scale=args.forward_noise_scale,
        projection_type=args.forward_projection_type,
        projection_rank=args.forward_projection_rank,
        resample_every_batch=args.forward_resample_every_batch,
        random_seed=args.seed,
        device=args.device,
        allow_trainable=allow_trainable,
    )
    
    print(f"\n✓ Random Forward Manager initialized")
    print(f"  Strategy: {args.random_forward_strategy}")
    print(f"  Apply to layers: {args.apply_random_forward_to_layers}")
    print(f"  Allow trainable: {allow_trainable}")
    
    # 记录到 wandb
    wandb.log({
        "random_forward_strategy": args.random_forward_strategy,
        "forward_noise_scale": args.forward_noise_scale,
        "forward_projection_type": args.forward_projection_type,
        "forward_projection_rank": args.forward_projection_rank,
        "forward_resample_every_batch": args.forward_resample_every_batch,
        "apply_forward_to_layers": args.apply_random_forward_to_layers,
        "forward_allow_trainable": allow_trainable,
    })
else:
    random_forward_manager = None


if args.use_langevin_baseline:
    
    langevin_manager = setup_langevin_baseline(
        model=model,
        noise_scale=args.langevin_noise_scale,
        apply_to_layers=args.langevin_apply_to,
        temperature=args.langevin_temperature,
        use_preconditioner=args.langevin_precond,
        random_seed=args.seed,
        device=args.device
    )

# 步骤7: 创建优化器（支持QKV细粒度控制）
print(f"\n[Step 7] Creating optimizer...")
optimizer = create_optimizer_with_qkv_control(
    model,
    layers_to_freeze,
    qkv_components_to_freeze,
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
print(f"✓ Optimizer created")

print("\n" + "="*70)
print("MODEL INITIALIZATION PIPELINE COMPLETED")
print("="*70 + "\n")

# ============ Embedding层交换与混合 ============

if args.embedding_swap:
    embA_path, embB_path = args.embedding_swap.split(',')
    embA = torch.load(embA_path)
    embB = torch.load(embB_path)
    mix_ratio = args.embedding_mix
    model.transformer.wte.weight.data = mix_ratio * embA['weight'] + (1 - mix_ratio) * embB['weight']

# ============ 层消融 ============
if args.layer_ablate >= 0:
    def ablate_layer(module, input, output):
        return torch.zeros_like(output)
    handle = model.transformer.h[args.layer_ablate].register_forward_hook(ablate_layer)

# ============ 层patching ============
if args.layer_patch >= 0:
    # 假设有另一个模型modelB
    modelB = GPT2LMHeadModel.from_pretrained(args.model_size)
    activations = {}
    def save_activation(module, input, output):
        activations['out'] = output.detach()
    handleA = modelB.transformer.h[args.layer_patch].register_forward_hook(save_activation)
    # 先用modelB forward一次，保存激活
    # modelB(input_ids) # 需要实际输入
    handleA.remove()
    class PatchModule(nn.Module):
        def forward(self, input):
            return activations['out']
    model.transformer.h[args.layer_patch] = PatchModule()

# PCA可视化
if args.pca_layer >= 0:
    # 收集激活
    all_acts = []
    # for batch in dataloader: # 需要实际数据
    #     outputs = model.transformer.h[args.pca_layer](...)
    #     all_acts.append(outputs.detach().cpu().numpy())
    # acts = np.concatenate(all_acts, axis=0)
    # pca = PCA(n_components=2)
    # acts_2d = pca.fit_transform(acts.reshape(acts.shape[0], -1))
    # plt.scatter(acts_2d[:,0], acts_2d[:,1])
    # plt.title(f'Layer {args.pca_layer} Activation PCA')
    # plt.show()
    pass

# 保存embedding
def save_embedding(model, path):
    torch.save({'weight': model.transformer.wte.weight.data.cpu()}, path)

# 加载embedding
def load_embedding(model, path):
    emb = torch.load(path)
    model.transformer.wte.weight.data = emb['weight']

def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

# ============ 准备训练数据 ============
print(f"Loading training data (max_samples={args.train_samples})...")
# raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")wikitext-103-raw-v1
raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )

block_size = min(args.max_length, tokenizer.model_max_length)

lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )
print(lm_datasets)
# train_texts = load_wikitext2_data(split='train', max_samples=args.train_samples)
# val_texts = load_wikitext2_data(split='validation', max_samples=4000)
# # 修改为LAMBADA:
# lambada_train_data = load_lambada_data(split='train', max_samples=args.train_samples)
# # 将LAMBADA数据格式转换为文本列表
# train_texts = [f"{item['context']} {item['target']}" for item in lambada_train_data]

# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, tokenizer, max_length=512):
#         self.encodings = []
#         for text in tqdm(texts, desc="Tokenizing", leave=False):
#             enc = tokenizer(text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
            
#             input_ids = enc['input_ids'].squeeze()
#             attention_mask = enc['attention_mask'].squeeze()

#             labels = input_ids.clone()
#             labels[attention_mask == 0] = -100   # ← 屏蔽掉padding位置

#             self.encodings.append({
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'labels': labels
#             })
    
#     def __len__(self):
#         return len(self.encodings)
    
#     def __getitem__(self, idx):
#         return self.encodings[idx]

# def tokenize_dataset(dataset, tokenizer):

#     #    """pre-tokenize the dataset before training; only collate during training"""
#     # TODO: update this to args.
#     dataset_text_field = "text"
#     def tokenize(element):
#         outputs = tokenizer(
#             element[dataset_text_field],
#             padding="max_length", # False,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt",
#         )
#         labels = torch.tensor(outputs["input_ids"])
#         # Ignore loss on pad tokens.
#         labels[outputs["input_ids"] == tokenizer.pad_token_id] = -100
#         model_inputs = {
#             "input_ids": outputs["input_ids"],
#             "labels": labels
#         }

#         return model_inputs
    
#     dataset = dataset.map(
#         tokenize,
#         batched=True,
#         remove_columns=dataset.column_names,
#         # num_proc=1, # training_args.dataset_num_proc,
#     )

#     return dataset

if not args.skip_training and args.weight_frozen != -1:
    # train_dataset = TextDataset(train_texts, tokenizer)
    # val_dataset = TextDataset(val_texts, tokenizer)
    # train_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split='train')
    # val_data = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split='test')
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
else:
    train_dataset = None
    print("Skipping training dataset creation (zero-shot mode)")

# ============ 训练参数 ============
output_dir = f"{args.output_dir}/{run_name}"
os.makedirs(output_dir, exist_ok=True)

# ============ 训练前评估 (Baseline) ============
print("\n" + "="*70)
print("BEFORE TRAINING - Zero-shot Performance")
print("="*70)
results_before = run_all_evaluations(model, tokenizer, args.device, args, phase="before_training")
 

# 添加一个callback来从eval_loss计算perplexity:
class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在评估后计算perplexity"""
        if metrics and 'eval_loss' in metrics:
            try:
                perplexity = np.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["eval_perplexity"] = perplexity
            logger.warning(f"Evaluation metrics: {metrics}")
            print(f"Evaluation metrics: {metrics}")
        return control

def _add_model_perplexity(metrics):
    try:        
        perplexity = np.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["eval_perplexity"] = perplexity

# ============ 训练 ============
if args.skip_training or args.weight_frozen == -1:
    print("\n" + "="*70)
    print("Skipping training (zero-shot evaluation only)")
    print("="*70)
    results_after = results_before
else:
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,  # 设置总训练步数
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_total_limit=22,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_dir=f'{output_dir}/logs',
        report_to=['wandb'],
        eval_strategy="steps",           # 按步数评估（而非epoch）
        eval_steps=2000,                  # 每100步评估一次

        load_best_model_at_end=True,          # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",    # 用 eval_loss 作为评估指标
        greater_is_better=False,

        save_safetensors= False if args.share_attention_weights or args.share_mlp_weights else True 
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    callbacks = []

    # 1. 原有的EvalCallback（计算perplexity）
    eval_callback = EvalCallback()
    callbacks.append(eval_callback)

    # 如果设置了random backprop并且需要在训练中途关闭
    if random_bp_manager is not None and args.disable_random_bp_at_ratio < 1.0:
        disable_callback = DisableRandomBackpropCallback(
            random_bp_manager=random_bp_manager,
            disable_ratio=args.disable_random_bp_at_ratio,
            max_steps=args.max_steps
        )
        callbacks.append(disable_callback)

        wandb.log({
            'disable_random_bp_at_ratio': args.disable_random_bp_at_ratio,
            'disable_random_bp_at_step': int(args.max_steps * args.disable_random_bp_at_ratio),
        })

    if random_forward_manager is not None and args.disable_random_forward_at_ratio < 1.0:
        rf_callback = DisableRandomForwardCallback(
            random_forward_manager=random_forward_manager,
            disable_ratio=args.disable_random_forward_at_ratio,
            max_steps=args.max_steps
        )
        callbacks.append(rf_callback)


    if langevin_manager and args.disable_langevin_at_ratio < 1.0:
        disable_callback = DisableLangevinCallback(
            langevin_manager, 
            args.disable_langevin_at_ratio, 
            args.max_steps
        )
        callbacks.append(disable_callback)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        eval_dataset=val_dataset,
        callbacks=callbacks,
        compute_metrics=None,
    )
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    trainer.train()
    metrics = trainer.evaluate()
    _add_model_perplexity(metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    # ============ 训练后评估 ============
    print("\n" + "="*70)
    print("AFTER TRAINING - Final Performance")
    print("="*70)
    results_after = run_all_evaluations(model, tokenizer, args.device, args, phase="after_training")
    
    # 保存模型
    trainer.save_model(f"{output_dir}/final_model")
    print(f"✓ Model saved to {output_dir}/final_model") 

# ============ 保存结果 ============
all_results = {
    'experiment_name': run_name,
    'freeze_strategy': freeze_strategy_names.get(args.weight_frozen, 'custom'),
    'weight_frozen': args.weight_frozen,
    'qkv_init_strategy': args.qkv_identity_init,
    'qkv_init_alpha': args.identity_alpha if args.qkv_identity_init == 'mixed' else None,
    'qkv_init_beta': args.identity_beta if args.qkv_identity_init == 'mixed' else None,
    'num_trainable': num_trainable,
    'num_frozen': num_frozen,
    'trainable_ratio': num_trainable / (num_trainable + num_frozen),
    'before_training': results_before,
    'after_training': results_after,
    'config': vars(args)
}

with open(f"{output_dir}/evaluation_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Results saved to {output_dir}/evaluation_results.json")

# ============ 性能对比 ============
print("\n" + "="*70)
print("Performance Comparison")
print("="*70)
print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
print("-"*70)

for metric in results_before.keys():
    before_val = results_before.get(metric, 0)
    after_val = results_after.get(metric, 0)
    
    if 'ppl' in metric.lower():
        change = after_val - before_val
        change_str = f"{change:+.2f}"
    else:
        change = (after_val - before_val) * 100
        change_str = f"{change:+.2f}%"
    
    print(f"{metric:<30} {before_val:<15.4f} {after_val:<15.4f} {change_str:<15}")

print("="*70)

wandb.finish()
print("\n✓ All done!")
