import argparse
import gc
import glob
import os
import random
import json
import numpy as np
import torch
import torch.nn
import wandb
import math
from pipeline.train.data import get_data
from pipeline.train.distributed import init_distributed_device, world_info_from_env
from pipeline.eval.rec_metrics import ndcg_at_k, hit_at_k, mrr_at_k, dcg_at_k
from pipeline.eval.utils import NumpyEncoder
import torch.distributed as dist
from pipeline.mm_utils.rec_dataset import RecDataset
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from transformers.generation.configuration_utils import GenerationConfig

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    CLIPImageProcessor,
)

from pipeline.train.train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
)
from tqdm import tqdm
import time

with open("../data/processed_filter_8_all/meta_all.json") as f:
            meta_data = json.load(f)

def extract_meta1( index):
    max_length = 20
    #sample = meta_data[str(index)]
    sample = meta_data.get(str(index))  # Use get to avoid KeyError
    if sample is None:
        category = brand = title = price = "Unknown"
        #return None  # Return None if no corresponding data is found
    else:
        category = "Unknown" if sample["category"] == "" else sample["category"]
        category = " ".join(category.split()[:max_length])
        brand = "Unknown" if sample["brand"] == "" else sample["brand"]
        brand = " ".join(brand.split()[:max_length])
        title = "Unknown" if sample["title"] == "" else sample["title"]
        title = " ".join(title.split()[:max_length])
        price = "Unknown" if sample["price"] == "" else sample["price"]

        text = f"Category {category} Price {price} Brand {brand} Title {title}"
        return text


def eval_model_rec(
    args,
    model,
    epoch,
    multi_instruct_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    accelerator,
    wandb,
):
    # num_batches_per_epoch = multi_instruct_loader.num_batches
    num_batches_per_epoch = len(multi_instruct_loader)

    total_training_steps = num_batches_per_epoch

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)


    model.eval()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    end = time.time()
    hr, ndcg, mrr=[], [], []
    hr_5, ndcg_5, mrr_5=[], [], []
    hr_3, ndcg_3, mrr_3=[], [], []
    local_hr_3, local_ndcg_3, local_mrr_3=[], [], []
    local_hr_5,local_ndcg_5,local_mrr_5=[], [], []
    local_hr,local_ndcg,local_mrr=[], [], []
    K=1
    ndcg_mean_3_rank0 = float('-inf')

    results = []

    # loop through dataloader
    for num_steps, (batch_multi_instruct) in tqdm(
        enumerate(multi_instruct_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MULTI_INSTRUCT FORWARD PASS ####

        images = (
            batch_multi_instruct["net_input"]["patch_images"].unsqueeze(2)
        )
        input_ids = batch_multi_instruct["net_input"]["input_ids"]
        attention_mask = batch_multi_instruct["net_input"]["attention_masks"]
        #print("input_ids",input_ids)
        print("0_input_ids.shape",input_ids.shape,"0_attention_mask.shape",attention_mask.shape)
        #input_seq=batch_multi_instruct["net_input"]["input_seq_old"]
        # just for seq rec
        #input_length = batch_multi_instruct["net_input"]["input_len"][0]
        output_ids = batch_multi_instruct["net_output"]["output_ids"][0]

        #id_length = input_ids[0].shape[0]

        labels = batch_multi_instruct["net_output"]["labels"]

        with torch.no_grad():
            with autocast():


                # output = model.generate(
                #     vision_x=images,
                #     lang_x=input_ids,
                #     attention_mask=attention_mask,
                #     num_beams=K,
                #     num_return_sequences=K,
                #     early_stopping=True,
                #     max_new_tokens=70,
                #     eos_token_id=args.tokenizer.eos_token_id,
                #     pad_token_id=args.tokenizer.eos_token_id,
                #     output_attentions=True
                # )
                # print(f"key(output){dict(output)}")
                # texts = tokenizer.batch_decode(output, skip_special_tokens=True)
                # attention_map=output.attentions
                # #print("dir(outputs)",dir(outputs))

                outputs = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    num_beams=K,
                    num_return_sequences=K,
                    early_stopping=True,
                    max_new_tokens=100,
                    eos_token_id=args.tokenizer.eos_token_id,
                    pad_token_id=args.tokenizer.eos_token_id,
                    output_attentions=True,  # 输出注意力
                    return_dict_in_generate=True,  # 返回字典形式
                    output_scores=True,  # 同时获取生成概率
                    output_hidden_states=True,  # 获取隐藏状态
                )
                #print(f"outputs{outputs}")
                output_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

                #print(f"all_layer_attention.shape: {all_layer_attention.shape}")
                #print(f"len(outputs.hidden_states[-1].shape): {(outputs.hidden_states[-1].shape)}")
                #print(f"len(outputs.hidden_states[-1][-1]): {len(outputs.hidden_states[-1][-1])}")
                print(f"outputs.hidden_states[-1].shape: {outputs.hidden_states[-2][-2].shape}")


                def extract_weighted_values(model, outputs, layer_idx):
                    """
                    Extract attention-weighted values before the final linear projection.
                    Args:
                        model: The Flamingo model
                        outputs: Model generation outputs
                        layer_idx: Index of the layer to analyze
                    Returns:
                        weighted_values: The attention-weighted values before final projection
                    """
                    # Get the layer's attention module
                    layer = model.lang_encoder.gpt_neox.layers[layer_idx].decoder_layer
                    print(f"model.lang_encoder.gpt_neox.layers{model.lang_encoder.gpt_neox.layers}")
                    attention = layer.attention

                    # Get hidden states for this layer
                    if isinstance(outputs.hidden_states, tuple):
                        hidden_states = outputs.hidden_states[layer_idx][-1]
                    else:
                        hidden_states = outputs.hidden_states[layer_idx]

                    # Get Q,K,V from the combined projection
                    qkv_output = attention.query_key_value(hidden_states)
                    q_size = k_size = v_size = qkv_output.size(-1) // 3
                    q, k, v = qkv_output.chunk(3, dim=-1)

                    # Reshape for attention computation
                    batch_size, seq_length = q.size(0), q.size(1)
                    q = q.view(batch_size, seq_length, q_size)
                    k = k.view(batch_size, seq_length, k_size)
                    v = v.view(batch_size, seq_length, v_size)

                    # Compute attention scores
                    attention_scores = torch.matmul(q, k.transpose(-1, -2))
                    attention_scores = attention_scores / math.sqrt(q_size)

                    # Apply softmax
                    attention_probs = F.softmax(attention_scores, dim=-1)
                    if hasattr(attention, 'attention_dropout'):
                        attention_probs = attention.attention_dropout(attention_probs)

                    # Get weighted values (this is what we want - before the final dense projection)
                    weighted_values = torch.matmul(attention_probs, v)

                    return weighted_values, attention_probs, v

                def analyze_attention_patterns(model, outputs, layer_idx):
                    """
                    Analyze attention patterns at a specific layer.
                    """
                    weighted_vals, probs, values = extract_weighted_values(model, outputs, layer_idx)

                    print(f"Layer {layer_idx} Analysis:")
                    print(f"Weighted values shape: {weighted_vals.shape}")
                    print(f"Attention probs shape: {probs.shape}")
                    print(f"Values shape: {values.shape}")

                    # Save intermediate representations
                    save_dir = f"./result_full_eval/attention_analysis/layer_{layer_idx}"
                    os.makedirs(save_dir, exist_ok=True)

                    # Save tensors
                    torch.save(weighted_vals, f"{save_dir}/weighted_values.pt")
                    torch.save(probs, f"{save_dir}/attention_probs.pt")
                    torch.save(values, f"{save_dir}/values.pt")

                    return weighted_vals, probs, values

                # For a specific layer
                layer_idx = 0  # or any layer you want to analyze
                weighted_values, attention_probs, values = analyze_attention_patterns(model, outputs, layer_idx)

                # To analyze all layers
                results = []
                for layer_idx in range(32):  # Flamingo has 32 layers
                    result = analyze_attention_patterns(model, outputs, layer_idx)
                    results.append(result)


                # 每处理 100 个批次，清理一次内存
                if num_steps % 2 == 0:
                    gc.collect()  # 清理 Python 垃圾回收

                # if args.rank == 0:
                #     print("labels",labels,"labels.shape",labels.shape)
                #     print("lm_logits",lm_logits,"lm_logits.shape",lm_logits.shape)
                #logits=outputs['logits'][0, -1]
                #logits=outputs_new['logits'][0, -1]
                #texts=tokenizer.decode(item_list[logits[item_list].argsort(dim=-1, descending=True)[:10]])
                #print("item_list.shape",item_list.shape)
                #texts = texts.split("item_")
                #texts = ["item_" + item for item in texts if item]
                #print("texts",texts)

    #             combined_texts = []

    #             for text in texts:
    #                 # Extract the item index from the text
    #                 try:
    #                     item_index = int(text.split("item_")[-1].split()[0])
    #                 except (ValueError, IndexError):
    #                     item_index = None

    #                 if item_index is not None:
    #                     meta_item = extract_meta1(item_index)  # Call the method on the instance
    #                 else:
    #                     meta_item = "Category Unknown Price Unknown Brand Unknown Title Unknown"  # Default value if item_index extraction fails

    #                 combined_text = f"{text} {meta_item}"
    #                 combined_texts.append(combined_text)

    #                 item_target = output_ids
    #                 # for text in item_target:
    #                 #     # Extract the item index from the text
    #                 #     try:
    #                 item_index_target = item_target.split("_")[1]
    #                     # except (ValueError, IndexError):
    #                     #     item_index_target = None

    #                 if item_index is not None:
    #                     meta_item_target = extract_meta1(item_index_target)  # Call the method on the instance
    #                 else:
    #                     meta_item_target = "Category Unknown Price Unknown Brand Unknown Title Unknown"  # Default value if item_index extraction fails

    #                 true_target_details = f"{item_target} {meta_item_target}"

    #                 # 删除不再使用的变量
    #             del input_ids, attention_mask, images
    #             gc.collect()  # 强制垃圾回收，释放内存

    # #     # naive id
    #     item_target = output_ids
    # # 每处理 100 个批次，清理一次内存
    #     if num_steps % 2 == 0:
    #         gc.collect()  # 清理 Python 垃圾回收

    #     # if num_steps == 0 and args.rank == 0:
    #     #     print("texts:", texts)
    #     #     print("item_target", item_target)
    #     #     print("item_target_type",type(item_target))

    #     gen_ids = [text == item_target for text in texts]
    #     r = np.array([0] * 10)
    #     r_ = np.array(gen_ids, dtype=int)
    #     r[:len(r_)] = r_

    #     # Function to extract meta information
    #     with open("/home/code/UniMP/data/processed_filter_8_all/meta_all.json") as f:
    #         meta_data = json.load(f)

    #     if args.rank == 0:
    #         user_dcg_5 = dcg_at_k(r, 10)
    #         #answers = [f'item_{item_id}' for item_id in texts.split('item_')[1:]]
    #         # print(
    #         #     "input_seq", batch_multi_instruct["net_input"]["input_seq"],
    #         #     #"input_ids",batch_multi_instruct["net_input"]["input_ids"],
    #         #     #"meta_item": batch_multi_instruct["net_input"]["meta_item"],
    #         #     #"item": batch_multi_instruct["net_input"]["item"].tolist() if isinstance(batch_multi_instruct["net_input"]["item"], torch.Tensor) else batch_multi_instruct["net_input"]["item"],
    #         #     "true item_target", output_ids,
    #         #     #"outputs",outputs,
    #         #     #"target_item_detail",target_combined_texts,
    #         #     "answer items", texts,
    #         #     #"answer items detail",combined_texts,
    #         #     #"r", r.tolist(),
    #         #     #"r_": r_.tolist(),
    #         #     "user_dcg_5", float(user_dcg_5))

    #         results.append({
    #             #"input_seq_old": batch_multi_instruct["net_input"]["input_seq_old"],
    #             #"input_seq_new": input_seq_new,
    #             #"vocab.items()":vocab.items(),
    #             #"input_ids":batch_multi_instruct["net_input"]["input_ids"].tolist(),
    #             #"meta_item": batch_multi_instruct["net_input"]["meta_item"],
    #             #"item": batch_multi_instruct["net_input"]["item"].tolist() if isinstance(batch_multi_instruct["net_input"]["item"], torch.Tensor) else batch_multi_instruct["net_input"]["item"],
    #             #"true item_target": item_target,
    #             "true_target_details":true_target_details,
    #             #"outputs":outputs,
    #             #"target_item_detail":target_combined_texts,
    #             "answer items": texts,
    #             "answer items detail":combined_texts,
    #             "r": r.tolist(),
    #             #"r_": r_.tolist(),
    #             #"top_10_texts":top_10_texts,
    #             #"sequence_scores":sequence_scores,
    #             "user_dcg_5": float(user_dcg_5)
    #         })

    #     user_mrr_3 = mrr_at_k(r, 3)
    #     user_hr_3 = hit_at_k(r, 3)
    #     user_ndcg_3 = ndcg_at_k(r, 3, 1)
    #     hr_3.append(user_hr_3)
    #     ndcg_3.append(user_ndcg_3)
    #     mrr_3.append(user_mrr_3)

    #     user_mrr_5 = mrr_at_k(r, 5)
    #     user_hr_5 = hit_at_k(r, 5)
    #     user_ndcg_5 = ndcg_at_k(r, 5, 1)
    #     hr_5.append(user_hr_5)
    #     ndcg_5.append(user_ndcg_5)
    #     mrr_5.append(user_mrr_5)

    #     user_mrr = mrr_at_k(r, K)
    #     user_hr = hit_at_k(r, K)
    #     user_ndcg = ndcg_at_k(r, K, 1)
    #     hr.append(user_hr)
    #     ndcg.append(user_ndcg)
    #     mrr.append(user_mrr)
    #     step_time_m.update(time.time() - end)
    #     end = time.time()

    #     # 每处理 100 个批次，清理一次内存
    #     if num_steps % 10 == 0:
    #         gc.collect()  # 清理 P

    # # hr_mean_3, ndcg_mean_3, mrr_mean_3 = np.mean(hr_3), np.mean(ndcg_3), np.mean(mrr_3)
    # # hr_mean_5, ndcg_mean_5, mrr_mean_5 = np.mean(hr_5), np.mean(ndcg_5), np.mean(mrr_5)
    # # hr_mean, ndcg_mean, mrr_mean = np.mean(hr), np.mean(ndcg), np.mean(mrr)

    # # ndcg_mean_3_rank0 = float('-inf')
    # # 将局部r添加到全局列表
    #     local_hr_3.append(user_hr_3)
    #     local_ndcg_3.append(user_ndcg_3)
    #     local_mrr_3.append(user_mrr_3)
    #     local_hr_5.append(user_hr_5)
    #     local_ndcg_5.append(user_ndcg_5)
    #     local_mrr_5.append(user_mrr_5)
    #     local_hr.append(user_hr)
    #     local_ndcg.append(user_ndcg)
    #     local_mrr.append(user_mrr)

    # # 使用all_gather收集所有GPU上的结果
    # hr_3_list = [None for _ in range(dist.get_world_size())]
    # ndcg_3_list = [None for _ in range(dist.get_world_size())]
    # mrr_3_list = [None for _ in range(dist.get_world_size())]

    # dist.all_gather_object(hr_3_list, hr_3)
    # dist.all_gather_object(ndcg_3_list, ndcg_3)
    # dist.all_gather_object(mrr_3_list, mrr_3)

    # hr_5_list = [None for _ in range(dist.get_world_size())]
    # ndcg_5_list = [None for _ in range(dist.get_world_size())]
    # mrr_5_list = [None for _ in range(dist.get_world_size())]

    # dist.all_gather_object(hr_5_list, hr_5)
    # dist.all_gather_object(ndcg_5_list, ndcg_5)
    # dist.all_gather_object(mrr_5_list, mrr_5)

    # hr_list = [None for _ in range(dist.get_world_size())]
    # ndcg_list = [None for _ in range(dist.get_world_size())]
    # mrr_list = [None for _ in range(dist.get_world_size())]

    # dist.all_gather_object(hr_list, hr)
    # dist.all_gather_object(ndcg_list, ndcg)
    # dist.all_gather_object(mrr_list, mrr)

    # # # 使用all_gather_object收集所有GPU上的results
    # # gathered_results = [None for _ in range(dist.get_world_size())]
    # # dist.all_gather_object(gathered_results, results)
    # gathered_results = None
    # if args.rank == 0:
    #     gathered_results = [None for _ in range(dist.get_world_size())]

    # # 使用 gather_object 而不是 all_gather_object
    # dist.gather_object(results, gathered_results, dst=0)

    # # 在主进程中计算全局平均
    # if args.rank == 0:
    #     all_results = []
    #     for result in gathered_results:
    #         all_results.extend(result)

    #     all_hr_3 = [item for sublist in hr_3_list for item in sublist]
    #     all_ndcg_3 = [item for sublist in ndcg_3_list for item in sublist]
    #     all_mrr_3 = [item for sublist in mrr_3_list for item in sublist]

    #     hr_mean_3 = np.mean(all_hr_3)
    #     ndcg_mean_3 = np.mean(all_ndcg_3)
    #     mrr_mean_3 = np.mean(all_mrr_3)

    #     all_hr_5 = [item for sublist in hr_5_list for item in sublist]
    #     all_ndcg_5 = [item for sublist in ndcg_5_list for item in sublist]
    #     all_mrr_5 = [item for sublist in mrr_5_list for item in sublist]

    #     hr_mean_5 = np.mean(all_hr_5)
    #     ndcg_mean_5 = np.mean(all_ndcg_5)
    #     mrr_mean_5 = np.mean(all_mrr_5)

    #     all_hr = [item for sublist in hr_list for item in sublist]
    #     all_ndcg = [item for sublist in ndcg_list for item in sublist]
    #     all_mrr = [item for sublist in mrr_list for item in sublist]

    #     hr_mean = np.mean(all_hr)
    #     ndcg_mean = np.mean(all_ndcg)
    #     mrr_mean = np.mean(all_mrr)

    #     gathered_results=all_results


    #     all_results.sort(key=lambda x: x["user_dcg_5"], reverse=True)
    #     with open(f"result_full_eval/heatmap_singlegpu_epoch_{epoch}HR@3: {hr_mean_3}.json", "w") as f:
    #         json.dump(all_results, f, cls=NumpyEncoder, indent=4)

    #     ndcg_mean_3_rank0 = ndcg_mean_3
    #     print("epoch: ", epoch)
    #     print(f"HR@3: {hr_mean_3} NDCG@3: {ndcg_mean_3} MRR@3: {mrr_mean_3} HR@5: {hr_mean_5} NDCG@5: {ndcg_mean_5} MRR@5: {mrr_mean_5} HR@10: {hr_mean} NDCG@10: {ndcg_mean} MRR@10: {mrr_mean}")

    # return ndcg_mean_3_rank0
