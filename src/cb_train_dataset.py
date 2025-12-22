from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
random.seed(0)

class CircuitBreakerDataset(Dataset):
    
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                lorra_args,
                model_name_or_path,
                template=None
                ):
        super(CircuitBreakerDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024

        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        use_refusal_retain = False
        user_tag, assistant_tag = None, None

        if 'llama-3' in self.model_name_or_path or 'finance' in self.model_name_or_path.lower():#"MonteXiaofeng/Finance-llama3_1_8B_instruct"
            print("USING LLAMA TEMPLATE")
            user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            switch_select = [0, 1]
            use_refusal_retain = True
        elif 'mistral' in self.model_name_or_path:
            print("USING MISTRAL TEMPLATE")
            # fix spacing issue in template
            tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            user_tag="[INST] "
            assistant_tag=" [/INST]"
            sep_token = " "
        elif 'R1' in self.model_name_or_path or 'r1' in self.model_name_or_path:
            print("USING R1 TEMPLATE")
            user_tag="<｜begin_of_sentence｜><｜User｜>"
            assistant_tag="<｜Assistant｜><think>\n"
        elif 'qwq' in self.model_name_or_path.lower():
            print("USING R1 TEMPLATE")
            user_tag="<|im_start|>user\n"
            assistant_tag="<|im_end|>\n<|im_start|>assistant\n<think>\n"
        elif 'qwq' in self.model_name_or_path.lower():
            print("USING R1 TEMPLATE")
            user_tag="<|im_start|>user\n"
            assistant_tag="<|im_end|>\n<|im_start|>assistant\n<think>\n"
        elif 'openthinker' in self.model_name_or_path:
            print("USING openthinker TEMPLATE")
            user_tag="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            assistant_tag="<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")
        
        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token
        # if template is not None:
        #     if 'R1_style' in template:
        #         user_tag="<|begin_of_sentence|><｜User｜>"
        #         assistant_tag="<|Assistant|>"

        #         user_tag_LA3 = "<|start_header_id|>user<|end_header_id|>\n\n"
        #         assistant_tag_LA3 ="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        #     else:
        #         raise NotImplementedError(f"Config {template} not found")
        # else:
        # ======================= Retain ======================= #
        # ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        ds = load_dataset("GAIR/LIMO", split="train")
        orig_s = []
        suffix = ""
        for idx in range(len(ds)):
            # messages = example["messages"]
            # if len(messages) < 2: continue

            # formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")

            # switch = np.random.choice(switch_select)
            # if switch == 0:
            #     formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            # elif switch == 1:
            #     formatted_input = one_shot_template.format(
            #         user_tag=user_tag, assistant_tag=assistant_tag,
            #         instruction="", response=messages[1]["content"])
            question = f"{ds[idx]['question']} {suffix}"
            try:
                thought_chain = ds[idx]['solution'].split(f"**Final Answer**")[0]
                anwser = ds[i]['solution'].split(f"**Final Answer**")[1]
                thinking_trajectories = f"<think>{thought_chain}</think>\n"
                final_answer = r"The final answer is "+ anwser +r""
                messages= [{'content':question, 'role':'user'}, {'content':f"{thinking_trajectories} {final_answer}", 'role':'assistant'}]
            except:
                try:
                    thought_chain = ds[idx]['solution'].split(f"Final Answer")[0]
                    anwser = ds[idx]['solution'].split(f"**Final Answer**")[1]
                    thinking_trajectories = f"<think>{thought_chain}</think>\n"
                    final_answer = r"The final answer is "+ anwser +r""
                    messages= [{'content':question, 'role':'user'}, {'content':f"{thinking_trajectories} {final_answer}", 'role':'assistant'}]
                except:
                    print(ds[idx]['solution'])
                    continue
            try:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            except:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)
            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        # print("orig_s_retain[0]", orig_s[0])
        # print("Orig s length:", len(self.orig_s_retain))

        # ======================= Borderline Retain ======================= #
        # from https://github.com/paul-rottger/exaggerated-safety
        with open(f'data/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row['final_label'] == "1_full_compliance"]
        
        borderline_orig_s = []
        for i, d in enumerate(data * 50):
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response="</think>"+d['completion'])
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response="</think>"+d['completion'])
            
            borderline_orig_s.append(formatted_input)

        self.orig_s_retain += borderline_orig_s
        random.shuffle(self.orig_s_retain)
        # print("borderline_orig_s[0]", borderline_orig_s[0])
        # print("Orig s length:", len(self.orig_s_retain))

        # ======================= Refusal Retain ======================= #
        if use_refusal_retain:
            with open("data/circuit_breakers_train.json") as file:
                dataset = json.load(file)

            random.shuffle(dataset)
            dataset = dataset[:2000]
            refusal_retain_orig = []
            for i, d in tqdm(enumerate(dataset*2)):
                switch = np.random.choice(switch_select)
                if switch == 0:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction=d['prompt'], response="</think>"+d['llama3_output'])
                elif switch == 1:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction=d['prompt'], response="</think>"+d['llama3_output'])
                
                refusal_retain_orig.append(formatted_input)

            self.orig_s_retain += refusal_retain_orig
            random.shuffle(self.orig_s_retain)
            # print("refusal_orig_s[0]", refusal_retain_orig[0])
            # print("Orig s length:", len(self.orig_s_retain))

        # ======================= Circuit Breaker ======================= #
        with open("data/circuit_breakers_train.json") as file:
            dataset = json.load(file)
        circuit_breaker_orig = []
        circuit_breaker_safe = []
        for i, d in tqdm(enumerate(dataset)):
            # cb_output = d['output']### The original one is output
            cb_output = d['llama3_output']### The original one is output
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response="")
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response="")

            formatted_input_safe = one_shot_template.format(
                    user_tag=user_tag, assistant_tag="Remember you should not generate any harmful or misleading content, you should directly reject harmful requests."+assistant_tag,
                    instruction=d['prompt'], response="")
            
            circuit_breaker_orig.append(formatted_input)
            circuit_breaker_safe.append(formatted_input_safe)
        self.circuit_breaker_orig = circuit_breaker_orig
        self.circuit_breaker_safe = circuit_breaker_safe
        # random.shuffle(self.circuit_breaker_orig)### Original One has shuffle
        # print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        # print("Short circuit length:", len(self.circuit_breaker_orig))

        # ======================= Val ======================= #
        with open("data/circuit_breakers_val.json") as file:
            dataset = json.load(file)
        val_orig = []
        for i, d in tqdm(enumerate(dataset)):
            val_orig.append(one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['prompt'], response=d['llama3_output']))

        self.val_orig = val_orig
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        circuit_breaker_orig = self.circuit_breaker_orig[i]
        circuit_breaker_safe = self.circuit_breaker_safe[i]
        val_orig = self.val_orig[i % len(self.val_orig)]

        cb_tokenized_kwargs = dict(max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        tokenize_kwargs = dict(max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        # =========== Circuit Breaker Inputs ===========
        # === split to [request, response] shape [512,512] to support different mask configs ===
        # cb_request, cb_response = circuit_breaker_orig.split('<SEPARATOR>')
        # cb_safereq, cb_saferesp = circuit_breaker_safe.split('<SEPARATOR>')
        
        # self.tokenizer.padding_side = "left"
        # tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        # tokenized_request_cb_safe = self.tokenizer(cb_safereq, **cb_tokenized_kwargs)
        # self.tokenizer.padding_side = "right"
        # response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        # response_tokenized_cb_safe = self.tokenizer(cb_saferesp, add_special_tokens=False, **cb_tokenized_kwargs)
        # self.tokenizer.padding_side = "left"


        combined_circuit_breaker = self.tokenizer(circuit_breaker_orig.replace('<SEPARATOR>', self.sep_token), **cb_tokenized_kwargs)
        combined_input_ids_circuit_breaker = combined_circuit_breaker["input_ids"]
        combined_attention_mask_circuit_breaker = combined_circuit_breaker["attention_mask"]

        safe_combined_circuit_breaker= self.tokenizer(circuit_breaker_safe.replace('<SEPARATOR>', self.sep_token), **cb_tokenized_kwargs)
        safe_combined_input_ids_circuit_breaker=safe_combined_circuit_breaker["input_ids"]
        safe_combined_attention_mask_circuit_breaker= safe_combined_circuit_breaker["attention_mask"]
        # print(safe_combined_input_ids_circuit_breaker)

        # combined_input_ids_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
        # combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]], dim=1)
        # print(combined_input_ids_circuit_breaker)
        # safe_combined_input_ids_circuit_breaker = torch.cat([tokenized_request_cb_safe["input_ids"], response_tokenized_cb_safe["input_ids"]], dim=1)
        # safe_combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_cb_safe["attention_mask"], response_tokenized_cb_safe["attention_mask"]], dim=1)

        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(orig_s_retain.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
        
        # =========== Val Inputs ===========
        tokenized_inputs_val = self.tokenizer(val_orig.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids_cbsafe = safe_combined_input_ids_circuit_breaker,
            attention_mask_cbsafe = safe_combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"],
            attention_mask=tokenized_inputs_retain["attention_mask"],
            input_ids_val=tokenized_inputs_val["input_ids"],
            attention_mask_val=tokenized_inputs_val["attention_mask"],
        )
