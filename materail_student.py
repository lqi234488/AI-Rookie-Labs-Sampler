import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import time

def Softmax( logits:torch.Tensor, temperature:float=0.7 ):
    """
        input) logits.shape=(batch size, vocab size)
        output) probs.shape=(batch size, vocab size)
    """
    #print( f"[INFO][SOFTMAX] using SOFTMAX" )
    # 避免 temperature 為 0 導致除以零
    temp = max(temperature, 1e-8)
    # 進行 Temperature scaling
    logits = logits / temp
    
    # 為了數值穩定性，減去最大值（防止 exp 爆掉）
    max_logit = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logit)
    
    # 計算真正的機率 (sum to 1)
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    return probs

class Sampler:
    
    @staticmethod
    def random( logits:torch.Tensor ):
        """
            input) logits.shape=(batch size, vocab size)
            output) next_token_id=(batch size, 1)
        """

        #print( f"[INFO][SAMPLER] using RANDOM" )

        logits_list = logits.squeeze(0)
        token_id_list = np.arange( len(logits_list) )
        next_token_id = random.choices( token_id_list, weights=logits_list, k=1 )
        next_token_id = torch.tensor( next_token_id, device=device ).unsqueeze(0)

        return next_token_id

    @staticmethod
    def test( strategy:str, logits:torch.Tensor ):

        print( f"[INFO][SAMPLER][TEST] TESTING" )

        match strategy:
            case "random":
                token_id = Sampler.random( logits )

            case _:
                print( "[INFO][SAMPLER][TEST] No such strategy" )

        print( token_id )

class Filter:
    @staticmethod
    def topK(probs:torch.Tensor, logits:torch.Tensor, threshold:float):
        # threshold 這裡代表 K 值
        print(f"[INFO][FILTER] using TOP-K")
        K = int(threshold)
        
        # 找出前 K 大的值
        top_k_values, _ = torch.topk(logits, K)
        # 取得第 K 個（最小的那個）值作為門檻
        min_logit = top_k_values[:, -1].unsqueeze(-1)
        
        # 將小於門檻的 logits 全部設為負無限大
        logits[logits < min_logit] = -float('inf')
        return logits

    @staticmethod
    def topP(probs:torch.Tensor, logits:torch.Tensor, threshold:float):
        # threshold 這裡代表 P 值 (例如 0.9)
        print(f"[INFO][FILTER] using TOP-P")
        
        # 1. 先對機率進行降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 2. 計算累積機率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 3. 找出累積機率超過 threshold 的位置
        # 我們要保留累積機率剛好達到 threshold 的那些 token，移除之後的
        sorted_indices_to_remove = cumulative_probs > threshold
        
        # 技巧：將遮罩向右移一位，確保「剛好超過」的那一個 token 也能被保留
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 4. 將對應的 logits 設為負無限大
        for i in range(logits.shape[0]): # 遍歷 batch
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = -float('inf')
            
        return logits

    @staticmethod
    def minP(probs:torch.Tensor, logits:torch.Tensor, threshold:float):
        print(f"[INFO][FILTER] using MIN-P")
        # 找出該次預測中最高的機率
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        # 門檻值 = 最高機率 * threshold
        min_prob_threshold = max_prob * threshold
        
        # 低於門檻的全部設為負無限大
        logits[probs < min_prob_threshold] = -float('inf')
        return logits

    @staticmethod
    def test( strategy:str, probs:torch.Tensor, logits:torch.Tensor, threshold:float ):
        print( f"[INFO][FILTER][TEST] TESTING" )
        match strategy:
            case "topK":
                logits = Filter.topK( probs, logits, threshold )
            case "topP":
                logits = Filter.topP( probs, logits, threshold )
            case "minP":
                logits = Filter.minP( probs, logits, threshold )
            case _:
                print( "[INFO][FILTER][TEST] No such strategy" )
        print( logits )

class Penalty:
    @staticmethod
    def repetition(logits:torch.Tensor, seen:torch.Tensor, penalty:float):
        # 只要出現過，就直接扣掉 penalty
        print(f"[INFO][PENALTY] using REPETITION")
        logits = logits - (seen > 0).float() * penalty
        return logits

    @staticmethod
    def frequency(logits:torch.Tensor, seen:torch.Tensor, penalty:float):
        # 扣掉 (出現次數 * penalty)，出現越多次扣越多
        print(f"[INFO][PENALTY] using FREQUENCY")
        logits = logits - (seen * penalty)
        return logits

    @staticmethod
    def presence(logits:torch.Tensor, seen:torch.Tensor, penalty:float):
        # 只要目前序列中「存在」該 token 就扣分 (類似 repetition)
        print(f"[INFO][PENALTY] using PRESENCE")
        mask = (seen > 0).float()
        logits = logits - (mask * penalty)
        return logits

    @staticmethod
    def test( strategy:str, logits:torch.Tensor, seen:torch.Tensor, penalty:float ):
        print( f"[INFO][PENALTY][TEST] TESTING" )
        match strategy:
            case "repetition":
                logits = Penalty.repetition( logits, seen, penalty )

            case "presence":
                logits = Penalty.presence( logits, seen, penalty )

            case "frequency":
                logits = Penalty.frequency( logits, seen, penalty )

            case _:
                print( "[INFO][PENALTY][TEST] No such strategy" )
        print( logits )

class Tokenizer:

    @staticmethod
    def tokenize_pipe( tokenizer:AutoTokenizer, prompt:str ):
        """
            output) input_ids.shape=(batch size, token length)
            output) attention_masks.shape=(batch size, token length)
        """

        print( f"[INFO][TOKENIZE] using PIPE" )

        input_dict = tokenizer( prompt, return_tensors="pt", add_special_tokens=False )
        input_ids = input_dict[ "input_ids" ]
        attention_masks = input_dict[ "attention_mask" ]

        return input_ids, attention_masks

    @staticmethod
    def tokenize_step_by_step( tokenizer:AutoTokenizer, prompt:str ):
        """
            output) input_ids.shape=(batch size, token length)
            output) attention_masks.shape=(batch size, token length)
        """

        print( f"[INFO][TOKENIZE] using STEP-BY-STEP" )

        # [EXP] please implement here
        # 1. 將原始文字編碼成 Token ID (這是一個 Python List)
        # add_special_tokens=True 會自動加上 <|begin_of_text|> 等特殊標籤
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        
        # 2. 為了觀察，將 ID 轉換回人類可讀的 Tokens (例如 [' 你', '喜歡', '吃', ...])
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        # 3. 打印出來觀察 (這是 LAB 的核心要求)
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Token length: {len(ids)}")

        # 4. 將 Python List 轉換為 PyTorch Tensor，並加上 Batch 維度 (變成 [1, seq_len])
        # 注意：這裡要確定 device 變數是抓得到的，通常是在外部定義
        input_ids = torch.tensor([ids], device=device)
        
        # 5. 建立 Attention Mask，因為目前只有一筆資料，全部設為 1 (表示全部都要注意)
        attention_masks = torch.ones_like(input_ids)
        for i, t_id in enumerate(ids):
            # 將單個 ID 轉回文字
            print(f"ID {t_id} -> '{tokenizer.decode([t_id])}'")

        return input_ids, attention_masks

    @staticmethod
    def test( strategy:str, tokenizer:AutoTokenizer, prompt:str ):

        print( f"[INFO][TOKENIZER][TEST] TESTING" )

        match strategy:
            case "pipe":
                input_ids, attention_masks = Tokenizer.tokenize_pipe( tokenizer, prompt )

            case "step-by-step":
                input_ids, attention_masks = Tokenizer.tokenize_step_by_step( tokenizer, prompt )

            case _:
                print( "[INFO][TOKENIZER][TEST] No such strategy" )

        print( input_ids )
        print( attention_masks )

class Generator:

    @staticmethod
    def generate_pipe( tokenizer:AutoTokenizer, model:AutoModelForCausalLM, input_ids:torch.Tensor, attention_masks:torch.Tensor, max_token_len=20 ):

        print( f"[INFO][GENERATE] using PIPE" )

        outputs = model.generate(
            input_ids, 
            attention_mask=attention_masks,
            max_new_tokens=max_token_len,  # avoid repeating without eos_token
            pad_token_id=tokenizer.eos_token_id
        )

        return outputs
    
    @staticmethod
    def generate_iterative(tokenizer, model, input_ids, attention_masks, max_token_len=400):
        print(f"\n[INFO][GENERATE] using ITERATIVE (統一參數面板)")
        device = input_ids.device
        
        # ==========================================
        # 🌟 寫作業調整區 🌟
        # ==========================================
        cfg_temperature  = 1.5    # [Temperature] 預設 1.0 (越低越保守，越高越隨機)
        cfg_greedy       = False  # [Decoding] True=Greedy(無), False=Random(抽籤)

        # 這次 Lab 要測試的 Filter (一次請只開一種來觀察)
        cfg_top_k        = 0      # [Filter] 設為 0 代表不使用 (通常設 40~50)
        cfg_top_p        = 0.0    # [Filter] 設為 1.0 代表不使用 (通常設 0.9)
        cfg_min_p        = 0.05    # [Filter] 設為 0.0 代表不使用 (通常設 0.05)

        # 未來 Lab 會用到的 Penalty
        cfg_rep_penalty  = 1.0    # [Penalty] 設為 1.0 代表不使用 (大於 1.0 會處罰)
        cfg_freq_penalty = 2.0    # [Penalty] 設為 0.0 代表不使用
        cfg_pres_penalty = 0.0    # [Penalty] 設為 0.0 代表不使用
        # ==========================================

        seen = torch.zeros(model.config.vocab_size, device=device)

        for _ in range(max_token_len):
            ### 1. 模型推論取得 Logits ###
            outputs = model(input_ids, attention_mask=attention_masks)
            next_token_logits = outputs.logits[:, -1, :]

            ### 2. 套用 Penalty (防跳針) ###
            if cfg_rep_penalty != 1.0:
                next_token_logits = Penalty.repetition(next_token_logits, seen, cfg_rep_penalty)
            if cfg_freq_penalty != 0.0:
                next_token_logits = Penalty.frequency(next_token_logits, seen, cfg_freq_penalty)
            if cfg_pres_penalty != 0.0:
                next_token_logits = Penalty.presence(next_token_logits, seen, cfg_pres_penalty)

            ### 3. 套用 Filter (砍掉雜訊) ###
            # 先算基準機率，供 Filter 參考
            temp_probs = Softmax(next_token_logits, temperature=cfg_temperature)
            
            if cfg_top_k > 0:
                next_token_logits = Filter.topK(temp_probs, next_token_logits, threshold=cfg_top_k)
            if cfg_top_p < 1.0:
                next_token_logits = Filter.topP(temp_probs, next_token_logits, threshold=cfg_top_p)
            if cfg_min_p > 0.0:
                next_token_logits = Filter.minP(temp_probs, next_token_logits, threshold=cfg_min_p)

            ### 4. 重新計算最終機率 ###
            final_probs = Softmax(next_token_logits, temperature=cfg_temperature)

            ### 5. 抽樣 (Sampling) ###
            if cfg_greedy:
                next_token_id = torch.argmax(final_probs, dim=-1, keepdim=True)
            else:
                next_token_id = Sampler.random(final_probs)

            ### 6. 解碼與印出 ###
            next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(next_token, end="", flush=True)

            ### 7. 更新狀態 ###
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            new_attn = torch.ones((1, 1), device=device)
            attention_masks = torch.cat([attention_masks, new_attn], dim=-1)
            seen[next_token_id.item()] += 1

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        print("\n")
        return input_ids

def main():

    ### set user prompt ###

    user_prompt = "請寫一個大約 100 字的短篇故事，開頭是：那扇門從來沒有上鎖，但鎮上的人都不敢打開它。" # <-- 作業要換題目的時候，改這裡就好！
    
    ### load model and tokenizer ###

    print( f"[INFO] loading tokenizer & model" )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 修正 torch_dtype 警告
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=torch.bfloat16)
    model.eval() # setting to evaluation mode -> freeze model weights

    ### set user prompt for chat format ###

    # chat models are tuned with chat format, so models need the format messages to know it needs to chat with users

    print( f"[INFO] converting user prompt to chat format" )

    prompt = [
        {"role": "user", "content": user_prompt }
    ]

    prompt = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, # not to tokenize because we want to tokenize ourselves
                add_generation_prompt=True # indicate model to start answering by adding assistant hint token
             )

    ### use tokenizer to get input_ids ( token id ) and attention mask ###

    print( f"[INFO] tokenizing" )

    input_ids, attention_masks = Tokenizer.tokenize_pipe( tokenizer=tokenizer, prompt=prompt )
    input_ids = input_ids.to(device) # token id sequence
    attention_masks = attention_masks.to(device) # used to mask padding part and future part

    ### generation ###

    print( f"[INFO] generating" )

    with torch.no_grad():
        output_ids = Generator.generate_iterative( tokenizer=tokenizer, model=model, attention_masks=attention_masks, input_ids=input_ids )

    ### decode generated input_ids to token sequence ###

    print( f"[INFO] decoding" )

    output = output_ids[0][len(input_ids[0]):] # only take answer part and specify 0 for first data because our batch size is 1
    output = tokenizer.decode(output, skip_special_tokens=True)

    ### print input & output for observasion ###

    print()
    print( f"[USER INPUT]: {user_prompt}" )
    print( f"[CHAT INPUT]: {prompt}" )
    print( f"[OUTPUT]: {output}" )

### device setting ###

device = torch.device( f"cuda:0" )
model_name = "/model/Llama-3.2-3B-Instruct"
#model_name = "/model/Qwen2.5-3B-Instruct"
if __name__ == '__main__':    
    '''
    ### Example test code ###

    # test tokenizing ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained( model_name )
    Tokenizer.test( strategy="step-by-step", tokenizer=tokenizer, prompt="你好" )
    # test softmax ----------------------------------------------------------------
    logits = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    temperature = 1.0
    out_logits = Softmax( logits=logits, temperature=temperature )
    print( out_logits )
    # test filtering ----------------------------------------------------------------
    probs = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    logits = torch.tensor( [[ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]] )
    threshold = 0.96
    Filter.test( strategy="topP", probs=probs, logits=logits, threshold=threshold )
    # test sampling ----------------------------------------------------------------
    logits = torch.tensor( [[ 0.07, 0.06, 0.25, 0.41, 0.04, 0.03, 0.14 ]] )
    Sampler.test( strategy="random", logits=logits )    
    # test penalty ----------------------------------------------------------------
    logits = torch.tensor( [[ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0 ]] )
    seen = torch.tensor( [ 1, 0, 0, 2, 1, 0, 0 ] )
    penalty = 0.5
    Penalty.test( strategy="presence", logits=logits, seen=seen, penalty=penalty )
    '''
    # generating flow
    main()
