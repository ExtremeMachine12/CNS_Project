"""
File which optimizes which selection of demonstrations give best results, or least loss
"""
import os

import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F

model_list = ['llama2-uncensored', 'mistral']

os.makedirs(f"RS_Demo_Pool", exist_ok=True)

for model_name in model_list:
    os.makedirs(f"RS_Demo_Pool/{model_name}", exist_ok=True)



model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def calculate_prompt_loss(demonstrations, target_prompt, model, tokenizer):
    """

    Args:
        demonstrations: A randomly sampled list of 8 I-FSJs
        target_prompt: The test prompt we are testing against
        model: Just LLama2
        tokenizer: An object which can turn the input into tokens compatible with the Llama model

    Returns:
         Loss value, aka how far away the model is from generating "Step" as the first token
   """
    prompt = "\n".join(demonstrations) + f"{target_prompt}"  # make the input the entire script
    # of demonstrations plus the target prompt

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():  # reduces memory consumptions
        outputs = model(**inputs)
        logits = outputs.logits  # get the logit values

    step_token_id = tokenizer.encode("Step")[0]  # tokenize the word step
    first_token_logits = logits[0, 0, :]  # Get logit for first token
    probs = F.softmax(first_token_logits, dim=0)
    step_token_prob = probs[step_token_id].item()

    loss = -torch.log(torch.tensor(step_token_prob))  # Negative log-likelihood

    return loss.item()

# Iterate over the demonstration pools for each model
for model_name in model_list:
    files = []  # reset the file list for each model type
    for file in os.listdir(f"{model_name}_demonstration_pool"):
        file_path = os.path.join(f"{model_name}_demonstration_pool", file)
        files.append(file_path)

    for file in files:
        df = pd.read_csv(file)
        # Iterate and optimize on our loss function for 10 iterations:
        Loss_min = float('inf')
        best_demos_so_far = []

        # Randlomly sample a target prompt from the current category
        target_prompt = df['target'].sample(n=1).tolist

        for i in range(10):
            print(f"Optimizing for {file.split('/')[-1]}: Iteration {i+1}")
            # Randomly sample 8 demonstrations
            random_samples = df['ideal_response'].sample(n=8).tolist() # sample 8 demonstrations since 8 shots
            # Randlomly sample a target prompt
            target_prompt = df['target'].sample(n=1).tolist()
            # Calculate the loss
            loss_value = calculate_prompt_loss(random_samples, target_prompt, model, tokenizer)
            print(f"Loss: {loss_value}")
            if loss_value < Loss_min:
                Loss_min = loss_value
                best_demos_so_far = random_samples

        # Save the best demonstrations using pandas
        best_demos_df = pd.DataFrame(best_demos_so_far, columns=["ideal_response"])
        #save in a csv file
        best_demos_df.to_csv(os.path.join(f"RS_Demo_Pool/{model_name}", f"{file.split('/')[-1]}"), index=False, quotechar='"')
        print(f"Best demonstrations for {file.split('/')[-1]} saved successfully!")
