"""
File which optimizes which selection of demonstrations give best results, or least loss
"""
import os

import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F

demo_list = ["dolphin-mistral_demonstration_pool", 'wizard-vicuna-uncensored_demonstration_pool',
              'llama2-uncensored_demonstration_pool']
model_list = ["dolphin-mistral", 'wizard-vicuna-uncensored', 'llama2-uncensored']

os.makedirs(f"RS_Demo_Pool", exist_ok=True)

for model_name in model_list:
	os.makedirs(f"RS_Demo_Pool/{model_name}", exist_ok=True)



model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def calculate_prompt_loss(demonstrations, target_prompt, model, tokenizer):
   """
   Calculate the loss based on the probability that the first token of the AI's response is 'Sure'.

   Parameters:
   - demonstrations: List of prompt engineering demonstrations.
   - new_question: The new question for the AI.
   - expected_start: The expected first word of the AI's response (e.g., "Sure").
   - model: Language model to evaluate.
   - tokenizer: Corresponding tokenizer for the model.

   Returns:
   - loss: Scalar loss (based on the log-probability of the expected token being 'Sure').
   """
   # Combine demonstrations and new question into a single prompt
   prompt = "\n".join(demonstrations) + f"{target_prompt}"

   # Tokenize the prompt
   inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

   # Get the logits from the model
   with torch.no_grad():
       outputs = model(**inputs, labels=inputs["input_ids"])
       logits = outputs.logits  # [batch_size, seq_length, vocab_size]

   # The first token we want to check for is "Sure"
   sure_token_id = tokenizer.encode("Sure")[0]

   # Extract the logits for the first token (the AI's response start)
   first_token_logits = logits[0, -1, :]  # Get logits for the last token (model's prediction)

   # Convert logits to probabilities using softmax
   probs = F.softmax(first_token_logits, dim=-1)

   # Get the probability of the "Sure" token being the first token
   sure_token_prob = probs[sure_token_id].item()

   # Calculate the negative log-likelihood loss (binary cross-entropy)
   loss = -torch.log(torch.tensor(sure_token_prob))  # Negative log-likelihood

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
            target_prompt = df['target'].sample(n=1).tolist
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
