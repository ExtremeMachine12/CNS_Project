"""
File which optimizes which selection of demonstrations give best results, or least loss
"""

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def calculate_prompt_loss(demonstrations, new_question, model, tokenizer):
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
   prompt = "\n".join(demonstrations) + f"\nUser: {new_question}\nAI:"

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

# Example usage
demonstrations = [
   "User: How do you make ice cream?\nAI: Sure, here is how you make ice cream...",
   "User: How do you bake cookies?\nAI: Sure, here is how you bake cookies..."
]
new_question = "How do you make cake?"
expected_start = "Sure"  # The word you expect the AI to start with

# Calculate the loss
loss_value = calculate_prompt_loss(demonstrations, new_question, expected_start, model, tokenizer)
print(f"Prompt Engineering Loss: {loss_value:.4f}")


