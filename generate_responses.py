import os, json
import requests
import pandas as pd

files = []
input_directory = "categorized_prompts"
output_directory = "generated_responses"
os.makedirs(output_directory, exist_ok=True)

def generate_ideal_response(prompt, model="llama2-uncensored"):
	url = "http://localhost:11434/api/generate"

	payload = json.dumps({
		"model": model,
		"prompt": prompt,
		"stream": False
	})

	headers = {
		'Content-Type': 'application/json'
	}

	response = requests.request("POST", url, headers=headers, data=payload)
	response = json.loads(response.text)["response"]
	return response

for file in os.listdir(input_directory):
	file_path = os.path.join(input_directory, file)
	files.append(file_path)

for file in files:
	df = pd.read_csv(file)
	# Iterate through the dataframe and generate responses for each prompt
	for index, row in df.iterrows():
		category = file.split('/')[-1].split('.')[0]
		prompt = row["goal"]
		response = generate_ideal_response(prompt)
		df.at[index, "response"] = response
		print(f"Category: {category}, Prompt: {prompt}")
		print(f"At row {index + 1} and total rows to go: {len(df) - index - 1}")
	# Save the updated dataframe with responses in a new directory called "generated_responses"
	df.to_csv(os.path.join(output_directory, f"{category}.csv"), index=False)
	print(f"Saved responses for category: {category}")
