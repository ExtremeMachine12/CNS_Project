import os, json
import requests
import pandas as pd
import re


files = []
input_directory = "generated_responses"
output_directory = "final_demonstration_pool"
os.makedirs(output_directory, exist_ok=True)

for file in os.listdir(input_directory):
    file_path = os.path.join(input_directory, file)
    files.append(file_path)

for file in files:
    df = pd.read_csv(file)
    # Iterate through the dataframe and generate responses for each prompt
    for index, row in df.iterrows():
        category = file.split('/')[-1].split('.')[0]
        ideal_response = row["ideal_response"]
        ideal_response = re.sub('\bStep 1:\b', '[/INST] Step 1:')
        ideal_response = re.sub('\bStep 2:\b', '[/INST][/INST][/INST][/INST] Step 2:')
        ideal_response = re.sub('\bStep 3:\b', '[/INST][/INST][/INST][/INST] Step 3:')
        ideal_response = re.sub('\bStep 4:\b', '[/INST][/INST][/INST][/INST] Step 4:')
        ideal_response = re.sub('\bStep 5:\b', '[/INST][/INST][/INST][/INST] Step 5:')
        df.at[index, "ideal_response"] = ideal_response
    df.to_csv(os.path.join(output_directory, f"{category}.csv"), index=False)
    print(f"Saved responses for category: {category}")
