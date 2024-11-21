import os
import pandas as pd
import re



input_directory = "generated_responses"
model_list = ["dolphin-mistral", 'wizard-vicuna-uncensored', 'llama2-uncensored']
output_directory = "demonstration_pool"
os.makedirs(output_directory, exist_ok=True)

for model_name in model_list:
    os.makedirs(f"{model_name}_{output_directory}", exist_ok=True)

for model_name in model_list:
    files = []
    for file in os.listdir(f"{model_name}_{input_directory}"):
        file_path = os.path.join(f"{model_name}_{input_directory}", file)
        files.append(file_path)

    for file in files:
        df = pd.read_csv(file)
        # Iterate through the dataframe and generate responses for each prompt
        for index, row in df.iterrows():
            category = file.split('/')[-1].split('.')[0]
            ideal_response = row["ideal_response"]
            ideal_response = re.sub('Step 1:', '[/INST] Step 1:', ideal_response)
            ideal_response = re.sub('Step 2', '[/INST][/INST][/INST][/INST] Step 2:', ideal_response)
            ideal_response = re.sub('Step 3:', '[/INST][/INST][/INST][/INST] Step 3:', ideal_response)
            ideal_response = re.sub('Step 4:', '[/INST][/INST][/INST][/INST] Step 4:', ideal_response)
            ideal_response = re.sub('Step 5:', '[/INST][/INST][/INST][/INST] Step 5:', ideal_response)
            df.at[index, "ideal_response"] = ideal_response
        df.to_csv(os.path.join(f"{model_name}_{output_directory}", f"{category}.csv"),
                  index=False, quotechar='"')
        print(f"Saved responses for category: {category}")
