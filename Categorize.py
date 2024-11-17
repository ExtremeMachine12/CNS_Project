import os, csv
import pandas as pd

categories = {
    'malware': ['script', 'exploit', 'antivirus', 'malware', 'botnet', 'viruses'],
    'hacking': ['hack', 'hacking', 'crack', 'cyber attack'],
    'weaponry': ['firearms', 'bomb', 'explosive', 'weapon', 'mass'],
    'violence': ['assassinate', 'poison', 'harm', 'violence', 'war', 'kill', 'assault', 'suicide',
                 'terrorist', 'fire', 'torture', 'violent', 'shooting', 'child'],
    'misinformation_or_manipulation': ['trick', 'manipulate', 'incites', 'promotes', 'false',
                                       'fake', 'propaganda', 'misinformation', 'radicalization',
                                       'social media', 'hoax'],
    'theft': ['steal', 'theft', 'stealing', 'scam', 'rob', 'phishing'],
    'financial_crime': ['stock', 'insider', 'trading', 'counterfeit', 'money', 'fraud', 'financial', 'taxes'],
    'illegal_drugs': ['drugs'],
    'harassment': ['cyberbully', 'stalk', 'bullying', 'bully', 'harass', 'cyberbullies'],
    'cheating': ['cheat', 'cheating'],
    'public_disorder': ['evade', 'hit and run', 'transportation']
}

output_directory = "categorized_prompts"
os.makedirs(output_directory, exist_ok=True)

# Function to categorize a sentence based on keywords
def categorize_sentence(sentence):
    for category, word_list in categories.items():
        if any(word in sentence.lower() for word in word_list):
            return category
    return "uncategorized"

# Read the CSV file and categorize each line
with open('harmful_behaviors.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if there is one

    # Dictionary to collect sentences by category
    categorized_data = {}

    # Process each line in the CSV
    for line in csv_reader:
        prompt = line[0]
        category = categorize_sentence(prompt)

        # Append the prompt to the appropriate category list. Line 1 has commas in it, so we need to append it as a string
        if category not in categorized_data:
            categorized_data[category] = []
        categorized_data[category].append([prompt, line[1]])

    for category, sentences in categorized_data.items():
        # Create a dataframe from the sentences
        df = pd.DataFrame(sentences, columns=["goal", "target"])
        df.to_csv(os.path.join(output_directory, f"{category}.csv"), index=False)
        
print("Sentences have been categorized and written to CSV files by category.")
