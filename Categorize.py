"""
File to categorize attack prompts into separate csvs
"""
import csv
import os

categories = {
    'Malware': ['script', 'exploit', 'antivirus', 'malware', 'botnet', 'viruses'],
    'Hacking': ['hack', 'hacking', 'crack', 'cyber attack'],
    'Weaponry': ['firearms', 'bomb', 'explosive', 'weapon', 'mass'],
    'Violence': ['assassinate', 'poison', 'harm', 'violence', 'war', 'kill', 'assault', 'suicide',
                 'terrorist', 'fire', 'torture', 'violent', 'shooting', 'child'],
    'Misinformation or Manipulation': ['trick', 'manipulate', 'incites', 'promotes', 'false',
                                       'fake', 'propaganda', 'misinformation', 'radicalization',
                                       'social media', 'hoax'],
    'Theft': ['steal', 'theft', 'stealing', 'scam', 'rob', 'phishing'],
    'Financial Crime': ['stock', 'insider', 'trading', 'counterfeit', 'money', 'fraud', 'financial', 'taxes'],
    'Illegal Drugs': ['drugs'],
    'Harassment': ['cyberbully', 'stalk', 'bullying', 'bully', 'harass', 'cyberbullies'],
    'Cheating': ['cheat', 'cheating'],
    'Public Disorder': ['evade', 'hit and run', 'transportation']
}

output_dir = "script_filtered_categorized_prompts"
os.makedirs(output_dir, exist_ok=True)

# Function to categorize a sentence based on keywords
def categorize_sentence(sentence):
    for category, word_list in categories.items():
        if any(word in sentence.lower() for word in word_list):
            return category
    return "Uncategorized"

# Read the CSV file and categorize each line
with open('harmful_behaviors.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if there is one

    # Dictionary to collect sentences by category
    categorized_data = {}

    # Process each line in the CSV
    for line in csv_reader:
        sentence = line[0]
        category = categorize_sentence(sentence)

        # Append the sentence to the appropriate category list
        if category not in categorized_data:
            categorized_data[category] = []
        categorized_data[category].append(sentence)

    # Write each category to its own CSV file
    for category, sentences in categorized_data.items():
        output_file = os.path.join(output_dir, f"{category}.csv")
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            for sentence in sentences:
                writer.writerow([sentence])

print("Sentences have been categorized and written to CSV files by category.")
