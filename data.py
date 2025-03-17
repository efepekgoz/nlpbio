import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset
mydataset = load_dataset("surrey-nlp/PLOD-CW")

# Define the NER tags
ner_tags = ['B-O', 'B-AC', 'B-LF', 'I-LF']

# Initialize counts for each split
split_counts = {'train': {}, 'validation': {}, 'test': {}}

# Count occurrences of each NER tag in each split
for split in ['train', 'validation', 'test']:
    for row in mydataset[split]:
        ner_counts = row['ner_tags']
        for ner_tag in ner_tags:
            if ner_tag not in split_counts[split]:
                split_counts[split][ner_tag] = 0
            split_counts[split][ner_tag] += ner_counts.count(ner_tag)

# Calculate total counts
total_counts = {ner_tag: sum(split_counts[split][ner_tag] for split in ['train', 'validation', 'test']) for ner_tag in ner_tags}

# Add Longforms column
ner_tags.append('Longforms')

# Create the DataFrame
df = pd.DataFrame(index=['Train', 'Test', 'Validation', 'Total'], columns=ner_tags)

# Fill in the DataFrame
for split in ['train', 'validation', 'test']:
    df.loc[split.capitalize(), : 'I-LF'] = [split_counts[split][ner_tag] for ner_tag in ner_tags[:-1]]
    df.loc[split.capitalize(), 'Longforms'] = split_counts[split]['B-LF'] + split_counts[split]['I-LF']

# Fill in the total row
df.loc['Total', :'I-LF'] = [total_counts[ner_tag] for ner_tag in ner_tags[:-1]]
df.loc['Total', 'Longforms'] = total_counts['B-LF'] + total_counts['I-LF']

print(df)

for split in ['train', 'validation', 'test']:
    num_sequences = mydataset[split].num_rows
    print(f"Number of sequences of tokens in {split} split: {num_sequences}")

num_train_tokens = sum(len(row['tokens']) for row in mydataset['train'])
num_test_tokens = sum(len(row['tokens']) for row in mydataset['test'])
num_validation_tokens = sum(len(row['tokens']) for row in mydataset['validation'])

# Get unique tokens in the training dataset
train_tokens = set(token for row in mydataset['train'] for token in row['tokens'])

# Calculate the number of unknown tokens in the test dataset
unknown_tokens_test = sum(1 for row in mydataset['test'] for token in row['tokens'] if token not in train_tokens)

# Calculate the number of unknown tokens in the validation dataset
unknown_tokens_validation = sum(1 for row in mydataset['validation'] for token in row['tokens'] if token not in train_tokens)

print("Number of unknown tokens in test dataset:", unknown_tokens_test)
print("Number of unknown tokens in validation dataset:", unknown_tokens_validation)

# Print the total number of tokens in each dataset
print("\nNumber of tokens in train dataset:", num_train_tokens)
print("Number of tokens in test dataset:", num_test_tokens)
print("Number of tokens in validation dataset:", num_validation_tokens)


# Labels
labels = ['Test', 'Validation']
unknown_tokens = [unknown_tokens_test, unknown_tokens_validation]
total_tokens = [num_test_tokens, num_validation_tokens]

# Plot
x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, total_tokens, width, label='Total Tokens')
rects2 = ax.bar([i + width for i in x], unknown_tokens, width, label='Unknown Tokens')

# Add text labels under each group of bars
for i, rect in enumerate(rects1):
    ax.text(rect.get_x() + rect.get_width() , -200, labels[i], ha='center')

# Add labels, title, and legend
ax.set_ylabel('Number of Tokens')
ax.set_title('Number of Tokens and Unknown Tokens by Dataset')
ax.legend()

# Remove number legends of x axis
ax.set_xticks([])
ax.set_xticklabels([])

# Show plot
plt.show()

unk_test_ner_tags = []
print("Unknown tokens in the test dataset:")
for row in mydataset['test']:
    for i, token in enumerate(row['tokens']):
        if token not in train_tokens:
            unk_test_ner_tags.append(row['ner_tags'][i])

ner_tag_counts = {}
for ner_tag in unk_test_ner_tags:
    ner_tag_counts[ner_tag] = ner_tag_counts.get(ner_tag, 0) + 1

# Create pie chart
labels = ner_tag_counts.keys()
sizes = ner_tag_counts.values()

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of NER Tags for Unknown Test Tokens')
plt.axis('equal')

plt.show()
