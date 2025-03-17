import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

def load_and_prepare_data():
    # Load the dataset
    dataset = load_dataset("surrey-nlp/PLOD-CW")
    ner_tags = ['B-O', 'B-AC', 'B-LF', 'I-LF']
    return dataset, ner_tags

def count_ner_tags(dataset, splits, ner_tags):
    split_counts = {split: {tag: 0 for tag in ner_tags} for split in splits}
    for split in splits:
        for row in dataset[split]:
            for ner_tag in row['ner_tags']:
                if ner_tag in ner_tags:
                    split_counts[split][ner_tag] += 1
    return split_counts

def calculate_totals(split_counts, ner_tags):
    totals = {ner_tag: sum(split_counts[split][ner_tag] for split in split_counts) for ner_tag in ner_tags}
    return totals

def create_dataframe(split_counts, totals, ner_tags):
    index_order = ['Train', 'Test', 'Validation', 'Total']
    ner_tags_extended = ner_tags + ['Longforms']
    df = pd.DataFrame(index=index_order, columns=ner_tags_extended)

    # Fill the DataFrame for each split and the totals
    for split, name in zip(['train', 'test', 'validation'], index_order[:-1]):
        df.loc[name, ner_tags] = [split_counts[split][tag] for tag in ner_tags]
        df.loc[name, 'Longforms'] = split_counts[split]['B-LF'] + split_counts[split]['I-LF']

    df.loc['Total', ner_tags] = [totals[tag] for tag in ner_tags]
    df.loc['Total', 'Longforms'] = totals['B-LF'] + totals['I-LF']

    return df

def print_additional_info(dataset):
    for split in ['train', 'validation', 'test']:
        num_sequences = dataset[split].num_rows
        print(f"Number of sequences of tokens in {split} split: {num_sequences}")

    num_train_tokens = sum(len(row['tokens']) for row in dataset['train'])
    num_test_tokens = sum(len(row['tokens']) for row in dataset['test'])
    num_validation_tokens = sum(len(row['tokens']) for row in dataset['validation'])

    print("\nNumber of tokens in train dataset:", num_train_tokens)
    print("Number of tokens in test dataset:", num_test_tokens)
    print("Number of tokens in validation dataset:", num_validation_tokens)

def plot_data(dataset):
    labels = ['Test', 'Validation']
    train_tokens = set(token for row in dataset['train'] for token in row['tokens'])
    num_test_tokens = sum(len(row['tokens']) for row in dataset['test'])
    num_validation_tokens = sum(len(row['tokens']) for row in dataset['validation'])

    unknown_tokens_test = sum(1 for row in dataset['test'] for token in row['tokens'] if token not in train_tokens)
    unknown_tokens_validation = sum(1 for row in dataset['validation'] for token in row['tokens'] if token not in train_tokens)
    
    unknown_tokens = [unknown_tokens_test, unknown_tokens_validation]
    total_tokens = [num_test_tokens, num_validation_tokens]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, total_tokens, width, label='Total Tokens')
    rects2 = ax.bar([i + width for i in x], unknown_tokens, width, label='Unknown Tokens')

    ax.set_ylabel('Number of Tokens')
    ax.set_title('Number of Tokens and Unknown Tokens by Dataset')
    ax.legend()

    plt.show()

def main():
    dataset, ner_tags = load_and_prepare_data()
    splits = ['train', 'validation', 'test']
    split_counts = count_ner_tags(dataset, splits, ner_tags)
    totals = calculate_totals(split_counts, ner_tags)
    df = create_dataframe(split_counts, totals, ner_tags)
    print(df)
    print_additional_info(dataset)
    plot_data(dataset)

if __name__ == '__main__':
    main()
