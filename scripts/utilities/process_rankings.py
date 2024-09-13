'''
Reads in csv created by udpipe scripts and reformats rankings 
Returns any missing pairs
'''

FINAL_DATA = "./golds/udpipe_scores_2000_all.csv"
import pandas as pd
import pickle 
from iso639 import Lang

'''
Reads the raw output from udpipe training script
returns a csv where column = target, row = train/source
'''
def reorganize_rankings():
    df = pd.read_csv("./conllu/models/2000-500/accuracy.csv")
    lang_train_values = df['transfer'].unique()
    lang_pred_values = df['source'].unique()
    reshaped_df = pd.DataFrame(columns=['lang_train'] + list(lang_pred_values))
    reshaped_df['lang_train'] = lang_train_values
    reshaped_df.set_index('lang_train', inplace=True)

    for _, row in df.iterrows():
        lang_train = row['transfer']
        lang_pred = row['source']
        score = row['apos']
        reshaped_df.at[lang_train, lang_pred] = score

    # Reset the index
    reshaped_df.reset_index(inplace=True)
    reshaped_df.rename(columns = {'lang_train':'train/target'}, inplace = True)
    # Display the reshaped dataframe
    reshaped_df.to_csv(FINAL_DATA, index = False)

'''
Reads output of reorganize_rankings
saves list of missing target languages and list of missing transfer languages
'''
def find_empty_cells(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Initialize lists to store column names and row names
    source = []
    transfer = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the value in the 'lang_train' column (assuming it's the first column)
        lang_train_value = row['train/target']

        # Check for empty cells in other columns
        for col in df.columns:
            if pd.isnull(row[col]):
                # If cell is empty, add column name to source list and row name to transfer list
                source.append(col)
                transfer.append(lang_train_value)

    return source, transfer


def main():
    reorganize_rankings()
    csv_file_path = FINAL_DATA 
    source_list, transfer_list = find_empty_cells(csv_file_path)
    with open("./resources/missing_trnsfr.txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(transfer_list))
    with open("./resources/missing_tgt.txt", mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(source_list))

if __name__ == "__main__":
   main()