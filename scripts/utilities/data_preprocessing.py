import numpy as np
import pandas as pd
import pickle
import pandas as pd
import scipy.stats as ss
import sklearn.metrics as sm
from statistics import mean
import os
import pkg_resources
from iso639 import Lang
import os
import sys

# setting path
current_dir = os.path.dirname(os.path.realpath(__file__))
langrank_path = os.path.join(current_dir, '../../')
sys.path.append(langrank_path)
import langrank as lr
import defaults as defaults
import gram2vec.lang2vec.lang2vec as l2v

'''
removes non-grambank results from the original langrank golds
'''
def reorganize_langrank_rankings():
    df = pd.read_csv(defaults.OLD_RAW)
    gram = l2v.GRAMBANK_DISTANCE_LANGUAGES
    gram.append("train/target")
    print(gram)
    df = df[df.columns.intersection(gram)]
    df = df[df["train/target"].isin(gram) == True]
    df.to_csv(defaults.GRAM_RAW, index = False)

'''
Reads the raw output from stanza training script
returns a csv where column = target, row = train/source
'''
def reorganize_stanza_rankings():
    df = pd.read_csv(defaults.STANZA_DEV)
    lang_train_values = df['transfer'].unique()
    lang_pred_values = df['target'].unique()
    reshaped_df = pd.DataFrame(columns=['lang_train'] + list(lang_pred_values))
    reshaped_df['lang_train'] = lang_train_values
    reshaped_df.set_index('lang_train', inplace=True)

    for _, row in df.iterrows():
        lang_train = row['transfer']
        lang_pred = row['target']
        score = row['UPOS']
        reshaped_df.at[lang_train, lang_pred] = score
    # Reset the index
    reshaped_df.reset_index(inplace=True)
    reshaped_df.rename(columns = {'lang_train':'train/target'}, inplace = True)
    # Display the reshaped dataframe
    df_clean = reshaped_df
    # df_clean = reshaped_df.map(lambda x: x.replace('%', ''))
    df_clean = df_clean.drop(columns=["jpn"])
    df_clean = df_clean[~df_clean['train/target'].str.contains('jpn')]

    df_clean.to_csv(defaults.FINAL_STANZA, index = False)

def reorganize_xlmr_rankings():
    df = pd.read_csv(defaults.XPOS_DEV)
    # Get unique values of "lang_train" and "lang_pred"
    lang_train_values = df['lang_train'].unique()
    lang_pred_values = df['lang_pred'].unique()

    # Create a new DataFrame with "lang_train" as the first column and "lang_pred" as the remaining columns
    reshaped_df = pd.DataFrame(columns=['train/target'] + list(lang_pred_values))
    reshaped_df['train/target'] = lang_train_values

    # Set "lang_train" as the index
    reshaped_df.set_index('train/target', inplace=True)

    # Iterate over the CSV to populate values in the new DataFrame
    for _, row in df.iterrows():
        lang_train = row['lang_train']
        lang_pred = row['lang_pred']
        score = row['score']
        reshaped_df.at[lang_train, lang_pred] = score

    # Reset the index
    reshaped_df.reset_index(inplace=True)
    # Display the reshaped dataframe
    reshaped_df.to_csv(defaults.FINAL_XLMR, index = False)

'''
Reads output of reorganize_rankings
saves list of missing target languages and list of missing transfer languages
'''
def find_empty_cells(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Initialize lists to store column names and row names
    target = []
    transfer = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the value in the 'lang_train' column (assuming it's the first column)
        lang_train_value = row['train/target']

        # Check for empty cells in other columns
        for col in df.columns:
            if pd.isnull(row[col]):
                # If cell is empty, add column name to source list and row name to transfer list
                target.append(col)
                transfer.append(lang_train_value)

    return target, transfer

'''
converts all lang codes to 3 letter iso codes
'''
def to_iso(language):
    return Lang(language).pt3

''' 
processes gold ranking files to convert all 2 letter iso codes to 3 letters
expects "train/target" to be in (0,0)
'''
def convert_datafile_isos(savepath):

    df = pd.read_csv(savepath)
    new_columns = {}
    for column in df.columns:
        if len(column) == 2:  
            new_columns[column] = to_iso(column)
        
    df = df.rename(columns=new_columns)
    for index, row in df.iterrows():
        if len(row['train/target']) == 2:
            df.at[index, 'train/target'] = to_iso(row['train/target'])


    df.to_csv(savepath, index=False)

'''
takes a list of 3 letter iso codes and removes all 3 letter isocodes from ranking data that are not present in the list
'''
def match_langs(match, data, column_name = "train/target"):
    filtered_data = data[data[column_name].isin(match)] #removes train language data not present in list
    filtered_data = filtered_data[[column_name] + [col for col in filtered_data.columns[1:] if col in match]] #removes target language data not present in list
    return filtered_data

def align_and_remove_unique_columns(df1, df2):
    # Find columns unique to each dataframe
    unique_cols_df1 = set(df1.columns) - set(df2.columns)
    unique_cols_df2 = set(df2.columns) - set(df1.columns)
    
    # Drop unique columns from each dataframe
    df1 = df1.drop(columns=unique_cols_df1)
    df2 = df2.drop(columns=unique_cols_df2)

    # Get the column order of df1
    column_order = df1.columns.tolist()
    
    # Reorder columns of df2 to match df1
    df2 = df2[column_order]

    return df1, df2

def align_and_remove_unique_rows(df1, df2, column_name="train/target"):
    # Find values unique to each dataframe in the specified column
    unique_values_df1 = set(df1[column_name]) - set(df2[column_name])
    unique_values_df2 = set(df2[column_name]) - set(df1[column_name])
    
    # Drop rows with unique values in the specified column from each dataframe
    df1 = df1[~df1[column_name].isin(unique_values_df1)]
    df2 = df2[~df2[column_name].isin(unique_values_df2)]
     # Use the values in the first column as the index
    df1_indexed = df1.set_index('train/target')
    df2_indexed = df2.set_index('train/target')
    
    # Reindex df2 to match the order of df1's index
    df2 = df2_indexed.reindex(df1_indexed.index)
    
    return df1, df2.reset_index()

'''
Given two organized datasets (all iso codes of 3 letters with "train/test" in (0,0)) 
reindex to match row and column order and save to file
'''
def match_data(a_path, b_path, aligned_a, aligned_b):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a, b = align_and_remove_unique_columns(a, b)
    a, b = align_and_remove_unique_rows(a, b)
    a.to_csv(aligned_a, index = False)
    b.to_csv(aligned_b, index = False)

'''
For whatever reason, langrank includes languages in the raw data that are not actually present in their POS datasets. 
This removes them so they don't cause issues during the creation of training data
'''
def remove_no_data_langs(data, task = "POS", column_name = "train/target"):
    feature_database = np.load("/projects/enri8153/langrank/gram2vec/lang2vec/data/grambank.npz", allow_pickle = True)
    mask = np.all(feature_database["data"] != -1.0, axis=0)
    langs = list(feature_database["langs"])
    avail = set()
    for l in langs:
        avail.add(l)
    target_langs =  list(data.columns)[1:]
    train_langs = list(data[column_name])
    datasets_dict = lr.map_task_to_data(task)
    for dt in datasets_dict:
        fn = os.path.join(langrank_path, 'indexed', task, datasets_dict[dt])
        features = np.load(fn, encoding='latin1', allow_pickle=True).item()
    missing = set()
    languages = set(target_langs + train_langs)
    for lang in languages:
        code = (lr.PREFIXES[task] + lang)
        if not code in features:
            code = lr.PREFIXES[task] + Lang(lang).pt1
        if task == "POS":
            code = [a for a in list(features.keys()) if a.startswith(lr.PREFIXES["POS"] + Lang(lang).pt1)]
            if len(code) > 0:
                code = code[0]
        if (code == []) or (not code in features) or (not lang in avail): 
            print(lang)
            missing.add(lang)
    data = data[~data[column_name].isin(missing)]
    missing_cols = missing.intersection(set(target_langs))
    data = data.drop(columns=missing_cols)
    return data


'''
Given a language and an organized dataset, returns 
ranked: a list of integers where the value at index i is the ranking of the corresponding language (based on order in the dataset) at index i 
'''
def numerical_rank(language, data):
    values = list(data[language])
    values = [-float(a) for a in values]
    ranked =  ss.rankdata(values, method = "ordinal") 
    ranked = [i - 1 for i in ranked]
    return ranked

'''
Given an organized dataset, returns 
langs_ranked: a dictionary where 
keys = language and 
values = a tuple of (indices, ranked) where the ranking at index i of ranked corresponds to the language at index i of indices
'''
def make_golds(data, col_name = "train/target"):
    test = list(data.columns)[1:]
    langs_ranked = {}
    for language in test: 
        df = data.copy()
        df = df.drop(data.loc[data[col_name].isin([language])].index) #for fairness, remove target language from potential training languages
        indices = list(df[col_name])
        ranked = numerical_rank(language, df)
        langs_ranked[language] = (indices, ranked)
    return langs_ranked

'''
Given an organized dataset and a language, returns 
a tuple of (indices, rankings)
indices: a list of ranked languages (full set of test languages - leave one out language)
rankings: a dictionary (as in make_golds); full set of test languages ranked with current lang removed
'''
def make_ranked(data, target_lang, col_name = "train/target"):
    remove = [target_lang]
    # removes current language from ranking (for leave one out)
    data = data.drop(remove,axis = 1) 
    data = data.drop(data.loc[data[col_name].isin(remove)].index) 
    test = list(data.columns)[1:] #list of target languages
    indices = list(data[col_name]) #list of training languages
    rankings = {}
    for test_language in test:
        ranked = numerical_rank(test_language, data)
        rankings[test_language] = ranked
    return (indices, rankings)

'''
Given a language, task and an organized dataset, saves gold rankings and leave-one-out ranked training data to pickle 
'''
def make_data_pickles(data_path, approach, training_dir = "./training-data/"):
    data = pd.read_csv(data_path)
    data = remove_no_data_langs(data)
    golds = make_golds(data)
    languages = list(data.columns)[1:]
    ranks = {language: make_ranked(data.copy(), language) for language in languages}
    save_path = f"{training_dir}{approach}_POS_"
    with open(save_path + "golds.pkl", 'wb') as f:
        pickle.dump(golds, f)
    with open(save_path + "ranked.pkl", 'wb') as f:
        pickle.dump(ranks, f)

'''
Prepares a reference list for looking up rank scores; 
returns a list of the same length as ranking but ranks gives a score instead
if gamma max were 3 then [0, 5 , 3, 4, 2, 1] would become [3, 0, 0, 0, 1, 2]
'''
def scores_ranking(ranking, gamma_max = 10):
    scores_by_index = [0] * len(ranking)
    for i in range(len(ranking)): 
        if ranking[i] <= gamma_max:
            scores_by_index[i] = gamma_max - (ranking[i])
    return scores_by_index

'''
Computes ncdg for a single pair of rankings. ranking and compare_ranking MUST be indexed the same
'''
def compute_ncdg(indices, ranking, compare_ranking, gamma_max = 10 , k = 5):
    # gives position in ranking based on index (if ranking[0] = 4 then the 0th language [ranking_langs[0]] is the 5th best)
    ranking = ranking 
    # creates vector to look up the relevance score of a given language by index
    golds_scores = scores_ranking(ranking, gamma_max)
    compare_scores = scores_ranking(compare_ranking, gamma_max)
    # ideal_score = [i for i in reversed(range(1, gamma_max + 1))] #creates an ideal score list up to gamma-max 
    # scores =  []
    # for i in range(gamma_max, 0, -1): 
    #     a = compare_scores.index(i) #gets the index of the predicted top n languages
    #     scores.append(golds_scores[a]) #appends the actual relevance score of each predicted language (based on gold rankings)
    return sm.ndcg_score(np.asarray([golds_scores]), np.asarray([compare_scores]),k=k)


'''
Given two gold data pickles; compute the average ncdg@3 (for comparing xpos and mtt)
'''
def compute_ncdg_from_golds(data_path, compare_path, training_dir= './training-data/'):
    golds = pickle.load(open(f'{training_dir}{data_path}', 'rb'))
    compare = pickle.load(open(f'{training_dir}{compare_path}', 'rb'))
    assert list(golds.keys()) == list(compare.keys())
    scores = []
    for lang in golds.keys():
        gold_indices = golds[lang][0]
        compare_indices = compare[lang][0]
        assert gold_indices == compare_indices
        gold_ranking = golds[lang][1] #these rankings are of the form [0,3,1,2] which correspond to the index lookup vectors
        compare_ranking = compare[lang][1]
        assert len(compare_ranking) == len(gold_indices)
        scores.append(compute_ncdg(gold_indices,gold_ranking, compare_ranking))
    return str(mean(scores))
    
'''
Given a pandas dataframe of sorted data, returns a dictionary of the top 3 languages 
'''
def get_topk_training_languages(data , k=3):
    top_train_languages = {}
    df = data.copy()
    # Iterate over each target language column
    for target_language in df.columns[1:]:
        # Sort the dataframe by the values of the current target language column
        sorted_df = df.sort_values(by=target_language, ascending=False)

        # Get the top 3 train languages for the current target language
        top_train = sorted_df['train/target'].head(k).tolist()

        # Store the top 3 train languages in the dictionary
        top_train_languages[target_language] = top_train

    return top_train_languages

def main():
    reorganize_stanza_rankings()
    reorganize_xlmr_rankings()
    stanza_path = defaults.FINAL_STANZA
    target_list, transfer_list = find_empty_cells(stanza_path)
    if not target_list == []:
        with open("./resources/missing_trnsfr.txt", mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(transfer_list))
        with open("./resources/missing_tgt.txt", mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(target_list))
        print("please train missing models")
    else:
        #first sort out the stanza data
        stanza= pd.read_csv(stanza_path)
        train = list(stanza["train/target"])
        with open(stanza_path, 'w') as f:
            stanza.to_csv(f, index=False)

        #then align xlmr data to stanza
        xlmr_path = defaults.FINAL_XLMR
        convert_datafile_isos(xlmr_path)
        match_data(xlmr_path, stanza_path, defaults.ALIGNED_XPOS, defaults.ALIGNED_STANZA)
        
        #then align original lstm data to new 
        match_data(defaults.OLD_RAW, stanza_path, defaults.ALIGNED_ORIG, defaults.ALIGNED_STANZA_ORIG)
        
        #then save the resultant rankings ot a langrank readable format
        make_data_pickles(defaults.ALIGNED_XPOS, "xlmr")
        make_data_pickles(defaults.ALIGNED_STANZA, "stanza")

        reorganize_langrank_rankings()
        make_data_pickles(defaults.GRAM_RAW, "orig")

        aligned_stanza = pd.read_csv("./golds/aligned_stanza.csv")
        with open("./golds/top_stanza.csv", mode='wt') as f:
            df = pd.DataFrame(get_topk_training_languages(aligned_stanza))
            df.to_csv(f)

        aligned_xlmr = pd.read_csv("./golds/aligned_xlmr.csv")
        with open("./golds/top_xlmr.csv", mode='wt') as f:
            df = pd.DataFrame(get_topk_training_languages(aligned_xlmr))
            df.to_csv(f)

        print(compute_ncdg_from_golds("stanza_POS_golds.pkl", "xlmr_POS_golds.pkl"))

if __name__ == "__main__":
   main()