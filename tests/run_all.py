import sys, getopt
import click
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from langrank import prepare_train_file_no_data, train_from_pickle, prepare_train_pickle_no_data, train
import pickle 
import pandas as pd
import re
import defaults as defaults
import langrank as lr

from statistics import mean
import sklearn.metrics as sm
import numpy as np
NUM_FEATS = 113
#do i want to do it one language at a time? how would that look? -- I can only train/predict one lang at a time because rank takes the average
#Maybe I should separate rank_all again? I don't think i will because handling batchig correct task languages would be annoying

def read_ablations_dictionary(source, path = defaults.ABLATIONS_PATH):
    if source == "syntax_grambank":
        key = "Grambank"
    elif source == "syntax_knn":
        key= "Uriel"
    else:
        return None
    a = pd.read_csv(open(path, 'rb'))
    categories = list(a["Categories"])[:-2]
    cat_dict = {}
    for cat in categories: #what to exclude?
        print(cat)
        # if cat in ["quantification", "non-verbal predication", "argument marking (non-core)", "valency", "verb complex"]: 
        #         continue
        category_rows = a[a["Categories"] == cat]
    
        # Check if there's at least one row for the category
        if not category_rows.empty:
            print(category_rows[key])
            feats = category_rows.iloc[0][key].split(",")
            pattern = r'[^\w\s]|_'
            # Use re.sub() to replace all matches of the pattern with an empty string
            print(feats)
            feats  = [re.sub(pattern, '', s) for s in feats] 
            print(feats)
            cat_dict[cat] = feats
            # Print the features
            print(f"Features for category '{cat}': {feats}")
        else:
            print(f"No rows found for category '{cat}'")
    return cat_dict


def train_one(dir_path, task, train_langs, target_langs, rank, test_lang, source, exclude, distances, dataset, arch):
    print(f"target languages ranked (leave one out): {target_langs}")
    print(f"transfer languages {train_langs}")
    print(f"type of rank {type(rank)}")


    prepare_train_pickle_no_data(train_langs=train_langs, target_langs=target_langs, rank=rank, tmp_dir=dir_path, task = task, distances = distances, source = source, exclude = exclude, dataset = dataset, arch = arch)
    output_model = "{}/{}.txt".format(dir_path,test_lang)
    train_from_pickle(tmp_dir= dir_path, output_model=output_model)
    assert os.path.isfile(output_model)

def predict(predict_dir, arch, task, dist, source, exclude, dataset):
    with open(defaults.TRAIN_FILE.format(task=task, arch = arch), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    model_path = predict_dir.replace("./results/", "./models/" )
    for target_lang in languages:
        lang_path = "{path}/{lang}.txt".format(path = model_path, lang = target_lang)
        print(lang_path)
        train_langs = list(rankings[target_lang][0])
        prepared = None
        if dataset:
            prepared = lr.prepare_featureset(lang=target_lang, task = task) #don't need if we exclude dataset dependent feats
        predicted[target_lang] = lr.rank(test_lang = target_lang, test_dataset_features = prepared, task=task, candidates=train_langs, model = lang_path, distances = dist, source = source, exclude = exclude, arch = arch)
    pf = predict_dir + "/predictions.pkl"
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)

def scores_ranking(ranking, gamma_max = 10):
    scores_by_index = [0] * len(ranking)
    for i in range(len(ranking)): 
        if ranking[i] <= gamma_max:
            scores_by_index[i] = gamma_max - (ranking[i])
    return scores_by_index
    
def make_ranking(predicted, ranked_langs):
    ranking = [0] * len(ranked_langs)
    for rank, lang in enumerate(predicted):
        i = ranked_langs.index(lang)
        ranking[i] = rank
    return ranking

def compute_ndcg(lang, ranked_langs, predicted, gamma_max= 9, k=3):
    ranking_langs = ranked_langs[lang][0] # list of languages for looking up index in ranking vector
    # gives position in ranking based on index (if ranking[0] = 4 then the 0th language [ranking_langs[0]] is the 5th best)
    ranking = ranked_langs[lang][1] 
    # creates vector to look up the relevance score of a given language by index
    predicted_rank = make_ranking(predicted, ranking_langs)
    predicted_scores = scores_ranking(predicted_rank)
    gold_scores = scores_ranking(ranking)
    print(gold_scores)
    print(predicted_scores)
    return sm.ndcg_score(np.asarray([gold_scores]), np.asarray([predicted_scores]),k=k)


def save_ndcg(task, arch, predict_dir):    
    with open(predict_dir + "/predictions.pkl", 'rb') as f:
        predictions = pickle.load(f)

    with open(defaults.GOLD_FILE.format(task=task, arch=arch), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    ndcg = {lang: compute_ndcg(lang, rankings, predictions[lang]) for lang in languages}
    score = str(mean(ndcg.values()))

    with open(predict_dir + "ncdg.pkl", 'wb') as f:
        pickle.dump(ndcg, f)


    return score


@click.command()
@click.option("-t", "--task", type=str, default="POS", help="NLP task")
@click.option("-a", "--ablation", type=str, default= "None", help="feature set to remove")
@click.option("-d", "--distance", type=bool, default=False, help="distance or no")
@click.option("-s", "--source", type=str, default= "syntax_knn", help="syntax_knn or syntax_grambank both or none")
@click.option("-s", "--arch", type=str, default= "", help="stanza xlmr or orig")
@click.option("-a", "--dataset", type=str, default= False, help="use dataset features or no")

def main(
    task,
    ablation,
    distance,
    source, 
    arch, 
    dataset
):
    exclude = []
    t_file = defaults.TRAIN_FILE.format(arch = arch, task = task)

    with open(t_file, 'rb') as f:
        training= pickle.load(f)

    # ab = read_ablations_dictionary(source)

    # if ablation:
    #     key = ablation
    #     print(key)
    #     exclude = ab[key]
    # else:
    key = "dist" if distance == True else "full"
    
    model_dir = "./models/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key, arch = arch, dataset = dataset)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir) 

    model_langs = list(training.keys())
    print(f"training {task} for ablation {key} from source {source}")
    for lang in model_langs:
        print(lang)
        target_languages = list(training[lang][1].keys()) #why did i format my data so weird!
        train_languages = list(training[lang][0]) 
        rank = list(training[lang][1].values())
        train_one(model_dir, task, train_languages, target_languages, rank, lang, source, exclude, distance, dataset, arch)
        
    
    print("finished training")
    print("now predicting")
    print(distance)
    predict_dir = "./results/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key, arch = arch, dataset = dataset)
    if not os.path.exists(predict_dir): 
            os.makedirs(predict_dir) 
    predict(predict_dir, arch, task, distance, source, exclude, dataset, arch)

    print("finished predicting")
    score = save_ndcg(task, arch, predict_dir)
 
    print("finished ranking")
    with open(defaults.RESULTS_PATH, "a") as f:
        f.write("\t".join([task, key, source, str(distance), arch, str(dataset)]) + f"\t{score}")
        if not exclude==[]:
            norm = NUM_FEATS - len(exclude)
            norm_score = float(score)/norm
            f.write(f"\t{norm_score}")
        f.write("\n")
    
if __name__ == "__main__":
   main(sys.argv[1:])