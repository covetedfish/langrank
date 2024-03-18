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

#do i want to do it one language at a time? how would that look? -- I can only train/predict one lang at a time because rank takes the average
#Maybe I should separate rank_all again? I don't think i will because handling batchig correct task languages would be annoying

def read_ablations_dictionary(source, path = defaults.ABLATIONS_PATH):
    if source == "syntax_grambank":
        key = "Grambank"
    else:
        key = "Uriel"
    a = pd.read_csv(open(path, 'rb'))
    categories = list(a["Categories"])[:-2]
    cat_dict = {}
    for cat in categories: #what to exclude?
        if cat in ["quantification", "non-verbal predication", "argument marking (non-core)", "valency", "verb complex"]: 
                continue
        feats = list(a[key][a.loc[a["Categories"] == cat].index])[0].split(",")
        pattern = r'[^\w\s]|_'
    # Use re.sub() to replace all matches of the pattern with an empty string
        feats  = [re.sub(pattern, '', s) for s in feats]
        cat_dict[cat] = feats
    return cat_dict

def train_one_distanceless(dir_path, task, langs, rank, test_lang, source, exclude):
    print(langs)
    prepare_train_pickle_no_data(langs=langs, rank=rank, tmp_dir=dir_path, task = task, distances = False, source = source, exclude = exclude)
    output_model = "{}/{}.txt".format(dir_path,test_lang)
    train_from_pickle(tmp_dir= dir_path, output_model=output_model)
    assert os.path.isfile(output_model)

def train_one_distances(dir_path, task, langs, rank, test_lang):
    print(langs)
    prepare_train_file_no_data(langs=langs, rank=rank, tmp_dir=dir_path, task = task)
    output_model = "{}/{}.txt".format(dir_path,test_lang)
    train(tmp_dir= dir_path, output_model=output_model)
    assert os.path.isfile(output_model)


def predict(predict_dir, task, dist, source, exclude):
    with open(defaults.TRAIN_FILE.format(task=task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    model_path = predict_dir.replace("./results/", "./models/" )
    for lang in languages:
        lang_path = "{path}/{lang}.txt".format(path = model_path, lang = lang)
        print(lang_path)
        cands = rankings[lang][0]
        cands.append(lang)
        prepared = lr.prepare_featureset(lang=lang, task = task)
        predicted[lang] = lr.rank(prepared, test_lang = lang, task=task, candidates=cands, model = lang_path, distances = dist, source = source, exclude = exclude)
    pf = predict_dir + "/predictions.pkl"
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)

def compute_ndcg(lang, ranked_langs, predicted, gamma_max= 9, k=3):
    ranking_langs = ranked_langs[lang][0] # list of languages for looking up index in ranking vector
    # gives position in ranking based on index (if ranking[0] = 4 then the 0th language [ranking_langs[0]] is the 5th best)
    ranking = ranked_langs[lang][1] 
    print(ranking)
    # creates vector to look up the relevance score of a given language by index
    scores_by_index = [0] * len(ranking)
    for i in range(len(ranking)): 
        if ranking[i] <= gamma_max:
            scores_by_index[i] = gamma_max - (ranking[i] -1)
    ideal_score = [i for i in reversed(range(1, gamma_max + 1))] + [0] * (len(ranking) - gamma_max)
    print(ideal_score)
    predicted_score = [0] * len(ranking)
    for j in range(len(predicted)): #for each language in ranking
        code = predicted[j]
        index = ranking_langs.index(code)
        score = scores_by_index[index] #finds the true relevance of each language
        predicted_score[j] = score
    print(predicted_score)
    return sm.ndcg_score(np.asarray([ideal_score]), np.asarray([predicted_score]),k=k)


def save_ndcg(task, predict_dir):    
    with open(predict_dir + "/predictions.pkl", 'rb') as f:
        predictions = pickle.load(f)

    with open(defaults.GOLD_FILE.format(task=task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    ndcg = {lang: compute_ndcg(lang, rankings, predictions[lang]) for lang in languages}
    score = str(mean(ndcg.values()))

    with open(predict_dir + "ncdg.pkl", 'wb') as f:
        pickle.dump(ndcg, f)


    return score


@click.command()
@click.option("-t", "--task", type=str, default="MT", help="NLP task")
@click.option("-a", "--ablation", type=str, default= "Noe", help="feature set to remove")
@click.option("-d", "--distance", type=bool, default=False, help="feature set to remove")
@click.option("-s", "--source", type=str, default= "syntax_knn", help="syntax_knn or syntax_grambank")

def main(
    task,
    ablation,
    distance,
    source
):
    exclude = []
    print(distance)
    t_file = defaults.TRAIN_FILE.format(task = task)
    print(t_file)

    with open(t_file, 'rb') as f:
        training= pickle.load(f)

    ab = read_ablations_dictionary(source)

    if ablation:
        key = ablation
        print(key)
        exclude = ab[key]
    else:
        key = "dist" if distance == True else "full"
    
    model_dir = "./models/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key)
    print(model_dir)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir) 

    model_langs = list(training.keys())
    print(f"training {task} for ablation {key} from source {source}")

    # for lang in model_langs:
    #     print(lang)
    #     languages = list(training[lang][1].keys())
    #     rank = list(training[lang][1].values())
    #     if distance:
    #         train_one_distances(model_dir, task, languages, rank, lang)
    #     else:
    #         train_one_distanceless(model_dir, task, languages, rank, lang, source, exclude)
    
    # print("finished training")

    predict_dir = "./results/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key)
    if not os.path.exists(predict_dir): 
            os.makedirs(predict_dir) 
    predict(predict_dir, task, distance, source, exclude)

    print("finished predicting")
    score = save_ndcg(task, predict_dir)
    print("finished ranking")
    with open(defaults.RESULTS_PATH, "a") as f:
        f.write("\t".join([task, key, source, distance]) + f"\t{score}\n")

    
if __name__ == "__main__":
   main(sys.argv[1:])
