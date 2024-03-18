import sys, getopt
import click
import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from langrank import prepare_train_file_no_data, train_from_pickle, prepare_train_pickle_no_data, train
import pickle 
import pandas as pd
import re
import langrank.defaults as defaults


def read_ablations_dictionary(source, path = defaults.ABLATIONS_PATH):
    if source == "syntax_grambank":
        key = "Grambank"
    else:
        key = "Uriel"
    a = pd.read_csv(open(path, 'rb'))
    categories = list(a["Categories"])[:-2]
    cat_dict = {}
    for cat in categories:
        print(cat)
        if cat in ["quantification", "non-verbal predication", "argument marking (non-core)", "valency", "verb complex"] and source == "syntax_knn":
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

@click command()
@click.option("-t", "--task", type=str, default="MT", help="NLP task")
@click.option("-a", "--ablation", type=str, default= "", help="feature set to remove")
@click.option("-d", "--distance", type=bool, default= "False", help="feature set to remove")
@click.option("-s", "--source", type=str, default= "syntax_knn", help="syntax_knn or syntax_grambank")

def main(
    task,
    key,
    distance,
    source
):
    t_file = defaults.TRAIN_FILE.format(task = task)
    print(t_file)

    with open(t_file, 'rb') as f:
        training= pickle.load(f)

    ab = read_ablations_dictionary(source)

    if not key = "":
        print(key)
        exclude = ab[key]
    else:
        key = "dist" if distance else "full"
    
    model_dir = "./models/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir) 

    model_langs = list(training.keys())
    for lang in model_langs:
        print(lang)
        languages = list(training[lang][1].keys())
        rank = list(training[lang][1].values())
        if dist:
            train_one_distances(model_dir, task, languages, rank, lang)
        else:
            train_one_distanceless(model_dir, task, languages, rank, lang, source, exclude)
    
    print("finished training {task}".format(task = task))

if __name__ == "__main__":
   main(sys.argv[1:])
