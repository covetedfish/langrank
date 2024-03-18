import os
import sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import langrank as lr
import pickle
import sys, getopt
import pandas as pd
import re

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


def predict(predict_dir, task, dist, source, exclude):
    with open(defaults.TRAIN_FILE.format(task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    model_path = predict_dir.replace("./results/", "./models/" )
    for lang in languages:
        lang_path = "{path}{lang}.txt".format(path = path, lang = lang)
        print(lang_path)
        cands = rankings[lang][0]
        cands.append(lang)
        prepared = lr.prepare_featureset(lang=lang, task = task)
        predicted[lang] = lr.rank(prepared, test_lang = lang, task=task, candidates=cands, model = lang_path, distances = dist, source = source, exclude = exclude)
    pf = predict_dir + "predictions.pkl"
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)



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
    ab = read_ablations_dictionary(source)
    if key:
        print(key)
        exclude = ab[key]
    else:
        key = "dist" if distance else "full"
    predict_dir = "./results/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key)
    if not os.path.exists(predict_dir): 
            os.makedirs(predict_dir) 
    predict(task, dist, source, exclude, predict_dir)

    print("finished predicting {task}".format(task = task))
  
if __name__ == "__main__":
   main(sys.argv[1:])
