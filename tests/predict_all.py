import os
import sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import langrank as lr
import pickle
import sys, getopt
import pandas as pd
import re

PATH = "./resources/fine_categorization.csv"
def read_ablations_dictionary(path, source):
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


def predict(task, dist, source, exclude):
    with open("./training-data/{}_gram_ranked_train_no_ties.pkl".format(task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    if dist:
        path = f"./models/uriel-gram/{task}/dist/"
    else:
        path = f"./models/uriel-gram/{task}/full/"
    for lang in languages:
        lang_path = "{path}{lang}.txt".format(path = path, lang = lang)
        print(lang_path)
        cands = rankings[lang][0]
        cands.append(lang)
        prepared = lr.prepare_featureset(lang=lang, task = task)
        predicted[lang] = lr.rank(prepared, test_lang = lang, task=task, candidates=cands, model = lang_path, distances = dist, source = source, exclude = exclude)
    if dist:
        pf = "./results/{t}/{source}/dist/predictions.pkl".format(t = task, source = source)
    else:
        pf = "./results/{t}/{source}/full/predictions.pkl".format(t = task, source = source)
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)

def main(argv):
   #need to create dictionary with leave one out 
    task = ''
    dist = False
    source = "syntax_knn"
    exclude = []
    opts, args = getopt.getopt(argv,"t:dga",["task", "distance", "grambank"])
    for opt, arg in opts:
      if opt in ("-t", "--task"):
         task = arg
      if opt in ("-d", "--distance"):
         dist = True
      if opt in ("-g", "--grambank"):
        source =  "syntax_grambank"

    predict(task, dist, source, exclude)

    print("finished predicting {task}".format(task = task))
  
if __name__ == "__main__":
   main(sys.argv[1:])
