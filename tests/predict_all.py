import os
import sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import langrank as lr
import pickle
import sys, getopt

def predict(task, dist):
    with open("./training-data/{}_original_ranked_train_no_ties.pkl".format(task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    if dist:
        path = f"./models/uriel/{task}/dist/"
    else:
        path = f"./models/uriel/{task}/full/"
    for lang in languages:
        lang_path = "{path}{lang}.txt".format(path = path, lang = lang)
        print(lang_path)
        cands = rankings[lang][0]
        cands.append(lang)
        prepared = lr.prepare_featureset(lang=lang, task = task)
        predicted[lang] = lr.rank(predicted, test_lang = lang, task=task, candidates=cands, model = lang_path, distances = dist)
    if dist:
        pf = "./results/{t}/dist/predictions.pkl".format(t = task)
    else:
        pf = "./results/{t}/full/predictions.pkl".format(t = task)
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)

def main(argv):
   #need to create dictionary with leave one out 
    task = ''
    dist = False
    opts, args = getopt.getopt(argv,"t:d",["task", "distance"])
    for opt, arg in opts:
      if opt in ("-t", "--task"):
         task = arg
      if opt in ("-d", "--distance"):
         dist = True
    

    predict(task, dist)

    print("finished predicting {task}".format(task = task))
  
if __name__ == "__main__":
   main(sys.argv[1:])
