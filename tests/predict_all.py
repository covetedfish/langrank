import os
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import langrank as lr
import pickle
import sys, getopt

def predict(task, syntax_flag, ablation):
    with open("./training-data/{}_original_ranked_train_no_ties.pkl".format(task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    predicted = {}
    path = "./models/uriel/{task}/"
    for lang in languages:
        lang_path = "{path}{lang}.txt".format(path = path, lang = lang)
        print(lang_path)
        cands = rankings[lang][0]
        cands.append(lang)
        prepared = lr.prepare_featureset(lang=lang, task = task)
        predicted[lang] = lr.rank(task=task, candidates=cands, model = lang_path, distances = False)
    pf = "./results/{t}/predictions.pkl".format(t = task)
    print(pf)
    with open(pf, 'wb') as f:
        pickle.dump(predicted, f)

def main(argv):
   #need to create dictionary with leave one out 
    task = ''
    opts, args = getopt.getopt(argv,"t:",["task"])
    for opt, arg in opts:
      if opt in ("-t", "--task"):
         task = arg
    

    predict(task)

    print("finished predicting {task}".format(task = task))
  
if __name__ == "__main__":
   main(sys.argv[1:])
