import sys, getopt

import os, sys
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from langrank import prepare_train_file_no_data, train_from_pickle, prepare_train_pickle_no_data, train
import pickle 

# def read_ablations_dictionary(path):
    

def train_one_distanceless(dir_path, task, langs, rank, test_lang):
    prepare_train_pickle_no_data(langs=langs, rank=rank, tmp_dir=dir_path, task = task, distances = False)
    output_model = "{}/{}.txt".format(dir_path,test_lang)
    train_from_pickle(tmp_dir= dir_path, output_model=output_model)
    assert os.path.isfile(output_model)

def train_one_distances(dir_path, task, langs, rank, test_lang):
    prepare_train_file_no_data(langs=langs, rank=rank, tmp_dir=dir_path, task = task)
    output_model = "{}/{}.txt".format(dir_path,test_lang)
    train(tmp_dir= dir_path, output_model=output_model)
    assert os.path.isfile(output_model)

def main(argv):
 
    task = ''
    dist = False
    opts, args = getopt.getopt(argv,"t:d",["task", "distances"])
    for opt, arg in opts:
      if opt in ("-t", "--task"):
         task = arg
    if opt in ("-d", "--distances"):
         dist = True

    t_file = "./training-data/{task}_gram_ranked_train_no_ties.pkl".format(task = task)
    print(t_file)
                                                            
    with open(t_file, 'rb') as f:
        training= pickle.load(f)
    if dist:
        model_dir = "./models/uriel/{task}/dist".format(task = task)
    else: 
        model_dir = "./models/uriel/{task}/full".format(task = task)

    model_langs = list(training.keys())
    for lang in model_langs:
        print(lang)
        languages = list(training[lang][1].keys())
        rank = list(training[lang][1].values())
        if dist:
            train_one_distances(model_dir, task, languages, rank, lang)
        else:
            train_one_distanceless(model_dir, task, languages, rank, lang)
    
    print("finished training {task}".format(task = task))

if __name__ == "__main__":
   main(sys.argv[1:])
