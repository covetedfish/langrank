import langrank as lr
import pickle
import langrank.defaults as defaults


from statistics import mean
import sklearn.metrics as sm
import numpy as np

def compute_ndcg(lang, ranked_langs, predicted, gamma_max= 10, k=3):
    ranking_langs = ranked_langs[lang][0] # list of languages for looking up index in ranking vector
    # gives position in ranking based on index (if ranking[0] = 4 then the 0th language [ranking_langs[0]] is the 5th best)
    ranking = ranked_langs[lang][1] 

    # creates vector to look up the relevance score of a given language by index
    scores_by_index = [0] * len(ranking)
    for i in range(len(ranking)): 
        if ranking[i] <= gamma_max:
            scores_by_index[i] = gamma_max - (ranking[i] -1)
    ideal_score = [i for i in reversed(range(1, gamma_max + 1))] + [0] * (len(ranking) - gamma_max)
    predicted_score = [0] * len(ranking)
    for j in range(len(predicted)): #for each language in ranking
        code = predicted[j]
        index = ranking_langs.index(code)
        score = scores_by_index[index] #finds the true relevance of each language
        predicted_score[j] = score
    return sm.ndcg_score(np.asarray([ideal_score]), np.asarray([predicted_score]),k=k)


def save_ndcg(task, predict_dir):    
    with open(predict_dir), 'rb') as f:
        predictions = pickle.load(f)

    with open(defaults.TRAIN_FILE.format(task), 'rb') as f:
        rankings = pickle.load(f)
    languages = list(rankings.keys())
    ndcg = {lang: compute_ndcg(lang, rankings, predictions[lang]) for lang in languages}
    score = str(mean(ndcg.values()))

    with open(predict_dir.replace("predictions.pkl" "ncdg.pkl"), 'wb') as f:
        pickle.dump(ndcg, f)


    return score

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
    if key:
        print(key)
        exclude = ab[key]
    else:
        key = "dist" if distance else "full"
    predict_dir = "./results/" + defaults.FILE_EXTENSION.format(task = task, source = source, key = key)
    score = save_ndcg(task, predict_dir)