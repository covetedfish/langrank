import gram2vec.lang2vec.lang2vec as l2v
import numpy as np
import pkg_resources
import os
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
import pandas as pd
import pickle
from iso639 import Lang
import defaults as defaults

TASKS = ["MT","DEP","EL","POS"]
PREFIXES = {"MT": "ted_", "DEP": "conll_", "EL":"wiki_en-", "POS": "datasets/pos/"}


MT_DATASETS = {
	"ted" : "ted.npy",
}
POS_DATASETS = {
	"ud" : "ud.npy" 
}

NEW_POS_DATASETS = {
	"ud" : "new_ud.npy" 
}

EL_DATASETS = {
	"wiki" : "wiki.npy"
}
DEP_DATASETS = {
	"conll" : "conll.npy"
}

MT_MODELS = {
	"all" : "all.lgbm",
	"geo" : "geo.lgbm",
	"aze" : "lgbm_model_mt_aze.txt",
	"fin" : "lgbm_model_mt_fin.txt",
	"ben" : "lgbm_model_mt_ben.txt",
	"best": "lgbm_model_mt_all.txt",
}
POS_MODELS = {
        "best": "lgbm_model_pos_all.txt"
}
EL_MODELS = {"best": "lgbm_model_el_all.txt"}
DEP_MODELS = {"best": "lgbm_model_dep_all.txt"}

# checks
def check_task(task):
	if task not in TASKS:
		raise Exception("Unknown task " + task + ". Only 'MT', 'DEP', 'EL', 'POS' are supported.")

def check_task_model(task, model):
	check_task(task)
	avail_models = map_task_to_models(task)
	if model not in avail_models:
		ll = ', '.join([key for key in avail_models])
		raise Exception("Unknown model " + model + ". Only "+ll+" are provided.")

def check_task_model_data(task, model, data):
	check_task_model(task, model)
	avail_data = map_task_to_data(task)
	if data not in avail_data:
		ll = ', '.join([key for key in avail_data])
		raise Exception("Unknown dataset " + data + ". Only "+ll+" are provided.")


# utils
def map_task_to_data(task, arch = "orig"):
	if task == "MT":
		return MT_DATASETS
	elif task == "POS" and arch == "orig":
		return POS_DATASETS
	elif task == "POS" and (arch == "xlmr" or source == "stanza"):
		return NEW_POS_DATASETS
	elif task == "EL":
		return EL_DATASETS
	elif task == "DEP":
		return DEP_DATASETS
	else:
		raise Exception("Unknown task")

def map_task_to_models(task):
	if task == "MT":
		return MT_MODELS
	elif task == "POS":
		return POS_MODELS
	elif task == "EL":
		return EL_MODELS
	elif task == "DEP":
		return DEP_MODELS
	else:
		raise Exception("Unknown task")

def read_vocab_file(fn):
	with open(fn) as inp:
		lines = inp.readlines()
	c = []
	v = []
	for l in lines:
		l = l.strip().split()
		if len(l) == 2:
			c.append(int(l[1]))
			v.append(l[0])
	return v,c


# used for ranking
def get_candidates(task, languages=None, arch = "orig"):
	if languages is not None and not isinstance(languages, list):
		raise Exception("languages should be a list of ISO-3 codes")

	datasets_dict = map_task_to_data(task, arch)
	cands = []
	for dt in datasets_dict:
		fn = pkg_resources.resource_filename(__name__, os.path.join('indexed', task, datasets_dict[dt]))
		d = np.load(fn, encoding='latin1', allow_pickle=True).item()
		cands += [(key,d[key]) for key in d if key != "eng"]
	# Possibly restrict to a subset of candidate languages
	if languages is not None:
		add_languages = []
		sub_languages = []
		for l in languages:
			if len(l) == 3:
				add_languages.append(l)
			else:
				# starts with -
				sub_languages.append(l[1:])
		# Keep a candidate if it matches the languages
		# -aze indicates all except aze
		print(f"len add languages: {len(add_languages)}")
		if len(add_languages) > 0:
			new_cands = []
			added = []
			for c in cands:
				if c[1]["lang"] not in added and c[1]["lang"] in add_languages and c[1]["lang"] not in sub_languages:
					if  not ((arch == "xlmr" or arch == "stanza") and "target" in c[1]): 
						new_cands.append(c)
						added.append(c[1]["lang"])
		else:
			new_cands = [c for c in cands if c[1]["lang"] not in sub_languages]
		print(f"len new_cands {len(new_cands)}")
		return new_cands

	return cands

# prepare new dataset
def prepare_new_dataset(lang, task="MT", dataset_source=None, dataset_target=None, dataset_subword_source=None, dataset_subword_target=None):
	features = {}
	features["lang"] = lang

	# Get dataset features
	if dataset_source is None and dataset_target is None and dataset_subword_source is None and dataset_subword_target is None:
		print("NOTE: no dataset provided. You can still use the ranker using language typological features.")
		return features
	elif dataset_source is None: # and dataset_target is None:
		print("NOTE: no word-level dataset provided, will only extract subword-level features.")
	elif dataset_subword_source is None: # and dataset_subword_target is None:
		print("NOTE: no subword-level dataset provided, will only extract word-level features.")


	source_lines = []
	if isinstance(dataset_source, str):
		with open(dataset_source) as inp:
			source_lines = inp.readlines()
	elif isinstance(dataset_source, list):
		source_lines = dataset_source
	else:
		raise Exception("dataset_source should either be a filnename (str) or a list of sentences.")
	'''
	if isinstance(dataset_target, str):
		with open(dataset_target) as inp:
			target_lines = inp.readlines()
	elif isinstance(dataset_target, list):
		source_lines = dataset_target
	else:
		raise Exception("dataset_target should either be a filnename (str) or a list of sentences.")
	'''
	if source_lines:
		if task != "EL":
			features["dataset_size"] = len(source_lines)
			tokens = [w for s in source_lines for w in s.strip().split()]
			features["token_number"] = len(tokens)
			types = set(tokens)
			features["type_number"] = len(types)
			features["word_vocab"] = types
			features["type_token_ratio"] = features["type_number"]/float(features["token_number"])
		elif task == "EL":
			features["dataset_size"] = len(source_lines)
			tokens = [w for s in source_lines for w in s.strip().split()]
			types = set(tokens)
			features["word_vocab"] = types

	if task == "MT":
		# Only use subword overlap features for the MT task
		if isinstance(dataset_subword_source, str):
			with open(dataset_subword_source) as inp:
				source_lines = inp.readlines()
		elif isinstance(dataset_subword_source, list):
			source_lines = dataset_subword_source
		elif dataset_subword_source is None:
			pass
			# Use the word-level info, just in case. TODO(this is only for MT)
			# source_lines = []
		else:
			raise Exception("dataset_subword_source should either be a filnename (str) or a list of sentences.")
		if source_lines:
			features["dataset_size"] = len(source_lines) # This should be be the same as above
			tokens = [w for s in source_lines for w in s.strip().split()]
			features["subword_token_number"] = len(tokens)
			types = set(tokens)
			features["subword_type_number"] = len(types)
			features["subword_vocab"] = types
			features["subword_type_token_ratio"] = features["subword_type_number"]/float(features["subword_token_number"])

	return features

def prepare_featureset(lang, task="MT", arch = "orig", target = True):
	features = {}
	datasets_dict = map_task_to_data(task, arch)
	for dt in datasets_dict:
		fn = pkg_resources.resource_filename(__name__, os.path.join('indexed', task, datasets_dict[dt]))
		features = np.load(fn, encoding='latin1', allow_pickle=True).item()

	if arch == "orig":
		code = (PREFIXES[task] + lang)
		if not code in features:
			code = PREFIXES[task] + Lang(lang).pt1
		if task == "POS":
			code = [a for a in list(features.keys()) if a.startswith(PREFIXES["POS"] + Lang(lang).pt1)]
			if len(code) > 0:
				code = code[0]
		if (code == []) or (not code in features): 
			return []
	else:
		if target:
			code = f"{lang}_target"
		else:
			code = f"{lang}_transfer"
	features = features[code]
	return features

#Prepares a train_data file consisting of feature comparison vectors between each target and transfer, 
# scores corresponding to ranking for each transfer, and
# train size corresponding to how many transfer languages make up the rankings for each target language
def prepare_train_pickle_no_data(train_langs, target_langs, rank, task="MT", tmp_dir="tmp", distances = True, exclude = [], source = "syntax_knn", dataset = "False", arch = "orig"):
	"""
	dataset: [ted_aze, ted_tur, ted_ben] 
	lang: [aze, tur, ben]
	rank: [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
	"""
	num_langs = len(train_langs)
	print(f"number of ranked train languages: {num_langs}")
	print(f"number of target languages: {len(target_langs)}")

	REL_EXP_CUTOFF = num_langs - 1 - 9

	if not isinstance(rank, np.ndarray):
		rank = np.array(rank)
	BLEU_level = -rank + len(train_langs) #invert rankings (lower score is worse)
	rel_BLEU_level = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF) #limit rankings to 1-cutoff
	target_features = {lang: prepare_featureset(lang, task, arch) for lang in target_langs}
	
	#check if we have precomputed features available for all target languages
	#we won't actually use these features because they are all dataset dependent (we only care about uriel)
	#but it helps to check if there are uriel distances precomputed because otherwise lang2vec will cry about it
	for lang in list(target_features.keys()): 
		if target_features[lang] == []:
			print(f"deleting lang: {lang}")
			del target_features[lang]

	train_features = {lang: prepare_featureset(lang, task, arch, False) for lang in train_langs}
	#ATTENTION: this should not work. if we delete a train language then the rankings are all wonky
	#check if we have precomputed features available for all train languages 
	for lang in list(train_features.keys()): 
		if train_features[lang] == []:
			raise(Exception("ERROR: all train languages should have URIEL features"))

	# all_langs = list(set(target_features.keys()).union(set(train_features.keys())))
	all_langs = list(set(target_langs).union(set(train_langs))) # sets up a common index for uriel lookups
	
	uriel = uriel_feat_vec(all_langs, distances) #creates a matrix of uriel features for each possible language pair
		
	if not os.path.exists(tmp_dir):
		os.mkdir(tmp_dir)
	scores = []
	#all this stuff saved to "{source}/{task}/{arch}/{key}/{dataset}"

	train_data = None
	train_file = os.path.join(tmp_dir, "train.pkl")
	train_file_f = open(train_file, "wb")
	train_labels = os.path.join(tmp_dir, "train_labels.pkl")
	train_labels_f = open(train_labels, "wb")
	train_size = os.path.join(tmp_dir, "train_size.csv")
	train_size_f = open(train_size, "w")

	train_len = len(train_features.keys())
	test_len = len(target_features.keys())
	for lang1 in target_langs:
		count = 0
		for lang2 in train_langs:
			i = all_langs.index(lang1) #target language index for uriel lookup
			j = all_langs.index(lang2) #transfer language index for uriel lookup
			distance_feats = {} 
			if dataset:
				#if we are using dataset features, then compute a usable feature vector comparison between target and transfer
				#dataset distance feats wil be a dictionary containing: word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size, transfer_ttr, 
				# task_ttr, distance_ttr
				distance_feats = distance_feat_dict(target_features[lang1], train_features[lang2], "POS")
			# do not write training data for target == transfer
			if i != j:
				count+=1
				if len(all_langs) == 2:
					uriel_features = uriel
				else:
					uriel_features = {u: uriel[u][i, j] for u in uriel.keys()} # gets uriel distances for each distance type in uriel
				distance_feats.update(uriel_features)
				#if we are using the full feature set (whole syntax vector) we want to add those features in
				if distances == False:
					inventory = l2v.get_feature_match_dict([lang1, lang2], "inventory_knn", exclude)
					phonological = l2v.get_feature_match_dict([lang1, lang2], "phonology_knn", exclude)
					#several options for syntax features so we feed source in here (both, none, syntax_knn, syntax, grambank)
					#still questionable on the get_feature_match_dict
					syntax_features = l2v.get_feature_match_dict([lang1, lang2], source, exclude)
					distance_feats.update(inventory)
					distance_feats.update(phonological)
					#if we are not excluding sintax features. then we add them in here
					if not syntax_features == None:
						distance_feats.update(syntax_features)
				if not train_data is None:
					#each row represents the feature comparison between the target (lang1) and the transfer (lang2)
					train_data = pd.concat([train_data, pd.DataFrame([distance_feats])], ignore_index=True)
				else:
					print(len(distance_feats))
					train_data = pd.DataFrame([distance_feats])
				score = str(rel_BLEU_level[target_langs.index(lang1), train_langs.index(lang2)]) 
				scores.append(score) #each score at index i corresponds to the feature comparison vector and index i of train_data
		train_size_f.write(f"{count}\n") #keeps track of how many languages correspond to a single target (for processing train_data and scores)
	pickle.dump(train_data, train_file_f)
	pickle.dump(scores, train_labels_f)
	train_file_f.close()
	train_size_f.close()
	print("Dump train file to {} ...".format(train_file_f))
	print("Dump train size file to {} ...".format(train_size_f))

def uriel_feat_vec(languages, distance): 
	# if distance is false, we only grab geographic and genetic distances (cannot be expanded)
	print('...geographic')
	geographic = l2v.geographic_distance(languages)
	print('...genetic')
	genetic = l2v.genetic_distance(languages)
	uriel_features = {"genetic": genetic, "geographic": geographic}
	#if we are using distances, we want to compute the remaining features here. If not, we will add them in later
	if distance:
		print('...syntactic')
		syntax = l2v.syntactic_distance(languages)
		uriel_features["syntactic"] = syntax
		print('...inventory')
		inventory = l2v.inventory_distance(languages)
		uriel_features["inventory"] = inventory
		print('...phonological')
		phonological = l2v.phonological_distance(languages)
		uriel_features["phonological"] = phonological
		print('...featural')
		featural = l2v.featural_distance(languages)
		uriel_features["featural"] = featural
	return uriel_features

def uriel_distance_vec(languages):
	# print('...grambank')
	# grambank = l2v.grambank_distance(languages)
	print('...geographic')
	geographic = l2v.geographic_distance(languages)
	print('...genetic')
	genetic = l2v.genetic_distance(languages)
	print('...inventory')
	inventory = l2v.inventory_distance(languages)
	print('...syntactic')
	syntactic = l2v.syntactic_distance(languages)
	print('...phonological')
	phonological = l2v.phonological_distance(languages)
	print('...featural')
	featural = l2v.featural_distance(languages)
	uriel_features = [genetic, syntactic, featural, phonological, inventory, geographic]
	return uriel_features


def distance_feat_dict(test, transfer, task):
	output = []
	# Dataset specific 
	# Dataset Size
	transfer_dataset_size = transfer["dataset_size"]
	task_data_size = test["dataset_size"]
	ratio_dataset_size = float(transfer_dataset_size)/task_data_size
	# TTR
	if task != "EL":
		transfer_ttr = transfer["type_token_ratio"]
		task_ttr = test["type_token_ratio"]
		distance_ttr = (1 - transfer_ttr/task_ttr) ** 2
	# Word overlap
	if task != "EL":
		word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"])))) / (transfer["type_number"] + test["type_number"])
	elif task == "EL":
		word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"]))))
	# Subword overlap
	if task == "MT":
		subword_overlap = float(len(set(transfer["subword_vocab"]).intersection(set(test["subword_vocab"])))) / (transfer["subword_type_number"] + test["subword_type_number"])

	if task == "MT":
		data_specific_features = {
			"word overlap": word_overlap, 
			"subword overlap": subword_overlap,
			"transfer_dataset_size": transfer_dataset_size,
			"task_data_size": task_data_size,
			"ratio_dataset_size": ratio_dataset_size,
			"transfer_ttr": transfer_ttr, 
			"task_ttr": task_ttr, 
			"distance_ttr": distance_ttr}
	elif task == "POS" or task == "DEP":
		data_specific_features = {
			"word overlap": word_overlap, 
			"transfer_dataset_size": transfer_dataset_size,
			"task_data_size": task_data_size,
			"ratio_dataset_size": ratio_dataset_size,
			"transfer_ttr": transfer_ttr, 
			"task_ttr": task_ttr, 
			"distance_ttr": distance_ttr}
	elif task == "EL":
			data_specific_features = {
			"word overlap": word_overlap, 
			"transfer_dataset_size": transfer_dataset_size,
			"task_data_size": task_data_size,
			"ratio_dataset_size": ratio_dataset_size}	
	return data_specific_features

def distance_vec(test, transfer, uriel_features, task):
	output = []
	# Dataset specific 
	# Dataset Size
	transfer_dataset_size = transfer["dataset_size"]
	task_data_size = test["dataset_size"]
	ratio_dataset_size = float(transfer_dataset_size)/task_data_size
	# TTR
	if task != "EL":
		transfer_ttr = transfer["type_token_ratio"]
		task_ttr = test["type_token_ratio"]
		distance_ttr = (1 - transfer_ttr/task_ttr) ** 2
	# Word overlap
	if task != "EL":
		word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"])))) / (transfer["type_number"] + test["type_number"])
	elif task == "EL":
		word_overlap = float(len(set(transfer["word_vocab"]).intersection(set(test["word_vocab"]))))
	# Subword overlap
	if task == "MT":
		subword_overlap = float(len(set(transfer["subword_vocab"]).intersection(set(test["subword_vocab"])))) / (transfer["subword_type_number"] + test["subword_type_number"])

	if task == "MT":
		data_specific_features = [word_overlap, subword_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
	elif task == "POS" or task == "DEP":
		data_specific_features = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size, transfer_ttr, task_ttr, distance_ttr]
	elif task == "EL":
		data_specific_features = [word_overlap, transfer_dataset_size, task_data_size, ratio_dataset_size]

	return np.array(data_specific_features + uriel_features)

def lgbm_rel_exp(BLEU_level, cutoff):
	if isinstance(BLEU_level, np.ndarray):
		# if BLEU_level[i,j] - cutoff +1 >= cutoff leave it be. Else, set value equal to 0
		a = np.where(BLEU_level >= cutoff, BLEU_level - cutoff + 1, 0)
		return np.where(BLEU_level >= cutoff, BLEU_level - cutoff + 1, 0)
	else:
		return BLEU_level - cutoff + 1 if BLEU_level >= cutoff else 0

def prepare_train_file_no_data(langs, rank, task="MT", tmp_dir="tmp"):
	"""
	dataset: [ted_aze, ted_tur, ted_ben] 
	lang: [aze, tur, ben]
	rank: [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
	"""
	num_langs = len(langs)
	print(f"initial number of langs {len(langs)}")

	REL_EXP_CUTOFF = num_langs - 1 - 9

	if not isinstance(rank, np.ndarray):
		rank = np.array(rank)
	BLEU_level = -rank + num_langs
	print(BLEU_level.shape)
	rel_BLEU_level = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)
	features = {lang: prepare_featureset(lang, task) for lang in langs}
	for lang in list(features.keys()): #stupid ass POS target languages aren't in the data??????
		if features[lang] == []:
			print(f"deleting {lang}")
			del features[lang]
	langs = list(features.keys())
	print(f"number of langs {len(langs)}")
	uriel = uriel_distance_vec(langs) 
	if not os.path.exists(tmp_dir):
		os.mkdir(tmp_dir)

	train_file = os.path.join(tmp_dir, "train.csv")
	train_file_f = open(train_file, "w")
	train_size = os.path.join(tmp_dir, "train_size.csv")
	train_size_f = open(train_size, "w")
	for i, lang1 in enumerate(langs):
		for j, lang2 in enumerate(langs):
			if i != j:
				if len(langs) == 2:
					uriel_features = [u for u in uriel]
				else:
					uriel_features = [u[i, j] for u in uriel] # a list of all uriel distance measures for language pair i, j
				distance_vector = np.array(uriel_features) # turn the list into an array (why?)
				# distance_vector = distance_vec(features[lang1], features[lang2], uriel_features, task)
				distance_vector = ["{}:{}".format(i, d) for i, d in enumerate(distance_vector)]
				print(f"i:{i}, j: {j}")
				line = " ".join([str(rel_BLEU_level[i, j])] + distance_vector)
				train_file_f.write(line + "\n")
		train_size_f.write("{}\n".format(num_langs-1))
	train_file_f.close()
	train_size_f.close()
	print("Dump train file to {} ...".format(train_file_f))
	print("Dump train size file to {} ...".format(train_size_f))

# preparing the file for training
def prepare_train_file(datasets, langs, rank, segmented_datasets=None, task="MT", tmp_dir="tmp"):
	"""
	dataset: [ted_aze, ted_tur, ted_ben]
	lang: [aze, tur, ben] #functions as a sort of index; could we not have 2 indices? 
	one for target languages and one for transfer languages
	rank: [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
	"""
	num_langs = len(langs) 
	REL_EXP_CUTOFF = num_langs - 1 - 9 

	if not isinstance(rank, np.ndarray):
		rank = np.array(rank)
	BLEU_level = -rank + len(langs)
	rel_BLEU_level = lgbm_rel_exp(BLEU_level, REL_EXP_CUTOFF)

	features = {}
	for i, (ds, lang) in enumerate(zip(datasets, langs)):
		with open(ds, "r") as ds_f:
			lines = ds_f.readlines()
		seg_lines = None
		if segmented_datasets is not None:
			sds = segmented_datasets[i]
			with open(sds, "r") as sds_f:
				seg_lines = sds_f.readlines()
		features[lang] = prepare_new_dataset(lang=lang, dataset_source=lines, dataset_subword_source=seg_lines)
	uriel = uriel_distance_vec(langs)

	if not os.path.exists(tmp_dir):
		os.mkdir(tmp_dir)

	train_file = os.path.join(tmp_dir, "train_mt.csv")
	train_file_f = open(train_file, "w")
	train_size = os.path.join(tmp_dir, "train_mt_size.csv")
	train_size_f = open(train_size, "w")
	for i, lang1 in enumerate(langs):
		for j, lang2 in enumerate(langs):
			if i != j:
				uriel_features = [u[i, j] for u in uriel]
				distance_vector = np.array(uriel_features)
				# distance_vector = distance_vec(features[lang1], features[lang2], uriel_features, task)
				distance_vector = ["{}:{}".format(i, d) for i, d in enumerate(distance_vector)]
				line = " ".join([str(rel_BLEU_level[i, j])] + distance_vector)
				train_file_f.write(line + "\n")
		train_size_f.write("{}\n".format(num_langs-1))
	train_file_f.close()
	train_size_f.close()
	print("Dump train file to {} ...".format(train_file_f))
	print("Dump train size file to {} ...".format(train_size_f))

def train(tmp_dir, output_model):
	train_file = os.path.join(tmp_dir, "train.csv")
	train_size = os.path.join(tmp_dir, "train_size.csv")
	X_train, y_train = load_svmlight_file(train_file)
	model = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=16,
						   max_depth=-1, learning_rate=0.1, n_estimators=100,
						   min_child_samples=5)
	model.fit(X_train, y_train, group=np.loadtxt(train_size))
	model.booster_.save_model(output_model)


#loads in train files produced with prepare_train_pickle_no_data
def train_from_pickle(tmp_dir, output_model):
	train_file = os.path.join(tmp_dir, "train.pkl")
	label_file = os.path.join(tmp_dir, "train_labels.pkl")
	train_size = os.path.join(tmp_dir, "train_size.csv")
	X_train = pickle.load(open(train_file, "rb"))
	y_train = pickle.load(open(label_file, "rb"))
	group = np.loadtxt(train_size, ndmin=1)
	print(f"shape of train data loaded in: {X_train.shape}") # (target * (transfer-target), num feats)
	print(f"length of train data loaded in: {len(y_train)}") #target * (transfer-target)
	model = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=16,
						   max_depth=-1, learning_rate=0.1, n_estimators=100,
						   min_child_samples=5) #same ranking settings as langrank
	model.fit(X_train, y_train, group=group)
	model.booster_.save_model(output_model)





def rank(test_lang, test_dataset_features = None, task="MT", candidates="all", model="best", print_topK=3, distances = True, exclude = [], source = "syntax_knn", return_langs = True, dataset = False, arch = "orig"):
	'''
	test_dataset_features : the output of prepare_new_dataset(). Basically a dictionary with the necessary dataset features.
	'''

	# Get candidates to be compared against
	#ATTENTION: NEEDS ALTERATION IF USING NEW DATASETS
	print("Preparing candidate list...")
	if candidates=='all':
		candidate_list = get_candidates(task, arch)
	else:
		# Restricts to a specific set of languages
		candidate_list = get_candidates(task, candidates, arch)
	print(f"cand list len: {len(candidate_list)}")
	#creates a dictionary where key= candidate language and value=dataset dependent feats
	features = {cand[1]['lang']: cand[1] for cand in candidate_list }
	print("Collecting URIEL distance vectors...")
	languages = [test_lang] + [c[1]["lang"] for c in candidate_list]
	# TODO: This takes forever...
	uriel = uriel_feat_vec(languages, distances)
	
	print("Collecting dataset distance vectors...")
	test_inputs = []
	test_data = None
	for i,c in enumerate(candidate_list):
		key = c[0]
		cand_dict = c[1]
		candidate_language = Lang(cand_dict['lang']).pt3
		uriel_features = {u: uriel[u][0, i+1] for u in uriel.keys()} # 0 because test lang is the 0th row
		distance_feats = {} 
		if dataset:
			if not test_dataset_features:
				raise Exception("precomputed dataset features for test input required to use dataset features")
			distance_feats = distance_feat_dict(test_dataset_features, features[candidate_language], task)
		distance_feats.update(uriel_features)

		if not distances:
			inventory = l2v.get_feature_match_dict([test_lang, candidate_language], "inventory_knn", exclude)
			phonological = l2v.get_feature_match_dict([test_lang, candidate_language], "phonology_knn", exclude)
			distance_feats.update(inventory)
			distance_feats.update(phonological)
			syntax_features = l2v.get_feature_match_dict([test_lang, candidate_language], source, exclude)
			if not syntax_features == None:
				distance_feats.update(syntax_features)

		if not test_data is None:
			test_data = pd.concat([test_data, pd.DataFrame([distance_feats])], ignore_index=True)
		else:
			test_data = pd.DataFrame([distance_feats])

	#test_data should be of length = #candidate langs (train/transfer langs)
	# # rank
	bst = lgb.Booster(model_file=model)
	key =  "no-dataset"
	if dataset: 
		key = "dataset"
	distance_key = "no-distance"
	if distances:
		distance_key = "distance"
	fname = f"{defaults.RESULTS_DIR}/importance/{key}/{distance_key}/{test_lang}_{arch}_{source}.tsv"
	importance_df = (
    pd.DataFrame({
        'feature_name': bst.feature_name(),
        'importance_gain': bst.feature_importance(importance_type='gain'),
        'importance_split': bst.feature_importance(importance_type='split'),
    })
    .sort_values('importance_gain', ascending=False)
    .reset_index(drop=True)
	)
	importance_df.to_csv(fname, sep='\t', index=False, header=False)

	print(f"test input size {test_data.shape}")
	print("predicting...")
	predict_contribs = bst.predict(test_data, pred_contrib=True)
	predict_scores = predict_contribs.sum(-1)
	

	print("Ranking with single features:")
	TOP_K=min(3, len(candidate_list))
	
	# 0 means we ignore this feature (don't compute single-feature result of it)
	if task == "MT":
		sort_sign_list = [-1, -1, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
		feature_name = ["Overlap word-level", "Overlap subword-level", "Transfer lang dataset size",
					"Target lang dataset size", "Transfer over target size ratio", "Transfer lang TTR",
					"Target lang TTR", "Transfer target TTR distance", "GENETIC", 
					"SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"] 
	elif task == "POS" or task == "DEP":
		sort_sign_list = [-1, 0, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1]
		feature_name = ["Overlap word-level", "Transfer lang dataset size", "Target lang dataset size", 
						"Transfer over target size ratio", "Transfer lang TTR", "Target lang TTR", 
						"Transfer target TTR distance", "GENETIC", "SYNTACTIC", "FEATURAL", 
						"PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
	elif task == "EL":
		sort_sign_list = [-1, -1, 0, -1, 1, 1, 1, 1, 1, 1]
		feature_name = ["Entity overlap", "Transfer lang dataset size", "Target lang dataset size", 
						"Transfer over target size ratio", "GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", 
						"INVENTORY", "GEOGRAPHIC"]
	ind = list(np.argsort(-predict_scores))
	if return_langs:
		return [candidate_list[i][1]["lang"] for j,i in enumerate(ind)]
	
	test_inputs = np.array(test_inputs)
	for j in range(len(feature_name)):
		if sort_sign_list[j] != 0:
			print(feature_name[j])
			values = test_inputs[:, j] * sort_sign_list[j]
			best_feat_index = np.argsort(values)
			for i in range(TOP_K):
				index = best_feat_index[i]
				print("%d. %s : score=%.2f" % (i, candidate_list[index][0], test_inputs[index][j]))

	ind = list(np.argsort(-predict_scores))
	print("Ranking (top {}):".format(print_topK))
	for j,i in enumerate(ind[:print_topK]):
		print("%d. %s : score=%.2f" % (j+1, candidate_list[i][0], predict_scores[i]))
		contrib_scores = predict_contribs[i][:-1]
		contrib_ind = list(np.argsort(contrib_scores))[::-1]
		print("\t1. %s : score=%.2f; \n\t2. %s : score=%.2f; \n\t3. %s : score=%.2f" % 
			  (feature_name[contrib_ind[0]], contrib_scores[contrib_ind[0]],
			   feature_name[contrib_ind[1]], contrib_scores[contrib_ind[1]],
			   feature_name[contrib_ind[2]], contrib_scores[contrib_ind[2]]))
	return ind




