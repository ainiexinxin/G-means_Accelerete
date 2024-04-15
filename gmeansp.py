import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn
import random

from sklearn import datasets
from pdb import set_trace
from model import *
import time

seed_num = 17
random_state = 1010
strictness_num = 4
exmple_nums = 10000
features_nums = 30
centers_nums = 30

def seed_globe(seed):
	np.random.seed(seed)
	random.seed(seed)


def drain(plot_data):
	title = ['best_cluster', 'gmeans', 'gmeansAcc']
	fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

	for i in range(3):
		sbn.scatterplot(x='x', y='y', data=plot_data, hue=title[i], ax=axis[i], palette='tab20_r')
		
		axis[i].set_title(title[i])
		plt.subplots_adjust(wspace=0.3, hspace=0.3)

	plt.show()


if __name__ == '__main__':
	seed_globe(seed_num)
	# iris = datasets.load_iris().data

	iris, labels = datasets.make_blobs(n_samples=exmple_nums,
		n_features=features_nums,
		centers=centers_nums)

	gmeans = GMeans_MB(random_state=random_state,
					   strictness=strictness_num)

	gmeansAcc = GMeans_Acc(random_state=random_state,
					   strictness=strictness_num)

	start1 = time.time()
	gmeans.fit(iris)
	end1 = time.time()


	# G-means with Acc
	start = time.time()
	gmeansAcc.fit(iris)
	end = time.time()

	# 画图
	plot_data = pd.DataFrame(iris[:, 0:2])
	plot_data.columns = ['x', 'y']

	plot_data['best_cluster'] = labels
	plot_data['gmeans'] = gmeans.labels_
	plot_data['gmeansAcc'] = gmeansAcc.labels_


	print("[G_means with minbatchkmeans] Time Elapsed:", end1 - start1)
	print("[G_meansAcc] Time Elapsed:", end - start)
	drain(plot_data)

