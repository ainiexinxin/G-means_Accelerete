import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn
import random

from sklearn import datasets
from pdb import set_trace
from model import *
import time


def seed_globe(seed):
	np.random.seed(seed)
	random.seed(seed)


def drain(plot_data):
	title = ['best_cluster', 'gmeans', 'gmeansAcc']
	fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

	for i in range(3):
		sbn.scatterplot(x='x', y='y', data=plot_data, hue=title[i], ax=axis[i], palette='tab20_r')
		axis[i].set_title(title[i])
		plt.subplots_adjust(wspace=0.3, hspace=0.3)

	plt.show()
	set_trace()


if __name__ == '__main__':
	seed_globe(17)
	# iris = datasets.load_iris().data

	iris, labels = datasets.make_blobs(n_samples=1000,
		n_features=5,
		centers=5)

	gmeans = GMeans_MB(random_state=1010,
					   strictness=4)

	gmeansAcc = GMeans_Acc(random_state=1010,
					   strictness=4)


	start = time.time()
	gmeans.fit(iris)
	end = time.time()

	print("[G_means with minbatchkmeans] Time Elapsed:", end - start)

	# G-means with Acc
	start = time.time()
	gmeansAcc.fit(iris)
	end = time.time()

	print("[G_meansAcc] Time Elapsed:", end - start)

	# 画图
	plot_data = pd.DataFrame(iris[:, 0:2])
	plot_data.columns = ['x', 'y']

	plot_data['best_cluster'] = labels
	plot_data['gmeans'] = gmeans.labels_
	plot_data['gmeansAcc'] = gmeansAcc.labels_

	drain(plot_data)
