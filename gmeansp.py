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
	title = ['best_cluster', 'gmeans', 'gmeansp', 'gmeanspp']
	fig, axis = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

	for i in range(4):
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

	gmeans = GMeans(random_state=1010,
					   strictness=4)

	gmeansp = GMeans_p(random_state=1010,
					   strictness=4)

	gmeanspp = GMeans_pp(random_state=1010,
					   strictness=4)

	start = time.time()
	gmeans.fit(iris)
	end = time.time()

	print("[G_means with minbatchkmeans] Time Elapsed:", end - start)


	start = time.time()
	gmeanspp.fit(iris)
	end = time.time()

	print("[G_means with kmeans] Time Elapsed:", end - start)

	start = time.time()
	gmeansp.fit(iris)
	end = time.time()

	print("[G_means+] Time Elapsed:", end - start)

	# 画图
	plot_data = pd.DataFrame(iris[:, 0:2])
	plot_data.columns = ['x', 'y']

	plot_data['best_cluster'] = labels
	plot_data['gmeans'] = gmeans.labels_
	plot_data['gmeansp'] = gmeansp.labels_
	plot_data['gmeanspp'] = gmeanspp.labels_

	drain(plot_data)
