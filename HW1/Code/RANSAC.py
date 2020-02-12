"""
ENPM673 - Perception for Autonomous Robots
Homework 1

Author(s): 
Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park

Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park

Achal Vyas
M.Eng in Robotics,
University of Maryland, College Park
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import argparse
from ls import leastSquares
import os

def get_model(points):
	'''
	y = a*x^2 + b*x + c
	'''

	x1 = float(points[0][0])
	y1 = float(points[0][1])
	x2 = float(points[1][0])
	y2 = float(points[1][1])
	x3 = float(points[2][0])
	y3 = float(points[2][1])

	d = (x1-x2) * (x1-x3) * (x2-x3);
	a = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / d;
	b = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / d;
	c = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / d;

	return [a, b, c]

def get_error(point, model):
	a = model[0]
	b = model[1]
	c = model[2]
	x1 = float(point[0])
	y1 = float(point[1])

	y2 = a*(x1**2) + b*x1 + c

	return abs(y2 - y1)

def get_updated_model(points):
	x = []
	y = []
	for p in points:
		x.append(float(p[0]))
		y.append(float(p[1]))
	
	# z = np.polyfit(x, y, 2)
	z = leastSquares(x, y)

	a = z[0]
	b = z[1]
	c = z[2]

	return [a, b, c]

def get_updated_error(points, model):
	err = 0
	for p in points:
		err = err + get_error(p, model)

	err = err/len(points)

	return err

def plotModel(model):
	x = np.linspace(0,500,1000)
	a = model[0]
	b = model[1]
	c = model[2]
	y = a*(x**2) + b*x + c
	# y = x**2
	plt.plot(x, y, 'r', linewidth=2)
	if not os.path.exists('Results/'):
		os.makedirs('Results/')
	plt.savefig('Results/RANSAC-2.png')
	plt.show()
	plt.close()

def RANSAC(data, n, k, t, d):
	"""
	data – A set of observations.
    model – A model to explain observed data points.
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.
	"""

	count = 0
	bestFit = None
	bestErr = float('inf')

	while count < k:
		rand_points = random.sample(data, 3)
		model = get_model(rand_points)
		inliers = []
		for dt in data:
			err = get_error(dt, model)
			if err < t:
				inliers.append(dt)

		if len(inliers) > d:
			model_u = get_updated_model(inliers)
			err_u = get_updated_error(inliers, model_u)

			if err_u < bestErr:
				print(len(inliers))
				bestErr = err_u
				bestFit = model_u

		count = count + 1

	return bestFit, bestErr


def main():

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DataPath', default='/home/akanksha/Documents/ENPM673/apatel44_hw1/data_2.csv', help='Base path of csv file containing data points, Default:/home/akanksha/Documents/ENPM673/apatel44_hw1/data_2.csv')
	Parser.add_argument('--k', type=int, default=100, help='Number of iterations, Default:100')
	Parser.add_argument('--t', type=int, default=40, help='Threshold value that deternime that a data point is inlier or outlier, Default:40')
	Parser.add_argument('--d', type=int, default=175, help='Minimum number of inliers for a model to be selected as a goof fit, Default:175')

	Args = Parser.parse_args()
	DataPath = Args.DataPath
	k = Args.k
	t = Args.t
	d = Args.d

	n = 3

	data = []
	x = []
	y = []

	with open(DataPath, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		count = 0
		for row in spamreader:
			if count == 0:
				count += 1
				continue
			data.append(row)
			x.append(row[0])
			y.append(row[1])
			count = count + 1

	plt.scatter(x, y)

	bestFit, bestErr = RANSAC(data, n, k, t, d)

	plotModel(bestFit)
	print(bestFit)
	print("Lowest Error: " + str(bestErr))

	err = get_updated_error(data, bestFit)
	print(err)

if __name__ == '__main__':
    main()
 