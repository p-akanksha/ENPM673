"""
ENPM673 - Perception for Autonomous Robots
Homework 1

Author(s): 
Sri Manika Makam
M.Eng in Robotics,
University of Maryland, College Park

Achal Vyas
M.Eng in Robotics,
University of Maryland, College Park

Akanksha Patel
M.Eng in Robotics,
University of Maryland, College Park
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse
import os

def get_error(point, model):
	a = model[0]
	b = model[1]
	c = model[2]
	x1 = float(point[0])
	y1 = float(point[1])

	y2 = a*(x1**2) + b*x1 + c

	return abs(y2 - y1)

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
	plt.plot(x, y, 'r', linewidth=2)
	if not os.path.exists('Results/'):
		os.makedirs('Results/')
	plt.savefig('Results/LS-1.png')
	plt.show()
	plt.close()

def leastSquares(xi,yi):
	l = len(xi)
	Y = np.array([0] * 250)
	X = np.array([[0] * 3] * 250)

	for i in range(l):
		X[i] = [xi[i]**2, xi[i], 1]		

	for i in range(l):
		Y[i] = yi[i] 

	r = np.matrix.transpose(X)
	s = np.matmul(r,X)
	t = np.linalg.inv(s)
	u = np.matmul(r,Y)
	B = np.matmul(t,u)

	return B

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DataPath', default='/home/akanksha/Documents/ENPM673/apatel44_hw1/data_1.csv', help='Base path of csv file containing data points, Default:/home/akanksha/Documents/ENPM673/apatel44_hw1/data_1.csv')

	Args = Parser.parse_args()
	DataPath = Args.DataPath

	data = []
	x = np.array([0] * 250)
	y = np.array([0] * 250)

	with open(DataPath, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		count = 0
		for row in spamreader:
			if count == 0:
				count += 1
				continue
			data.append(row)
			x[count-1] = float(row[0])
			y[count-1] = float(row[1])
			count = count + 1

	plt.scatter(x, y)
	bestFit = leastSquares(x,y)

	plotModel(bestFit)
	err = get_updated_error(data, bestFit)
	print("Error: " + str(err))

if __name__ == '__main__':
	main()