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
import scipy.linalg as la

def svd(x,y,xp,yp):
	# Generate A matrix from the input
	A = np.array([[0] * 9] * 8)
	for i in range(4):          # A for loop for row entries  
		p1 = [-x[i], -y[i], -1, 0, 0, 0, x[i] * xp[i], y[i] * xp[i], xp[i]]
		p2 = [0, 0, 0, -x[i], -y[i], -1, x[i] * yp[i], y[i] * yp[i], yp[i]]
		A[2*i] = p1
		A[2*i + 1] = p2

	# Compute square matrices AA' and A'A
	r = np.matrix.transpose(A)
	s = np.matmul(A,r)
	t = np.matmul(r,A)

	# Compute left singular vector U
	eigvals1, eigvecs1 = la.eig(s)
	eigvals = eigvals1[eigvals1.argsort()[::-1]]
	temp1 = eigvecs1[:,eigvals1.argsort()[::-1]]
	eigvecs1 = temp1
	U = eigvecs1

	# Compute right singular vector V
	eigvals2, eigvecs2 = la.eig(t)
	temp2 = eigvecs2[:,eigvals2.argsort()[::-1]]
	eigvecs2 = temp2
	eigvals2 = np.real(eigvals2)
	V = eigvecs2

	Vtrans=np.matrix.transpose(V)

	# Compute singular values of A
	eigvals = np.real(eigvals)
	for i in range(len(eigvals)):
		eigvals[i] = np.sqrt(eigvals[i])

	D=np.diag(eigvals)
	temp = np.array([[0]]*8)
	D = np.append(D, temp, axis = 1)

	return U, D, Vtrans

def main():	
	np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
	x = [5,150,150,5]
	y = [5,5,150,150]
	xp = [100,200,220,100]
	yp =[100,80,80,200]
	U, D, V  = svd(x,y,xp,yp)

	print("Singular Value Decomposition of A is given as follows:")
	print("U = ")
	print(U)
	print("V' = ")
	print(V)
	print("D = ")
	print(D)

if __name__ == '__main__':
	main()









    