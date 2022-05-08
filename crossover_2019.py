# -*- coding: utf-8 -*-
import random
# DISCLAIMER: This code was obtained from Evolutionary Computation's practical classes material

# Recombination Operators

# Generic	
def uniform_cross(cromo_1, cromo_2,prob_cross):
	value = random.random()
	if value < prob_cross:
		f1=[]
		f2=[]
		for i in range(0,len(cromo_1)):
			if random.random() < 0.5:
				f1.append(cromo_1[i])
				f2.append(cromo_2[i])
			else:
				f1.append(cromo_2[i])
				f2.append(cromo_1[i])

		return [f1,f2]
	else:
		return [cromo_1,cromo_2]

# Permutations
# OX - order crossover

def order_cross(cromo_1,cromo_2,prob_cross):
	""" Order crossover."""
	size = len(cromo_1)
	value = random.random()
	if value < prob_cross:
		pc= random.sample(range(size),2)
		pc.sort()
		pc1,pc2 = pc
		f1 = [None] * size
		f2 = [None] * size
		f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
		f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
		for j in range(size):
			for i in range(size):
				if (cromo_2[j] not in f1) and (f1[i] == None):
					f1[i] = cromo_2[j]
					break
			for k in range(size):
				if (cromo_1[j] not in f2) and (f2[k] == None):
					f2[k] = cromo_1[j]
					break
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]
	
	
def pmx_cross(cromo_1,cromo_2,prob_cross):
	""" Partially mapped crossover."""
	size = len(cromo_1)
	value = random.random()
	if value < prob_cross:
		pc= random.sample(range(size),2)
		pc.sort()
		pc1,pc2 = pc
		f1 = [None] * size
		f2 = [None] * size
		f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
		f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
		# first offspring
		# middle part
		for j in range(pc1,pc2+1):
			if cromo_2[j] not in f1:
				pos_2 = j
				g_j_2 = cromo_2[pos_2]
				g_f1 = f1[pos_2]
				index_2 = cromo_2.index(g_f1)
				while f1[index_2] != None:
					index_2 = cromo_2.index(f1[index_2])
				f1[index_2] = g_j_2
		# remaining
		for k in range(size):
			if f1[k] == None:
				f1[k] = cromo_2[k]
		# secong offspring	
		# middle part
		for j in range(pc1,pc2+1):
			if cromo_1[j] not in f2:
				pos_1 = j
				g_j_1 = cromo_1[pos_1]
				g_f2 = f2[pos_1]
				index_1 = cromo_1.index(g_f2)
				while f2[index_1] != None:
					index_1 = cromo_1.index(f2[index_1])
				f2[index_1] = g_j_1
		# remaining
		for k in range(size):
			if f2[k] == None:
				f2[k] = cromo_1[k]				
		return [f1,f2]
	else:
		return [cromo_1,cromo_2]