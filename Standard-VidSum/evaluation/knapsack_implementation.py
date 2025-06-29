import numpy as np

def knapSack(W, wt, val, n): 
	''' 
	A Dynamic Programming based Python 
    Program for 0-1 Knapsack problem 
    Returns the maximum value that can 
    be put in a knapsack of capacity W 
	'''

	K = [[0 for _ in range(W + 1)] for _ in range(n + 1)] 

	# Build table K[][] in bottom up manner 
	for i in range(n + 1): 
		for w in range(W + 1): 
			if i == 0 or w == 0:
				K[i][w] = 0 
			elif wt[i-1] <= w: 
				K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]) 
			else: 
				K[i][w] = K[i-1][w]

	selected = []
	w = W
	for i in range(n,0,-1):
		if K[i][w] != K[i-1][w]:
			selected.insert(0,i-1)
			w -= wt[i-1]

	return selected 

def Test_Knapsack():
  # Example usage
  weights = [1, 1, 1, 1, 2, 2, 3]
  values = [1, 1, 2, 3, 1, 3, 5]
  W = 7
  selected = knapSack(W, weights, values, len(values))
  print("Selected elements :", selected)


if __name__ == "__main__":
	Test_Knapsack()