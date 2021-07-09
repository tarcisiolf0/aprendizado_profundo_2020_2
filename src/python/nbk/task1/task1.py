#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:47:36 2021

@author: tarcisio
"""


#%%
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polygrid2d 
from numpy.polynomial.polynomial import polygrid3d 
from PIL import Image 
from matplotlib.image import imread 

#%% Questions Numpy Array

#How to create an empty and a full NumPy array?
empty_array = np.empty([5,2])
print("Q1\n",empty_array)
full_array = np.full([5,2], 10)
print("Q1\n",full_array)

#Create a Numpy array filled with all zeros
zeros_array = np.zeros((5,2))
print("Q2\n",zeros_array)

#Create a Numpy array filled with all ones
ones_array = np.ones((5,2))
print("Q3\n",ones_array)

#Check whether a Numpy array contains a specified row
array_q4 = np.array([1 , 2 ,3])
print("Q4\n",[1,2,3,4,5] in array_q4.tolist())

#How to Remove rows in Numpy array that contains non-numeric values?
array_q5 = np.array([[1 , 2 ,3], 
                  [41, np.nan, np.nan]])
print("Q5")
print(array_q5)
array_q5 = array_q5[~np.isnan(array_q5).any(axis=1)]
print(array_q5) #np.isnan shows which elements are Not a Number

#Remove single-dimensional entries from the shape of an array
array_q6 = np.zeros((3, 1, 4)) # 3 arrays with 1 row and 4 collumns
print("Q6\n", array_q6)
print(np.squeeze(array_q6).shape)

#Find the number of occurrences of a sequence in a NumPy array

array_q7 = np.array([[1, 2, 9, 4],  
                   [3, 4, 9, 4], 
                   [5, 6, 9, 7], 
                   [7, 9, 4, 3]])

result = repr(array_q7).count("9, 4")
print("Q7\n", result)

#Find the most frequent value in a NumPy array
array_q8 = np.array([1,2,3,4,5,1,2,1,1,1]) 
print("Q8\n",np.bincount(array_q8).argmax()) #Count number of occurrences of each value in array of non-negative ints.

#Combining a one and a two-dimensional NumPy Array
array1_q9 = np.arange(5) 
array2_q9 = np.arange(10).reshape(2,5)

print("Q9")
for a, b in np.nditer([array1_q9, array2_q9]): #Efficient multi-dimensional iterator object to iterate over arrays.
    print(a,":",b)

#How to build an array of all combinations of two NumPy arrays?
print("Q10")
array_q10 = np.array(np.meshgrid([1, 2, 3], [4, 5])).T.reshape(-1,2) 
# Join a sequence of arrays along a new axis.
# Return coordinate matrices from coordinate vectors.
print(array_q10)

#How to add a border around a NumPy array?
print("Q11")
array_q11 = np.ones((2, 2)) 

array_q11 = np.pad(array_q11, pad_width=1, mode='constant', constant_values=0) #Pad an array.

print(array_q11)

#How to compare two NumPy arrays?
print("Q12")
array1_q12 = np.array([[1, 2], [3, 4]]) 
array2_q12 = np.array([[1, 2], [3, 4]]) 

arrayr_q12 = np.array_equal(array1_q12,array2_q12) #True if two arrays have the same shape and elements, False otherwise.
print(arrayr_q12)

#How to check whether specified values are present in NumPy array?
print("Q13")
array_q13 = np.array([[2, 3, 0], [4, 1, 6]]) 
print(2 in array_q13)
print(10 in array_q13) 

# How to get all 2D diagonals of a 3D NumPy array?
print("Q14")
array_q14 = np.arange(3*4*5).reshape(3,4,5)
print(array_q14)
array_q14 = np.diagonal(array_q14, axis1=1, axis2=2) # Return specified diagonals.
print(array_q14)

# Flatten a Matrix in Python using NumPy
print("Q15")
array_q15 = np.array([[2, 3], [4, 5]]) 
print(array_q15)
array_q15 = array_q15.flatten() #Return a copy of the array collapsed into one dimension.
print(array_q15)

# Flatten a 2d numpy array into 1d array
print("Q16")
array_q16 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]) 
print(array_q16)
array_q16 = array_q16.ravel() # Return a contiguous flattened array
print(array_q16)

#Move axes of an array to new positions
print("Q17")
array_q17 = np.zeros((2, 3, 4)) 
print(array_q17)
print(np.moveaxis(array_q17, 0, -1).shape) # Move axes of an array to new positions.
print(np.moveaxis(array_q17, -1, 0).shape)

#Interchange two axes of an array
print("Q18")
array1_q18 = np.array([[1,2,3]])
print(array1_q18)
array2_q18 =  np.swapaxes(array1_q18,0,1) # Interchange two axes of an array.
print(array2_q18)

#NumPy – Fibonacci Series using Binet Formula
print("Q19")
a = np.arange(1, 11) 
lengthA = len(a) 
  
# splitting of terms for easiness 
sqrtFive = np.sqrt(5) 
alpha = (1 + sqrtFive) / 2
beta = (1 - sqrtFive) / 2
  
# Implementation of formula 
# np.rint is used for rounding off to integer 
Fn = np.rint(((alpha ** a) - (beta ** a)) / (sqrtFive)) #Round elements of the array to the nearest integer.
print("The first {} numbers of Fibonacci series are {} . ".format(lengthA, Fn))

# Counts the number of non-zero values in the array
print("Q20")
array_q20 = [[0, 1, 2, 3, 0], [0, 5, 6, 0, 7]]
result = np.count_nonzero(array_q20)  #Counts the number of non-zero values in the array a.
print(result)

# Count the number of elements along a given axis
print("Q21")
array_q21 = np.array([[2, 3], [4, 5]]) 
result = np.size(array_q21, axis=1) # Number of elements in the array.
print(result)

# Trim the leading and/or trailing zeros from a 1-D array
print("Q22")
array_q22 = np.array((0, 0, 0, 0, 1, 5, 7, 0, 6, 2, 9, 0, 10, 0, 0))
print(array_q22)
result = np.trim_zeros(array_q22) # Trim the leading and/or trailing zeros from a 1-D array or sequence.
print(result)

# Change data type of given numpy array
print("Q23")
array_q23 = np.array([10, 20, 30, 40, 50]) 
print(array_q23)
print(array_q23.dtype)
array_q23 = array_q23.astype('float64') # Copy of the array, cast to a specified type.
print(array_q23)
print(array_q23.dtype)

# Reverse a numpy array
print("Q24")
array_q24 = np.array([10, 20, 30, 40, 50]) 
print(array_q24)
result = np.flip(array_q24, axis=0)
print(result)

# How to make a NumPy array read-only?
print("Q25")
array_q25 = np.zeros(11) 
array_q25[0] = 1
print(array_q25)
array_q25.flags.writeable = False
array_q25[0] = 10

#%% Questions on Numpy Matrix

# Get the maximum value from given matrix
print("Q1")
x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
print(x.max())

# Get the minimum value from given matrix
print("Q2")
x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
print(x.min())

# Find the number of rows and columns of a given matrix using NumPy
print("Q3")
x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
print(x.shape)

# Select the elements from a given matrix
print("Q4")
x = [4, 3, 5, 7, 6, 8]
indices = [0, 1, 4]
result = np.take(x, indices)
print(result)

# Find the sum of values in a matrix
print("Q5")
x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
print(x.sum())

# Calculate the sum of the diagonal elements of a NumPy array
print("Q6")
x = np.array([[55, 25, 15, 41], 
                    [30, 44, 2, 54], 
                    [11, 45, 77, 11], 
                    [11, 212, 4, 20]])
print(x)
result = np.trace(x)
print(result)

# Adding and Subtracting Matrices in Python
print("Q7")
x = np.array([[1, 2], [4, 5]])
y = np.array([[7, 8], [9, 10]])

print(x)
print(y)
print(np.add(x,y))
print(np.subtract(x,y))

# Ways to add row/columns in numpy array
print("Q8")
x = np.array([[1, 2, 3], [45, 4, 7], [9, 6, 10]])

# Array to be added as column
column_to_be_added = np.array([1, 2, 3])

# Adding column to numpy array
result = np.hstack((x, np.atleast_2d(column_to_be_added).T))

# np.hstack - Stack arrays in sequence horizontally (column wise).
# np.atleast_2d - View inputs as arrays with at least two dimensions.

print(result)

# Matrix Multiplication in NumPy
print("Q9")
a = np.array([[1, 0],
              [0, 1]])
b = np.array([[4, 1],
              [2, 2]])
r1 = np.dot(a, b) # Dot product of two arrays
r2 = np.matmul(a, b) # Matrix product of two arrays.
print(r1)
print(r2)

# Get the eigen values of a matrix
print("Q10")
w, v = np.linalg.eig(np.diag((1, 2, 3))) # Compute the eigenvalues and right eigenvectors of a square array.
print(w)
print(v)

# How to Calculate the determinant of a matrix using NumPy?
print("Q11")
x = np.matrix([[50, 29], [30, 44]]) 
print(x)
det = np.linalg.det(x) 
print(det)

# How to inverse a matrix using NumPy
print("Q12")
x = np.array([[1,2],[3,4]]) 
y = np.linalg.inv(x) 

print (x) 
print (y) 
print (np.dot(x,y))

# How to count the frequency of unique values in NumPy array?
print("Q13")
x = np.array([1, 1, 2, 3, 4, 4, 1])
(unique, counts) = np.unique(x, return_counts=True) # Find the unique elements of an array.
frequencies = np.asarray((unique, counts)).T # Convert the input to an array.
print(frequencies)

# Multiply matrices of complex numbers using NumPy in Python
print("Q14")

# Return the dot product of two vectors. The vdot(a, b) function handles complex numbers differently than dot(a, b)
print("Q15")
x = np.array([2+3j, 4+5j]) 
y = np.array([8+7j, 5+6j]) 
z = np.vdot(x, y) 
print(z)

x = np.array([[2+3j, 4+5j], [4+5j, 6+7j]]) 
y = np.array([[8+7j, 5+6j], [9+10j, 1+2j]])
z = np.vdot(x, y) 
print(z)

# Compute the outer product of two given vectors using NumPy in Python
print("Q16")
x = np.array([6,2]) 
y = np.array([2,5]) 
result = np.outer(x,y) 
print(result)

# Calculate inner, outer, and cross products of matrices and vectors using NumPy
print("Q17")

#Inner Product of Matrices
x = np.matrix([[2, 3, 4], [3, 2, 9]]) 
y = np.matrix([[1, 5, 0], [5, 10, 3]]) 
result = np.inner(x, y) # Compute the inner product of two vectors. 
print(result)

#Outer Product of Matrices
x = np.matrix([[3, 6, 4], [9, 4, 6]]) 
y = np.matrix([[1, 15, 7], [3, 10, 8]]) 
result = np.outer(x, y) # Compute the outer product of two vectors.
print(result)

#Cross Product of Matrices
x = np.matrix([[2, 6, 9], [2, 7, 3]])
y = np.matrix([[7, 5, 6], [3, 12, 3]])
result = np.cross(x, y) #Compute the cross product of two vectors.
print(result)

# Compute the covariance matrix of two given NumPy arrays
print("18")
x = np.matrix([0, 1, 2])
y = np.matrix([2, 1, 0])

result = np.cov(x,y) # Estimate a covariance matrix, given data and weights.
print(result)

# Convert covariance matrix to correlation matrix using Python
print("19")
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance)) #Return specified diagonals. # Return the non-negative square-root of an array
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

result1 = correlation_from_covariance(result)
print(result1)

# Compute the Kronecker product of two mulitdimension NumPy arrays
print("Q20")

array1 = np.matrix([[1, 2], [3, 4]]) 
print('Array1:\n', array1) 
  
array2 = np.matrix([[5, 6], [7, 8]]) 
print('\nArray2:\n', array2) 
  
# Computing the Kronecker Product 
kroneckerProduct = np.kron(array1, array2) # Kronecker product of two arrays.
print(kroneckerProduct)

# Convert the matrix into a list
print("Q21")

x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
x = x.tolist()
print(x)

#%% Questions on Numpy Indexing


# Replace NumPy array elements that doesn’t satisfy the given condition
print("Q1")
# Vector
x =  np.array([75.42436315, 42.48558583, 60.32924763]) 
print(x) 
print("\nReplace all elements of array which are greater than 50. to 15.50") 
x[x > 50.] = 15.50
print(x)

# 2d Numpy Array
x = np.array([[45.42436315, 52.48558583, 10.32924763], 
                  [5.7439979, 50.58220701, 25.38213418]])

print(x)
print("\nReplace all elements of array which are greater than 30. to 5.25")
x[x > 30.] = 5.25
print(x)        

# 3d Numpy Array
x = np.array([[[11, 25.5, 70.6], [30.9, 45.5, 55.9], [20.7, 45.8, 7.1]], 
                  [[50.1, 65.9, 8.2], [70.4, 85.8, 10.3], [11.3, 22.2, 33.6]], 
                  [[19.9, 69.7, 36.8], [1.2, 5.1, 24.4], [4.9, 20.8, 96.7]]]) 
print(x)
print("\nReplace all elements of array which are less than 10 to Nan")              
x[x < 10.] = np.nan
print(x)  

# Return the indices of elements where the given condition is satisfied
print("Q2")
a = np.array([[1, 2, 3], [4, 5, 6]]) 
print(a) 
print ('Indices of elements <4') 
b = np.where(a<4) # Return elements chosen from x or y depending on condition.
print(b) 
print("Elements which are <4") 
print(a[b]) 

# Replace NaN values with average of columns
print("Q3")
a = np.array([  [ 0.93230948, np.nan, 0.47773439, 0.76998063],
                [ 0.94460779, 0.87882456, 0.79615838, 0.56282885],
                [ 0.94272934, 0.48615268, 0.06196785, np.nan],
                [ 0.64940216, 0.74414127, np.nan, np.nan]])

#Obtain mean of columns as you need, nanmean is convenient.
col_mean = np.nanmean(a, axis=0)
print(col_mean)
#Find indices that you need to replace
inds = np.where(np.isnan(a))
#Place column means in the indices. Align the arrays using take
a[inds] = np.take(col_mean, inds[1])
print(a)

# Replace negative value with zero in numpy array 
print("Q4")
x = np.array([1, 2, -3, 4, -5, -6]) 
result = np.where(x<0, 0, x) 
print(result)

# How to get values of an NumPy array at certain index positions?
print("Q5")

# 1-D array 
x1 = np.array([11, 10, 22, 30, 33]) 
print("Array 1 :") 
print(x1) 
  
x2 = np.array([1, 15, 60]) 
print("Array 2 :") 
print(x2) 
  
print("\nTake 1 and 15 from Array 2 and put them in\ 1st and 5th position of Array 1") 
  
x1.put([0, 4], x2) 
  
print("Resultant Array :") 
print(x1) 

# Creating 2-D Numpy array 
x2 = np.array([[11, 10, 22, 30], 
               [14, 58, 88, 100]]) 
  
print("Array 1 :") 
print(x2) 
  
y2 = np.array([1, 15, 6, 40]) 
print("Array 2 :") 
print(y2) 
  
print("\nTake 1, 15 and 6 from Array 2 and put them in 1st,\ 4th and 7th positions of Array 1") 
  
x2.put([0, 3, 6], y2) 
  
print("Resultant Array :") 
print(x2) 

# Creating 3-D Numpy array 
x3 = np.array([[[11, 25, 7], [30, 45, 55], [20, 45, 7]], 
               [[50, 65, 8], [70, 85, 10], [11, 22, 33]], 
               [[19, 69, 36], [1, 5, 24], [4, 20, 9]]]) 
  
  
print("Array 1 :") 
print(x3) 
  
# Creating 2-D array 
y2 = np.array([[1, 15, 10], 
               [6, 40, 50], 
               [11, 5, 10]]) 
  
print("\nArray 2 :") 
print(y2) 
  
print("\nTake 1, 15, 10, 6, 40 and 50 from Array 2 and put\ them in 1st, 3rd, 5th, 9th, 11th and 15th positions of Array 1") 
  
x3.put([0, 2, 4, 8, 10, 14], y2) 
  
print("Resultant Array :") 
print(x3) 

#Find indices of elements equal to zero in a NumPy array
print("Q6")
x = np.array([1, 0, 2, 0, 3, 0, 0, 5, 6, 7, 5, 0, 8]) 
print("\nIndices of elements equal to zero of the \ given 1-D array:") 
result = np.where(x == 0)[0] 
print(result) 

# How to Remove columns in Numpy array that contains non-numeric values?
print("Q7")

x = np.array([[10.5, 22.5, np.nan], 
                  [41, 52.5, np.nan]]) 

print("\nRemove all columns containing non-numeric elements ") 
print(x[:, ~np.isnan(x).any(axis=0)])  # test element-wise for NaN and return result as a boolean array.

# How to access different rows of a multidimensional NumPy array?
print("Q8")
x = np.array([[10, 20, 30],  
                [40, 5, 66],  
                [70, 88, 94]]) 
  
print("Given Array :") 
print(x) 
  
# Access the First and Last rows of array 
result = x[[0,2]]
print("\nAccessed Rows :") 
print(result)

# Get row numbers of NumPy array having element larger than X
print("Q9")
x = np.array([[1, 2, 3, 4, 5], 
                  [10, -3, 30, 4, 5], 
                  [3, 2, 5, -4, 5], 
                  [9, 7, 3, 6, 5]  
                 ]) 
  
# declare specified value 
y = 6
  
# view array 
print("Given Array:\n", x) 
  
# finding out the row numbers 
output = np.where(np.any(x > y, axis = 1)) 
  
# view output 
print("Result:\n", output)

# Get filled the diagonals of NumPy array
print("Q9")
a = np.zeros((3, 3), int)
print(a)
b = np.fill_diagonal(a, 6) # Fill the main diagonal of the given array of any dimensionality.
print(b)

# Check elements present in the NumPy array
print("Q10")
element = 2*np.arange(4).reshape((2, 2))
print(element)
test_elements = [1, 2, 4, 8]
mask = np.isin(element, test_elements)

# Calculates element in test_elements, broadcasting over element only. Returns a boolean array 
# of the same shape as element that is True where an element of element is in test_elements and False otherwise.
print(mask)

# Combined array index by index
print("Q11")
x = np.arange(10,1,-1)
array = x[np.array([3, 3, 1, 8])]
print(array)

#%% Questions on NumPy Linear Algebra

# Find a matrix or vector norm using NumPy
print("Q1")
x = np.arange(10) 
# compute norm of vector 
vec_norm = np.linalg.norm(x) # Matrix or vector norm.
print("Vector norm:") 
print(vec_norm) 

# Calculate the QR decomposition of a given matrix using NumPy
print("Q2")
x = np.array([[1, 2, 3], [3, 4, 5]]) 
print(x) 
  
# Decomposition of the said matrix 
q, r = np.linalg.qr(x) # Compute the qr factorization of a matrix. Factor the matrix
# a as qr, where q is orthonormal and r is upper-triangular.
print('\nQ:\n', q) 
print('\nR:\n', r) 

# Compute the condition number of a given matrix using NumPy
print("Q3")
x = np.array([[4, 2], [3, 1]]) 
print("Original matrix:") 
print(x) 
  
# Output 
result =  np.linalg.cond(x) 
# Compute the condition number of a matrix. This function is capable of returning the
# condition number using one of seven different norms, depending on the value of p
print("Condition number of the matrix:") 
print(result) 

# Compute the eigenvalues and right eigenvectors of a given square array using NumPy?
print("Q4")

# create numpy 2d-array 
m = np.array([[1, 2], 
              [2, 3]]) 
  
print("Printing the Original square array:\n", m) 

# finding eigenvalues and eigenvectors 
w, v = np.linalg.eig(m) 
  
# printing eigen values 
print("Printing the Eigen values of the given square array:\n", w) 
  
# printing eigen vectors 
print("Printing Right eigenvectors of the given square array:\n", v)

# create numpy 2d-array 
m = np.array([[1, 2, 3], 
              [2, 3, 4], 
              [4, 5, 6]]) 
  
print("Printing the Original square array:\n", m) 
  
# finding eigenvalues and eigenvectors 
w, v = np.linalg.eig(m) 
  
# printing eigen values 
print("Printing the Eigen values of the given square array:\n", w) 
  
# printing eigen vectors 
print("Printing Right eigenvectors of the given square array:\n", v) 

# Calculate the Euclidean distance using NumPy
print("Q5")
point1 = np.array((1, 2, 3)) 
point2 = np.array((1, 1, 1)) 
  
# calculating Euclidean distance 
# using linalg.norm() 
dist = np.linalg.norm(point1 - point2) 
  
# printing Euclidean distance 
print(dist) 

#%% Questions on NumPy Random

# Create a Numpy array with random values
print("Q1")
x = np.random.rand(3,2) # Random values in a given shape.
print(x)

# How to choose elements from the list with different probability using NumPy?
print("Q2")
num_list = [10, 20, 30, 40, 50] 
  
# uniformly select any element  rom the list 
number = np.random.choice(num_list) # Generates a random sample from a given 1-D array
print(number)

# How to get weighted random choice in Python?
print("Q3")
sampleList = [100, 200, 300, 400, 500] 
  
randomList = random.choices(sampleList, weights=(10, 20, 30, 40, 50), k=5) 
  
print(randomList) 

# Generate Random Numbers From The Uniform Distribution using NumPy
print("Q4")
# numpy.random.uniform() method 
r = np.random.uniform(size=4) # Draw samples from a uniform distribution.
  
# printing numbers 
print(r)

# Get Random Elements form geometric distribution
print("Q5")
# Using geometric() method 
gfg = np.random.geometric(0.65, 1000) # Draw samples from the geometric distribution.
  
count, bins, ignored = plt.hist(gfg, 40, density = True) 
plt.show()

# Get Random elements from Laplace distribution
print("Q6")
# Using numpy.random.laplace() method 
gfg = np.random.laplace(1.45, 15, 1000)
# Draw samples from the Laplace or double exponential
#distribution with specified location (or mean) and scale (decay).
  
count, bins, ignored = plt.hist(gfg, 30, density = True) 
plt.show()

# Return a Matrix of random values from a uniform distribution
print("Q7")
x = np.random.uniform(-1,0, size=(2,3))
print(x)

# Return a Matrix of random values from a Gaussian distribution
print("Q8")
mu, sigma = 0, 0.1 # mean and standard deviation
x = np.random.normal(mu, sigma, size=(2,3))
print(x)

#%% Questions on NumPy Sorting and Searching

# How to get the indices of the sorted array using NumPy in Python?
print("Q1")
array = np.array([10, 52, 62, 16, 16, 54, 453]) 
print(array) 
  
# Indices of the sorted elements of a given array 
indices = np.argsort(array) # Returns the indices that would sort an array.
print(indices) 

# Finding the k smallest values of a NumPy array
print("Q2")
arr = np.array([23, 12, 1, 3, 4, 5, 6]) 
print("The Original Array Content") 
print(arr) 
  
# value of k 
k = 4
  
# sorting the array 
arr1 = np.sort(arr) # Return a sorted copy of an array.
  
# k smallest number of array 
print(k, "smallest elements of the array") 
print(arr1[:k])

# How to get the n-largest values of an array using NumPy?
print("Q3")
# create numpy 1d-array 
arr = np.array([2, 0,  1, 5, 
                4, 1, 9]) 
  
print("Given array:", arr) 
  
# sort an array in ascending order 
# np.argsort() return array of indices for sorted array 
sorted_index_array = np.argsort(arr) 
  
# sorted array 
sorted_array = arr[sorted_index_array] 
print("Sorted array:", sorted_array) 
  
# we want 1 largest value 
n = 1
  
# we are using negative indexing concept
# take n largest value 
rslt = sorted_array[-n : ] 
  
# show the output 
print("{} largest value:".format(n), rslt[0]) 

# Sort the values in a matrix
print("Q4")
a = np.array([[1,4], [3,1]])
a = np.sort(a, axis=1)
print(a)

# Filter out integers from float numpy array
print("Q5")
ini_array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0]) 
  
# printing initial array 
print ("initial array : ", str(ini_array)) 
  
# filtering integers 
result = ini_array[~np.equal(np.mod(ini_array, 1), 0)] 
  
# printing resultant 
print ("final array : ", str(result)) 

# filtering integers 
mask = np.isclose(ini_array, ini_array.astype(int)) 
result = ini_array[~mask] 
  
# printing resultant 
print ("final array : ", str(result)) 

# Find the indices into a sorted array
print("Q6")
x = np.array([1,2,3,100,5])
result = np.argsort(x)
print(result)

#%% Questions on NumPy Mathematics

# How to get element-wise true division of an array using Numpy?
print("Q1")
x = np.arange(5) 
print("Original array:", x) 
  
# apply true division  on each array element 
result = np.true_divide(x, 4) 
print("After the element-wise division:", result)

# How to calculate the element-wise absolute value of NumPy array?
print("Q2")

x = np.array([1, -2, 3])  
print("Given array:\n", x) 
  
# find element-wise absolute value 
result = np.absolute(x) 
print("Absolute array:\n", result)

# Compute the negative of the NumPy array
print("Q3")

x = np.array([[2, -7, 5], [-6, 2, 0]])  
print ("Input array : ", x)  
    
result = np.negative(x)  # Numerical negative, element-wise.
print ("negative of array elements: ", result) 

# Multiply 2d numpy array corresponding to 1d array
print("Q4")

x = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]) 
y = np.array([0, 2, 3]) 
  
# printing initial arrays 
print("initial array", str(x)) 
  
# Multiplying arrays 
result = x * y[:, np.newaxis] 
  
# printing result 
print("New resulting array: ", result) 

# Computes the inner product of two arrays
print("Q5")

a = np.array([1,2,3])
b = np.array([0,1,0])
r = np.inner(a, b)
print(r)

# Compute the nth percentile of the NumPy array

#1-D
arr = [20, 2, 7, 1, 34]
print("arr : ", arr) 
print("50th percentile of arr : ", np.percentile(arr, 50))
print("25th percentile of arr : ", np.percentile(arr, 25))
print("75th percentile of arr : ", np.percentile(arr, 75))

#2-D
arr = [[14, 17, 12, 33, 44],  
       [15, 6, 27, 8, 19], 
       [23, 2, 54, 1, 4,]] 
print("\narr : \n", arr) 
    
# Percentile of the flattened array 
print("\n50th Percentile of arr, axis = None : ", np.percentile(arr, 50)) 
print("0th Percentile of arr, axis = None : ", np.percentile(arr, 0)) 
    
# Percentile along the axis = 0 
print("\n50th Percentile of arr, axis = 0 : ", np.percentile(arr, 50, axis =0)) 
print("0th Percentile of arr, axis = 0 : ", np.percentile(arr, 0, axis =0)) 

#2-D

arr = [[14, 17, 12, 33, 44],  
       [15, 6, 27, 8, 19], 
       [23, 2, 54, 1, 4,]] 
print("\narr : \n", arr) 
 
# Percentile along the axis = 1 
print("\n50th Percentile of arr, axis = 1 : ", np.percentile(arr, 50, axis =1)) 
print("0th Percentile of arr, axis = 1 : ", np.percentile(arr, 0, axis =1)) 
  
print("\n0th Percentile of arr, axis = 1 : \n", np.percentile(arr, 50, axis =1, keepdims=True))
print("\n0th Percentile of arr, axis = 1 : \n", np.percentile(arr, 0, axis =1, keepdims=True))

# Calculate the n-th order discrete difference along the given axis
print("Q6")
x = np.array([1, 2, 4, 7, 0])
r1 = np.diff(x) # Calculate the n-th order discrete difference along given axis.
print(r1)
r2 = np.diff(x, n=2)
print(r2)

# Calculate the sum of all columns in a 2D NumPy array
print("Q7")

num = np.arange(36)
arr1 = np.reshape(num, [4, 9])
print("Original array:")
print(arr1)
result  = arr1.sum(axis=0)
print("\nSum of all columns:")
print(result)

# Calculate average values of two given NumPy arrays
print("Q8")

arr1 = np.array([3, 4]) 
arr2 = np.array([1, 0]) 
  
# find average of NumPy arrays 
avg = (arr1 + arr2) / 2
print("Average of NumPy arrays:\n", avg)

# How to compute numerical negative value for all elements in a given NumPy array?
print("Q9")

x = np.array([-1, -2, -3, 1, 2, 3, 0]) 
print("Printing the Original array:", x) 
  
# converting array elements to its corresponding negative value 
r1 = np.negative(x) 
print("Printing the negative value of the given array:", r1)

# How to get the floor, ceiling and truncated values of the elements of a numpy array?
print("Q10")
a = np.array([1.2]) 
  
# Get floor value 
a = np.floor(a) 
print(a)

b = np.array([1.2]) 

# Get ceil value 
b = np.ceil(b) 
print(b) 

c = np.array([1.2]) 
  
# Get truncate value 
c = np.trunc(c) 
print(c) 

# How to round elements of the NumPy array to the nearest integer?
print("Q10")

x = np.array([-.7, -1.5, -1.7, 0.3, 1.5, 1.8, 2.0])
print("Original array:")
print(x)

x = np.rint(x)
print("Round elements of the array to the nearest integer:")
print(x)

# Find the round off the values of the given matrix
print("Q11")

x = np.matrix('[1.2, 2.3; 4.7, 5.5; 7.2, 8.9]') 
           
# applying matrix.round() method 
result = x.round()    
print(result) 

# Determine the positive square-root of an array
print("Q12")

arr1 = np.sqrt([1, 4, 9, 16]) 
arr2 = np.sqrt([6, 10, 18]) 
  
print("square-root of an array1  : ", arr1) 
print("square-root of an array2  : ", arr2) 

# Evaluate Einstein’s summation convention of two multidimensional NumPy arrays
print("Q13")

matrix1 = np.array([[1, 2], [0, 2]]) 
matrix2 = np.array([[0, 1], [3, 4]]) 
  
print("Original matrix:") 
print(matrix1) 
print(matrix2) 
  
# Output 
result = np.einsum("mk,kn", matrix1, matrix2) 
print("Einstein’s summation convention of the two matrix:") 
print(result) 

#%% Questions on NumPy Statistics

# Compute the median of the flattened NumPy array
print("Q1")

x_odd = np.array([1, 2, 3, 4, 5, 6, 7]) 
print("\nPrinting the Original array:") 
print(x_odd) 
  
# calculating median 
med_odd = np.median(x_odd) 
print("\nMedian of the array that contains odd no of elements:") 
print(med_odd) 

# Find Mean of a List of Numpy Array
print("Q2")

Input = [np.array([1, 2, 3]), 
         np.array([4, 5, 6]), 
         np.array([7, 8, 9])] 
  
# Output list initialization 
Output = [] 
  
# using np.mean() 
for i in range(len(Input)): 
   Output.append(np.mean(Input[i])) 
  
# Printing output 
print(Output) 

# Calculate the mean of array ignoring the NaN value
print("Q3")
# 2-D
arr = np.array([[20, 15, 37], [47, 13, np.nan]])  
    
print("Shape of array is", arr.shape)  
print("Mean of array without using nanmean function:", np.mean(arr))  
print("Using nanmean function:", np.nanmean(arr))

# 2-D matrix with nan value  
arr = np.array([[32, 20, 24],  
                [47, 63, np.nan],    
                [17, 28, np.nan], 
                [10, 8, 9]])  
    
print("Shape of array is", arr.shape)  
print("Mean of array with axis = 0:", np.mean(arr, axis = 0))     
print("Using nanmedian function:", np.nanmean(arr, axis = 0))  

# 2-D matrix with nan value 
arr = np.array([[32, 20, 24],  
                [47, 63, np.nan],    
                [17, 28, np.nan], 
                [10, 8, 9]])  
    
print("Shape of array is", arr.shape)  
print("Mean of array with axis = 1:", np.mean(arr, axis = 1))  
print("Using nanmedian function:", np.nanmean(arr, axis = 1))

# Get the mean value from given matrix
print("Q4")
x = np.matrix(np.arange(12).reshape((3, 4)))
print(x)
print(x.mean(0)) # Returns the average of the matrix elements along the given axis.

# Compute the variance of the NumPy array
print("Q5")

# Compute the variance along the specified axis.
a = np.array([[1, 2], [3, 4]])
print(np.var(a))
print(np.var(a, axis=0))

# Compute the standard deviation of the NumPy array
print("Q6")

a = np.array([[1, 2], [3, 4]])
print(np.std(a))
print(np.std(a, axis=0))
# Compute the standard deviation along the specified axis.

# Compute pearson product-moment correlation coefficients of two given NumPy arrays
print("Q7")

array1 = np.array([0, 1, 2]) 
array2 = np.array([3, 4, 5]) 
  
# pearson product-moment correlation coefficients of the arrays 
rslt = np.corrcoef(array1, array2) 
print(rslt)

# Calculate the mean across dimension in a 2D NumPy array
print("Q8")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
  
# Calculating mean across Rows 
row_mean = np.mean(arr, axis=1) 
  
row1_mean = row_mean[0] 
print("Mean of Row 1 is", row1_mean) 
  
row2_mean = row_mean[1] 
print("Mean of Row 2 is", row2_mean) 
  
row3_mean = row_mean[2] 
print("Mean of Row 3 is", row3_mean) 
  
  
# Calculating mean across Columns 
column_mean = np.mean(arr, axis=0) 
  
column1_mean = column_mean[0] 
print("Mean of column 1 is", column1_mean) 
  
column2_mean = column_mean[1] 
print("Mean of column 2 is", column2_mean) 
  
column3_mean = column_mean[2] 
print("Mean of column 3 is", column3_mean) 

# Calculate the average, variance and standard deviation in Python using NumPy
print("Q9")
x = np.array([2, 4, 4, 4, 5, 5, 7, 9] )
  
# Calculating average using average() 
print(np.average(x))

# Calculating variance using var() 
print(np.var(x))

# Calculating standard deviation using var() 
print(np.std(x))

# Describe a NumPy Array in Python
print("Q10")

arr = np.array([4, 5, 8, 5, 6, 4, 9, 2, 4, 3, 6]) 
print(arr)

#%% Questions on Polynomial

# Define a polynomial function
print("Q1")

p = np.poly1d([1, 2, 3]) # A one-dimensional polynomial class.
print(np.poly1d(p))

# How to add one polynomial to another using NumPy in Python?
print("Q2")

# define the polynomials p(x) = 5(x**2) + (-2)x +5 
px = (5,-2,5) 
  
# q(x) = 2(x**2) + (-5)x +2 
qx = (2,-5,2) 
  
# add the polynomials 
rx = np.polynomial.polynomial.polyadd(px,qx) 
  
# print the resultant polynomial 
print(rx)

# How to subtract one polynomial to another using NumPy in Python?
print("Q3")

# define the polynomials p(x) = 5(x**2) + (-2)x +5 
px = (5,-2,5) 
  
# q(x) = 2(x**2) + (-5)x +2 
qx = (2,-5,2) 
  
# subtract the polynomials 
rx = np.polynomial.polynomial.polysub(px,qx) 
  
# print the resultant polynomial 
print(rx)

# How to multiply a polynomial to another using NumPy in Python?
print("Q4")

# define the polynomials p(x) = 5(x**2) + (-2)x +5 
px = (5, -2, 5) 
# q(x) = 2(x**2) + (-5)x +2 
qx = (2, -5, 2) 
  
# mul the polynomials 
rx = np.polynomial.polynomial.polymul(px, qx) 
  
# print the resultant polynomial 
print(rx) 

# How to divide a polynomial to another using NumPy in Python?
print("Q5")

# define the polynomials p(x) = 5(x**2) + (-2)x +5 
px = (5, -2, 5) 
  
# g(x) = x +2 
gx = (2, 1, 0) 
  
# divide the polynomials 
qx, rx = np.polynomial.polynomial.polydiv(px, gx) 
  
# print the result quotient 
print(qx) 
  
# remainder 
print(rx) 

# Find the roots of the polynomials using NumPy
print("Q6")

# Enter the coefficients of the poly in the array 
coeff = [1, 2, 1] 
print(np.roots(coeff)) 

# Evaluate a 2-D polynomial series on the Cartesian product
print("Q7")

c = np.array([[1, 3, 5], [2, 4, 6]])  
  
# using np.polygrid2d() method  
ans = polygrid2d([7, 9], [8, 10], c) 
print(ans) 

# Evaluate a 3-D polynomial series on the Cartesian product
print("Q8")

c = np.array([[1, 3, 5], [2, 4, 6], [10, 11, 12]])  
  
# using np.polygrid3d() method  
ans = polygrid3d([7, 9], [8, 10], [5, 6], c) 
print(ans) 

#%% Questions on NumPy String

# Repeat all the elements of a NumPy array of strings
print("Q1")
arr = np.array(['Akash', 'Rohit', 'Ayush', 'Dhruv', 'Radhika'], dtype = np.str) 
print("Original Array :") 
print(arr) 
  
# with the help of np.char.multiply repeating the characters 3 times 
new_array = np.char.multiply(arr, 3) 
print("\nNew array :") 
print(new_array)

# How to split the element of a given NumPy array with spaces?
print("Q2")

array = np.array(['PHP C# Python C Java C++'], dtype=np.str) 
print(array) 
  
# Split the element of the said array with spaces 
sparr = np.char.split(array) 
print(sparr) 

# How to insert a space between characters of all the elements of a given NumPy array?
print("Q3")

x = np.array(["geeks", "for", "geeks"], dtype=np.str) 
print("Printing the Original Array:") 
print(x) 
  
# inserting space using np.char.join() 
r = np.char.join(" ", x) 
print("Printing the array after inserting space between the elements") 
print(r) 

# Find the length of each string element in the Numpy array
print("Q4")

arr = np.array(['New York', 'Lisbon', 'Beijing', 'Quebec'])  
print(arr) 

# Use vectorize function of numpy 
length_checker = np.vectorize(len) 
  
# Find the length of each element 
arr_len = length_checker(arr) 
  
# Print the length of each element 
print(arr_len) # Repeat all the elements of a NumPy array of strings
print("Q1")
arr = np.array(['Akash', 'Rohit', 'Ayush', 'Dhruv', 'Radhika'], dtype = np.str) 
print("Original Array :") 
print(arr) 
  
# with the help of np.char.multiply repeating the characters 3 times 
new_array = np.char.multiply(arr, 3) 
print("\nNew array :") 
print(new_array)

# How to split the element of a given NumPy array with spaces?
print("Q2")

array = np.array(['PHP C# Python C Java C++'], dtype=np.str) 
print(array) 
  
# Split the element of the said array with spaces 
sparr = np.char.split(array) 
print(sparr) 

# How to insert a space between characters of all the elements of a given NumPy array?
print("Q3")

x = np.array(["geeks", "for", "geeks"], dtype=np.str) 
print("Printing the Original Array:") 
print(x) 
  
# inserting space using np.char.join() 
r = np.char.join(" ", x) 
print("Printing the array after inserting space between the elements") 
print(r) 

# Find the length of each string element in the Numpy array
print("Q4")

arr = np.array(['New York', 'Lisbon', 'Beijing', 'Quebec'])  
print(arr) 

# Use vectorize function of numpy 
length_checker = np.vectorize(len) 
  
# Find the length of each element 
arr_len = length_checker(arr) 
  
# Print the length of each element 
print(arr_len) 

# Swap the case of an array of string
print("Q5")

in_arr = np.array(['P4Q R', '4q Rp', 'Q Rp4', 'rp4q']) 
print ("input array : ", in_arr) 
  
out_arr = np.char.swapcase(in_arr) 
print ("output swapcasecased array :", out_arr) 

# Change the case to uppercase of elements of an array
print("Q6")

c = np.array(['a1b c', '1bca', 'bca1'])
print(np.char.upper(c))

# Change the case to lowercase of elements of an array
print("Q7")

c = np.array(['A1B C', '1BCA', 'BCA1'])
print(np.char.lower(c))

# Join String by a seperator
print("Q8")

in_arr = np.array(['Python', 'Numpy', 'Pandas']) 
print ("Input original array : ", in_arr)  
  
# creating the separator 
sep = np.array(['-', '+', '*']) 
  
out_arr = np.core.defchararray.join(sep, in_arr) 
print ("Output joined array: ", out_arr)  

# Check if two same shaped string arrayss one by one
print("Q9")

in_arr1 = np.array('numpy') 
print ("1st Input array : ", in_arr1) 
  
in_arr2 = np.array('numpy') 
print ("2nd Input array : ", in_arr2)   
  
# checking if they are equal 
out_arr = np.char.equal(in_arr1, in_arr2) 
print ("Output array: ", out_arr)

# Count the number of substrings in an array
print("Q10")

in_arr = np.array(['Sayantan', '  Sayan  ', 'Sayansubhra']) 
print ("Input array : ", in_arr)  
  
# output arrays  
out_arr = np.char.count(in_arr, sub ='an') 
print ("Output array: ", out_arr)  

# Find the lowest index of the substring in an array
print("Q11")
arr = ['vdsdsttetteteAAAa']
sub = 'ds'
res = np.char.find(arr, sub, 0)
print(res)

# Get the boolean array when values end with a particular character
print("Q12")
a = np.array(['geeks', 'for', 'geeks']) 
gfg = np.char.endswith(a, 'ks') 
  
print(gfg)

# Different ways to convert a Python dictionary to a NumPy array
print("Q13")

# Creating a Dictionary with Integer Keys 
dict = {1: 'Geeks', 
        2: 'For', 
        3: 'Geeks'} 
  
# to return a group of the key-value pairs in the dictionary 
result1 = dict.items() 
  
# Convert object to a list 
data1 = list(result1) 
  
# Convert list to an array 
numpyArray1 = np.array(data1) 
  
# print the numpy array 
print(numpyArray1)

# Creating a Nested Dictionary 
dict = {1: 'Geeks', 
        2: 'For', 
        3: {'A': 'Welcome', 
            'B': 'To', 
            'C': 'Geeks'} 
        } 
  
# to return a group of the key-value pairs in the dictionary 
result2 = dict.items() 
  
# Convert object to a list 
data2 = list(result2) 
  
# Convert list to an array 
numpyArray2 = np.array(data2) 
  
# print the numpy array 
print(numpyArray2)

# How to convert a list and tuple into NumPy arrays?
print("Q14")

# list 
list1 = [3, 4, 5, 6] 
print(type(list1)) 
print(list1) 
print() 
  
# conversion 
array1 = np.asarray(list1) 
print(type(array1)) 
print(array1) 
print() 
  
# tuple 
tuple1 = ([8, 4, 6], [1, 2, 3]) 
print(type(tuple1)) 
print(tuple1) 
print() 
  
# conversion 
array2 = np.asarray(tuple1) 
print(type(array2)) 
print(array2)

# Ways to convert array of strings to array of floats
print("Q15")

# 1
ini_array = np.array(["1.1", "1.5", "2.7", "8.9"]) 
  
# printing initial array 
print ("initial array", str(ini_array)) 
  
# conerting to array of floats 
# using np.astype 
res = ini_array.astype(np.float) 
  
# printing final result 
print ("final array", str(res)) 

# 2
ini_array = np.array(["1.1", "1.5", "2.7", "8.9"]) 
  
# printing initial array 
print ("initial array", str(ini_array)) 
  
# conerting to array of floats using np.fromstring 
ini_array = ', '.join(ini_array) 
ini_array = np.fromstring(ini_array, dtype = np.float, sep =', ' ) 
  
# printing final result 
print ("final array", str(ini_array))

# 3

ini_array = np.array(["1.1", "1.5", "2.7", "8.9"]) 
  
# printing initial array 
print ("initial array", str(ini_array)) 
  
# conerting to array of floats using np.asarray 
final_array = b = np.asarray(ini_array, dtype = np.float64, order ='C') 
  
# printing final result 
print ("final array", str(final_array)) 

# Convert a NumPy array into a csv file
print("Q16")

# 1
arr = np.arange(1,11).reshape(2,5) 
  
# display the array 
print(arr) 
  
# convert array into dataframe 
DF = pd.DataFrame(arr) 
  
# save the dataframe as a csv file 
DF.to_csv("data1.csv")

# 2 
arr = np.arange(1,11) 
  
# display the array 
print(arr) 
  
# use the tofile() method  
# and use ',' as a separator 
# as we have to generate a csv file 
arr.tofile('data2.csv', sep = ',')

# 3 
a = np.array([[1, 6, 4], 
                 [2, 4, 8], 
                 [3, 9, 1]]) 
  
# save array into csv file 
np.savetxt("data3.csv", a, delimiter = ",")

# How to Convert an image to NumPy array and save it to CSV file using Python?
print("Q17")

# 1
# read an image 
img = Image.open('my_image1.jpg') 
  
# convert image object into array 
imageToMatrice = np.asarray(img) 
  
# printing shape of image 
print(imageToMatrice.shape)

# 2
# read an image 
imageToMatrice  = imread('my_image1.jpg') 
  
# show shape of the image 
print(imageToMatrice.shape)

# How to save a NumPy array to a text file?
print("Q18")

x = y = z = np.arange(0.0,5.0,1.0)
np.savetxt('test.out', x, delimiter=',')   # X is an array
np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

# Load data from a text file
print("Q19")
from io import StringIO   # StringIO behaves like a file object
c = StringIO("0 1\n2 3")
np.loadtxt(c)

# Plot line graph from NumPy array
print("Q20")

# data to be plotted 
x = np.arange(1, 11)  
y = x * x 
  
# plotting 
plt.title("Line graph")  
plt.xlabel("X axis")  
plt.ylabel("Y axis")  
plt.plot(x, y, color ="red")  
plt.show()

# Create Histogram using NumPy
print("Q21")

np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
np.histogram(np.arange(4), bins=np.arange(5), density=True)
np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
