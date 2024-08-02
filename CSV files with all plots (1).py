#!/usr/bin/env python
# coding: utf-8

# In[1]:


odd_numbers = [1, 3, 5, 7, 9]
squares = list(map(lambda x: x ** 2, odd_numbers))
print(squares)


# In[2]:


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
primes = list(filter(is_prime, numbers))
print(primes)


# In[4]:


import pandas as pd

df = pd.read_csv("C:/Users/HP/Desktop/ssss/11/mtcars.csv")


print("First 5 rows of the dataset:")
print(df.head())

print("Summary statistics of the dataset:")
print(df.describe())

print("Column names of the dataset:")
print(df.columns)


# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
df = pd.read_csv("C:/Users/HP/Desktop/ssss/11/mtcars.csv")

plt.hist(df['mpg'], bins=10, edgecolor='black')
plt.title('Histogram of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

hist, bin_edges = np.histogram(df['mpg'], bins=10)
max_freq_bin = bin_edges[np.argmax(hist)]
print(max_freq_bin)  


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/Users/HP/Desktop/ssss/11/mtcars.csv")

plt.scatter(df['wt'], df['mpg'])
plt.title('Scatter Plot of Weight vs MPG')
plt.xlabel('Weight (1000 lbs)')
plt.ylabel('MPG')
plt.grid(True)
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/Users/HP/Desktop/ssss/11/mtcars.csv")

transmission_counts = df['am'].value_counts()
transmission_labels = ['Automatic', 'Manual']

plt.bar(transmission_labels, transmission_counts, color=['blue', 'orange'])
plt.title('Frequency Distribution of Transmission Types')
plt.xlabel('Transmission Type')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[ ]:




