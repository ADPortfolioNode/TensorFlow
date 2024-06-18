import pandas as pd
import pylab as pl
import numpy as np
import requests

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import matplotlib.patches as mpatches

tf.disable_v2_behavior()

# Download the dataset from the given URL
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
response = requests.get(url)

# Ensure we fail fast if the web request is not successful
response.raise_for_status()

# Open the file in write mode and write the content of the response to it
with open('FuelConsumption.csv', 'wb') as f:
    f.write(response.content)

plt.rcParams['figure.figsize'] = (10, 6)

# Generate an array of values from 0.0 to 5.0 with a step of 0.1
X = np.arange(0.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

# Calculate the dependent variable Y based on the equation Y = a * X + b
Y = a * X + b

# Plot the graph
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Print updates during calculation
print("X values:", X)
print("Y values:", Y)