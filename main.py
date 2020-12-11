
import pandas as pd #To read in file
import numpy as np #To use arrays that work well in python
import math #To use sqrt()
from collections import Counter #to count occurences of particular Y's in the KNN
import random #To use random when there are ties

filename = 'Data/Absenteeism_at_work.xls'
data =  pd.read_excel(filename)
data = data[["Age", "Social drinker", "Social smoker", "Weight", "Height", "Body mass index", "Absenteeism time in hours"]]
data = data.to_numpy()


rows = 189
cols = 7
K = 5
TT = 50 #TT is the Training threshhold which is the initial number of Instances to be loaded into the Training set

def normalize_features(input):

    normalizedList = []
    for i in range(cols): #iterate through coloumns (features)
        col = []
        if (i == cols - 1): # This is the Y do not normalize
            for j in range(rows):
                col.append(input.item(j, i))
            normalizedList.append(col)
            continue
        maximum = input.item(0, i) # Set maximum arbitrarily to the first value to allow program to work with any value range
        minimum = input.item(0, i) # Same ^^
        #print(input.item(20,1))
        for j in range(rows): #Loop to find minimum and Maximum feature value
            if maximum < input.item((j, i)):
                maximum = input.item((j, i))
            elif minimum > input.item((j, i)):
                minimum = input.item((j, i))
        if maximum == 1 and minimum == 0: #if this feature is binary skip normalization
            for j in range(rows):
                col.append(input.item(j, i)) #Place values directly into list
            normalizedList.append(col)
            continue
        else:
            for j in range(rows): #Iternate Through this column on ROWS (values)
                current = input.item((j, i))
                normValue = (current - minimum)/(maximum - minimum) # Apply Min - Max formula
                col.append(normValue) # Append Normalized value to the bottom of the column
            normalizedList.append(col)
    return [normalizedList]

def euclideanDistance(Instance1, Instance2):
    ED = 0
    for i in range(6):
        ED = ED + (Instance1[i] - Instance2[i])**2
    ED = math.sqrt(ED)
    return ED

def majorityValue(input):
    #print(input)
    Yvals = Counter(input).most_common(5)
    Yvals = np.array(Yvals)
    if Yvals[0][1] == 5:
        ismajority = Yvals[0][0]
        return ismajority
    if Yvals[0][1] == Yvals[1][1] and Yvals[2][1]:
        if Yvals[0][1] == Yvals[2][1] and Yvals[3][1]:
            if Yvals[0][1] == Yvals[3][1] and Yvals[4][1]:
                if Yvals[0][1] == Yvals[4][1]:
                        ismajority = random.choice(Yvals[:4][0]) #Rare 5 way tie
                else:
                    ismajority = random.choice(Yvals[:3][0])# 4 way tie
            else:
                ismajority = random.choice(Yvals[:2][0]) #3 way tie
        else:
            ismajority = random.choice(Yvals[:1][0])
    else:
        ismajority = Yvals[0][1]

    return ismajority

def main():

    output = normalize_features(data)
    A = np.array(output)
    A = np.squeeze(A)
    D = A[:, :TT]
    X = A[:6, TT + 1:rows]
    YC = A[6:, TT + 1:rows]
    distPairings = []
    np.array(distPairings)
    y_col = []
    for i in range(rows - TT - 1): # maybe use rows and coloumns
        for j in range(TT):
            ED = euclideanDistance(X[:, i], D[:, j])
            distPairings.append([ED, D[:, j][6]]) # fill list with tuples of form [Distance, D(Y)]
        distPairings = np.array(distPairings)
        distPairings = distPairings[distPairings[:,0].argsort()]
        #distPairings = np.sort(distPairings, axis = 1)
        distPairings = distPairings[:,1]
        y_col.append(majorityValue(distPairings[0:K]))
        distPairings = []
    Y = np.array(y_col)
    print("Y", Y)
    print("YCorrect", YC)
    match = 0
    for i in range(np.size(Y)):
        if Y[i] == YC[0][i]:
            match = match + 1
    print("Accuracy: ", match/(rows-TT))


if __name__ == "__main__":
    main()
