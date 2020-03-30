KNN_for_mixed_data
------------------
The KNeighborsClassifier of sklearn package rests mainly on Euclidian distance and some other metrics that are suitable only to quantify the similarity between two observations for numerical variables. So, our purpose is to find out how to exploit these metrics on a mixed data so that we can apply the KNN algorithm. We apply the Factor Analysis of Mixed Data (FAMD), we subset from them 90% of the information, and transform it into numerical data, which are the coordinates of each observation on 19 components.
