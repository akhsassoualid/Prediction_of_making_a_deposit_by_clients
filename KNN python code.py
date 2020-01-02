# Import packages 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data and first look on it, and detect missied values 
df = pd.read_csv('C:/Users/admin/Desktop/Udemy/MY ALGO OF DATA SCIENCE/KNN for classification/bank-additional-full.csv', 
                 sep=';', comment='#', na_values='NA')
df.head()
df.info()
df.columns
df = df.drop(['duration'], axis=1)
df = df.drop(['poutcome'], axis=1)
df.y=df.y.astype('category')
if df.notnull().all().all() == True:
    print('there are no missing values') # it show no missing values

# Convert object variables to category and histogram for each category of each variable
for name_col in df.columns[1:10]:
    df[name_col]=df[name_col].astype('category')
    plt.figure()
    chart = sns.countplot(x=name_col, hue='y', data = df, palette='RdBu')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

# Heatmap of correlation matrix between numeric variables
my_corr = df.iloc[:,10:].corr()
sns.heatmap(my_corr, vmin=-1, vmax=1, annot=True, fmt='.2g')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# restructre the variables in a new order
df = df[['y','job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week',
         'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed']]

# let's use Multiple Correspondance Analysis t categorical data
"pip install prince"
import prince
" We creat a sub-data that contai only categorical data
" Our Purpose it to transoform these attributes to continious data so that we could use euclidien distance
cat_df = df.iloc[:,1:10]
mca_df = prince.MCA(n_components = 30,
                    n_iter = 4,
                    copy = True,
                    check_input=True,
                    engine='auto',random_state=42)
mca= mca_df.fit(cat_df)
    
import pprint
pp = pprint.PrettyPrinter()
pp.pprint(mca.explained_inertia_)
mca = mca.transform(cat_df)
mca.columns
# Merge the data 
newdf = pd.concat([df['y'],df.iloc[:,10:],mca], axis=1)
from sklearn import preprocessing as prpr
scaler = prpr.MinMaxScaler()
scaled_newdf = scaler.fit_transform(newdf.iloc[:,1:])
newdf.columns
scaled_newdf = pd.DataFrame(scaled_newdf, columns=newdf.columns[1:])
scaled_newdf['y'] = newdf['y']
scaled_newdf.columns = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
                        'cp0','cp1','cp2','cp3','cp4','cp4','cp5','cp6','cp7','cp8','cp9','cp10','cp11','cp12','cp13','cp14','cp15','cp16','cp17','cp18',
                        'cp19','cp20','cp21','cp22','cp23','cp24','cp25','cp26','cp27','cp28','y']

# Split the data into train and test
from sklearn.model_selection import train_test_split
X=scaled_newdf.iloc[:,0:39]
Y=scaled_newdf['y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Creat a KNN model
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=6)
model_knn.fit(X_train, Y_train)

# Print the accuracy
print(model_knn.score(X_test, Y_test))

# Overfitting and underfitting in function of k 
neighbor = np.arange(4,15)
train_accuracy = np.empty(len(neighbor))
test_accuracy = np.empty(len(neighbor))
for ind, k in enumerate(neighbor):
    mod_knn = KNeighborsClassifier(n_neighbors=k)
    mod_knn.fit(X_train, Y_train)
    train_accuracy[ind] = mod_knn.score(X_train, Y_train)
    test_accuracy[ind] = mod_knn.score(X_test, Y_test)
    
# compare the accuracy in function of multiple value of k
import matplotlib.pyplot as plt
plt.figure()
plt.plot(neighbor, train_accuracy, label='training accuracy')
plt.plot(neighbor, test_accuracy, label='testing accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
















ax = mca.plot_coordinates(
     X=cat_df,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1)
ax.get_figure()











