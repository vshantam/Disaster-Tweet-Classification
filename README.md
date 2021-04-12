# Disaster-Tweet-Classification

<H3> This Project is to work on Natural language processing techniques to slove  language based problems.</H3>

This project deals with the classification problem for the disaster tweets.
So that it can be classified that which tweets is the real disaster tweets.

<H3> Pre requisistes</H3>

1. Python3 
2. Jupyter Notebook
3. Machine Learning
4. NLTK
5. Pandas
6. Sklearn
7. Regex
8. Vectorization 


<H3> Section Wise Implementation Guides </H3>

1. Data Loading.
2. Exploratory Data Analysis
3. PreProcessing
4. Fitting the model with Parameter Tuning.
5. Performance evaluation

<H2> Data Loading </H2>

I like padas , my basic approach is to load the data into the dataframe and then performing operations and explorations like EDA

```df = pd.read_csv("train.csv",engine="python", delimiter=",")```

<H2> Exploratory Data Analysis </H2>

Dataset describe

![image](https://user-images.githubusercontent.com/22946038/114455076-f2a49880-9bf8-11eb-954b-cce1e2ca39c0.png)

Dataset info

![image](https://user-images.githubusercontent.com/22946038/114455247-24b5fa80-9bf9-11eb-8db8-536938c6f975.png)

Top Locations used in Dataset

```locations_vc = df["location"].value_counts()
sns.barplot(y=locations_vc[0:30].index, x=locations_vc[0:30], orient='h')
plt.title("Top 30 Locations")
plt.show()
```

![image](https://user-images.githubusercontent.com/22946038/114455316-3e574200-9bf9-11eb-83c6-121ac30f6409.png)

Top Keywords used in Dataset 

```keyword_vc = df["keyword"].value_counts()
sns.barplot(y=keyword_vc[0:30].index, x=keyword_vc[0:30], orient='h')
plt.title("Top 30 keyword")
plt.show()
```

![image](https://user-images.githubusercontent.com/22946038/114455363-4d3df480-9bf9-11eb-87b0-00c7d2a4ef6e.png)


Word Cloud of the Abbreveations used 

![image](https://user-images.githubusercontent.com/22946038/114455492-71013a80-9bf9-11eb-885d-b2c32520e846.png)


<H2> PreProcessing</H2>

Pre processing is the most important phase .
As we are dealing with NLP , it is little different than the Numeric preprocessing Techninques.

The list of filters used for preprocessing the tweets are as follows 

1. Url
2. Html
3. Non Ascii
4. abbreveation replacement
5. removing mentions
6. number
7. punctuations
8. stop words

The above were used to clean the text before vectorization

Below is the glimpse of befor and after

![image](https://user-images.githubusercontent.com/22946038/114456754-dbff4100-9bfa-11eb-80cc-9e50eb183eda.png)

for vectorization of textual data , the two most popular methods are 
1. count vectorization 
2. TIDIF vectorization

<H2> Fitting the Model </H2>

To classify Randonm Forest Classifies has been used for the demonstration

```classifier = RandomForestClassifier(n_estimators=1000, random_state=0)```

The number of estimator can be decided by the testing with different values which size of estimator is giving you the best result.

For eg. the following model has been tested over multiple estimator size to determine which one giveing the most accurate results

![image](https://user-images.githubusercontent.com/22946038/114457364-90996280-9bfb-11eb-9b7d-03821bb63228.png)


<H2>Performance evaluation <H2>
  
 ```print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
```

![image](https://user-images.githubusercontent.com/22946038/114457528-c50d1e80-9bfb-11eb-8bb6-015abd29cbc1.png)

The number of estimator = 1000 gave the best result .

