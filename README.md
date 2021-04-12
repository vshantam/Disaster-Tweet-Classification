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
4. Scaling and Fitting the model.
5. Performance evaluation
6. Optimizing Model with Parameter Tuning

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
plt.show()```

![image](https://user-images.githubusercontent.com/22946038/114455316-3e574200-9bf9-11eb-83c6-121ac30f6409.png)

Top Keywords used in Dataset 

```keyword_vc = df["keyword"].value_counts()
sns.barplot(y=keyword_vc[0:30].index, x=keyword_vc[0:30], orient='h')
plt.title("Top 30 keyword")
plt.show()```

![image](https://user-images.githubusercontent.com/22946038/114455363-4d3df480-9bf9-11eb-87b0-00c7d2a4ef6e.png)


Word Cloud of the Abbreveations used 

![image](https://user-images.githubusercontent.com/22946038/114455492-71013a80-9bf9-11eb-885d-b2c32520e846.png)

