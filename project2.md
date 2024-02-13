## Project description:
Machine learning and statistics are 2 closely related and intertwined fields, with many machine learning models being heavily based on statistical theories.<br>
The goal of this project is to use Natural Language Processing (NLP) on the corpus of data scraped from r/statistics and r/machinelearning to explore the intersection of statistics and machine learning and gain a deeper understanding of how these two fields intertwine. After which, a text classifier will be built to classify whether the post belongs to r/machinelearning or r/statistics.

<br>
---

## Project Objectives: 
1. To uncover the common and different topics and communities within the 2 subreddit threads
2. To build a text classifier to differentiate whether a reddit post is from r/statistics or r/machinelearning

<br>
---

## Tech Used:
- Python Reddit API Wrapper (PRAW) (data acquisition from Reddit)
- Pandas/numpy (data manipulation)
- Matplotlib/seaborn (data visualization)
- Sklearn (Machine Learning)
- KMeans (Community detection)

<br>
---

## Project Outcomes
1000 posts each from r/machinelearning and r/statistics were scrapped. After some data cleaning, each post (title and contents) was tokenized and saved. A KMeans model is then fitted on the word tokens, and the KMeans model with the lowest inertia value was chosen. The distinct and overlapping topics were then analyzed.

<br>

Several models (LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier and GradientBoostClassifier) were built for text classification, and the most optimal model built was a Logistic Regression model, considering its good performance as well as relatively low computational cost compared to the other models used.

<br>
---

### 1. Datasets Used
1000 posts each from r/statistics and r/machinelearning were scrapped using PRAW. Features scrapped included the post timestamp, title, contents, top 10 comments (saved as an array of strings), number of upvotes, number of comments, url, tags and popularity score. <br>

An additional column 'is_ml' is added to distinguish whether the post is from r/machinelearning or not (ie r/statistics).<br><br>

---

### 2. Data Analysis & Key Insights

Post titles and contents were tokenized using a TfidfVectorizer.
```python3
tfidf_title = TfidfVectorizer()

X_title = tfidf_title.fit_transform(df['tokenized_title'])
```

TfidfVectorizer was used here instead of CountVectorizer because we want unique keywords relevant to each subreddit thread to stand out more, which will not be achieved if we use CountVectorizer which simply counts the number of ocurences of each word in the corpus.<br>

The KMeans model with the lowest _inertia_ was identified:<br>

```python3
inertia_title = []
for i in range(1,50):
    kmeans = KMeans(n_clusters=i,max_iter=500)
    kmeans.fit(X_title)
    inertia_title.append((i,kmeans.inertia_))

inertia_values = [tup[1] for tup in inertia_title]
plt.plot(range(1,50),inertia_values)
plt.show()

```
<br>After identifying n=36 (elbow point),  the cluster number was appended to the dataframe.

```python3
kmeans = KMeans(n_clusters=37,max_iter=500, random_state=42)
kmeans.fit(X_title)
df['title_cluster'] = kmeans.labels_
```

<br>Then, clusters with a majority (>90%) of either r/statistics or r/machinelearning posts were subsetted to analyze distinct differences in communities and topics amongst both subreddit threads. Clusters with a homogeneous mix (~50% of r/statistics and r/machinelearning) were also analyzed.

```python3
def get_top_keywords(tfidf_matrix, clusters, feature_names, n_terms):
    '''
    Prints the top n terms in each cluster from the TF-IDF matrix.

    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.
        clusters (numpy.ndarray): An array indicating the cluster to which each row of the TF-IDF matrix belongs.
        feature_names (list of str): The names of the features corresponding to the columns of the TF-IDF matrix.
        n_terms (int): The number of top terms to print for each cluster.

    Returns:
        None
    '''
    
    df = pd.DataFrame(tfidf_matrix.todense()).groupby(clusters).mean()
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([feature_names[t] for t in np.argsort(r)[-n_terms:]]))

for i in range(36):
  #Extracting clusters with distinct topics
  if (df[df['title_cluster']==i].sort_values('title_cluster')['is_ml'].value_counts(normalize=True).max()>0.9) == True:
      print(f"Cluster {i}:")
      display(df[df['title_cluster']==i].sort_values('title_cluster')['is_ml'].value_counts())
      get_top_keywords(X_content, kmeans_content.labels_, tfidf_content.get_feature_names_out(), 10)
      print("\n")

for i in range(36):
  #Extracting clusters with similar topics
    if (df_temp[df_temp['content_cluster']==i].sort_values('content_cluster')['is_ml'].value_counts(normalize=True).max()<0.6)== True:
        print(f"Cluster {i}:")
        display(df_temp[df_temp['content_cluster']==i].sort_values('content_cluster')['is_ml'].value_counts())
        # Call the function to get the top keywords
        print("\n")

```
<br>

Clusters with majority (>90%) r/statistics or majority r/machinelearning posts were identified, and the common words were analyzed.<br>
r/statistics seems **more oriented towards understanding and correctly applying statistical principles and methods**, often in academic or research contexts. On the other hand, r/machinelearning is more focused on advanced machine learning techniques, **practical implementation**, emerging technologies, and **industry trends**.

<br>
<table>
<thead>
<tr>
<th>Cluster</th>
<th>Value Counts</th>
<th>Predominant Topic</th>
<th>Common Words</th>
</tr>
</thead>
<tbody>
<tr>
<td>2</td>
<td>0 - 49<br>1 - 2</td>
<td>r/statistics</td>
<td>grad,year,school,course,algebra,class,im,statistic,stats,math</td>
</tr>
<tr>
<td>4</td>
<td>0 - 26<br>1- 0</td>
<td>r/statistics</td>
<td>sample,data,test,concentration,group,control,spss,treatment,different,anova</td>
</tr>
<tr>
<td>9</td>
<td>0 - 41<br>1 - 0</td>
<td>r/statistics</td>
<td>value,standard,error,disease,variance,mean,population,size,deviation,sample</td>
</tr>
<tr>
<td>13</td>
<td>0 - 21</td>
<td>r/statistics</td>
<td>test,speed,list,analysis,mediation,two,relationship,dependent,independent,variable</td>
</tr>
<tr>
<td>14</td>
<td>0 - 43<br>1 - 3</td>
<td>r/statistics</td>
<td>effect,data,hi,linear,categorical,analysis,dataset,regression,model,variable</td>
</tr>
<tr>
<td>15</td>
<td>0 - 49<br>1 - 5</td>
<td>r/statistics</td>
<td>skewed,time,data,one,probability,sample,mean,value,normal,distribution</td>
</tr>
<tr>
<td>17</td>
<td>0 - 20<br>1 - 0</td>
<td>r/statistics</td>
<td>jamovi,question,function,wilcoxon,answer,differenced,welchs,student,interval,confidence</td>
</tr>
<tr>
<td>26</td>
<td>0 - 31<br>1 - 1</td>
<td>r/statistics</td>
<td>mean,table,reject,test,hypothesis,variable,change,null,cell,survival</td>
</tr>
<tr>
<td>29</td>
<td>0 - 62<br>1 - 1</td>
<td>r/statistics</td>
<td>appropriate,anova,significant,im,variable,ttest,data,sample,group,test</td>
</tr>
<tr>
<td>32</td>
<td>0 - 15<br>1 - 0</td>
<td>r/statistics</td>
<td>costeffectiveness,two,analysis,randomized,cea,effect,conduct,itc,trial,treatment</td>
</tr>
<tr>
<td>33</td>
<td>0 - 32<br>1 - 2</td>
<td>r/statistics</td>
<td>green,one,anova,marble,compare,test,box,red,subject,group</td>
</tr>
<tr>
<td>34</td>
<td>0 - 27<br>1 - 1</td>
<td>r/statistics</td>
<td>control,education,covariate,data,comparison,year,variable,effect,salary,age</td>
</tr>
<tr>
<td>5</td>
<td>0 - 3<br>1 - 36</td>
<td>r/machinelearning</td>
<td>yet,project,token,text,model,gpt4,code,api,openai,prompt</td>
</tr>
<tr>
<td>7</td>
<td>0 - 0<br>1 - 71</td>
<td>r/machinelearning</td>
<td>one,language,dataset,question,user,could,task,like,model,llm</td>
</tr>
<tr>
<td>8</td>
<td>0 - 4<br>1 - 52</td>
<td>r/machinelearning</td>
<td>research,paper,chatgpt,ai,state,diffusion,llama,language,generative,model</td>
</tr>
<tr>
<td>12</td>
<td>0 - 8<br>1 - 79</td>
<td>r/machinelearning</td>
<td>architecture,one,ml,new,like,dataset,data,train,training,model</td>
</tr>
<tr>
<td>18</td>
<td>0 - 1<br>1 - 36</td>
<td>r/machinelearning</td>
<td>text,like,segmentation,similarity,network,input,output,task,model,image</td>
</tr>
<tr>
<td>31</td>
<td>0 - 3<br>1 - 29</td>
<td>r/machinelearning</td>
<td>link,post,text,code,window,transformer,model,finetuning,context,token</td>
</tr>
</tbody>
</table>
<br>

As for overlapping topics in r/statistics and r/machinelearning, it seems that there is a **common interest in sharing learning materials** (youtube,course,recommendation, book) here.<br>

<table>
<thead>
<tr>
<th>Cluster</th>
<th>Value Counts</th>
<th>Common Words</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0 - 399<br>1 - 350</td>
<td>criterion,paper,youtube,content,dataset,course,textbook,learn,book,recommendation</td>
</tr>
</tbody>
</table>


<br>
---

### 3. Machine Learning

6 models were identified as candidates to fit a text classifier model to distinguish whether the post belongs to r/statistics or r/machinelearning.<br>
The 6 models were:
- LogisticRegression
- SVC
- DecisionTreeClassifier
- RandomForestClassifier
- AdaBoostClassifier
- GradientBoostClassifier

The decision tree classifier is used here in conjunction with the random forest classifier to serve as a "sanity check" . The random forest classifier is expected to perform and generalize better to the holdout set compared to the decision tree, and seeing otherwise could be signs of underlying issues in the dataset.<br>

A Gridsearch was conducted, followed by the optimal parameters being taken to train on the above 6 models, and we see the results below. <br><br>

**Grid Search:**
<br>
```python3

#Creating the pipeline object
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('classifier', LogisticRegression(random_state=42))
])


#Defining the parameter grids for each model

parameter_grids = [

    {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty':[None,'l1','l2'],
        'classifier__solver':['lbfgs','liblinear'],
    },
    {
        'classifier': [SVC(random_state=42)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf', 'poly']
    },
    
        {
        'classifier': [DecisionTreeClassifier(random_state=42)],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10, 20]
    },
    
       {
        'classifier': [AdaBoostClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    },
    
      {
        'classifier': [GradientBoostingClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    }
]


#Resetting sns gridstyle
sns.set_style("white")

#Ignoring warnings

warnings.filterwarnings('ignore')


# Perform grid search for each model
for parameter_grid in parameter_grids:
    grid_search = GridSearchCV(pipeline, parameter_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and the classification report
    print("Best parameters:", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=grid_search.classes_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot()
    plt.show()
    print("---------------------------------------")
    
warnings.filterwarnings("default")


```
<br>

**Model Training:**
```python3
#Commenting out the following code  -results of the gridsearch have been appended in table below

#Resetting sns gridstyle
sns.set_style("white")

#Ignoring warnings

warnings.filterwarnings('ignore')


# Perform grid search for each model
for parameter_grid in parameter_grids:
    grid_search = GridSearchCV(pipeline, parameter_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters and the classification report
    print("Best parameters:", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=grid_search.classes_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot()
    plt.show()
    print("---------------------------------------")
    
warnings.filterwarnings("default")


```

The results were seen as follows:<br>
<table>
<thead>
<tr>
<th>Model</th>
<th>Best Score</th>
<th>Best Estimator</th>
</tr>
</thead>
<tbody>
<tr>
<td>LogisticRegression</td>
<td>0.982</td>
<td>LogisticRegression(C=0.1, penalty=None, solver='lbfgs', random_state=42)</td>
</tr>
<tr>
<td>SVC</td>
<td>0.980</td>
<td>SVC(C=10.0, kernel='linear', random_state=42)</td>
</tr>
<tr>
<td>DecisionTreeClassifier</td>
<td>0.964</td>
<td>DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=42)</td>
</tr>
<tr>
<td>RandomForestClassifier</td>
<td>0.977</td>
<td>RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)</td>
</tr>
<tr>
<td>AdaBoostClassifier</td>
<td>0.980</td>
<td>AdaBoostClassifier(n_estimators=300, learning_rate=1.0, random_state=42)</td>
</tr>
<tr>
<td>GradientBoostClassifier</td>
<td>0.968</td>
<td>GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42)</td>
</tr>
</tbody>
</table>
<br>
The following are some conclusions that can be drawn from the above table:

- DecisionTreeClassifier consistently performed the worst amongst the 6 models
- LogisticRegression is optimal here given that it is the least computationally expensive model to train, and has great model accuracy relative to the other models
<br><br>
There are certain distinct keywords that are strong indicators of whether a post belongs to r/statistics or r/machinelearning (eg ai, gpt for r/machinelearning and population, sampling, hypothesis for r/statistics). This gives credence to the high accuracies of all the models across the board.

---


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
