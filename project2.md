**Project description:** 
Machine learning and statistics are 2 closely related and intertwined fields, with many machine learning models being heavily based on statistical theories.<br>
The goal of this project is to use Natural Language Processing (NLP) on the corpus of data scraped from r/statistics and r/machinelearning to explore the intersection of statistics and machine learning and gain a deeper understanding of how these two fields intertwine. After which, a text classifier will be built to classify whether the post belongs to r/machinelearning or r/statistics.

<br><br>
**Project Objectives**: 
1. To uncover the common and different topics and communities within the 2 subreddit threads
2. To build a text classifier to differentiate whether a reddit post is from r/statistics or r/machinelearning


**Tech Used**:
- Python Reddit API Wrapper (PRAW) (data acquisition from Reddit)
- Pandas/numpy (data manipulation)
- Matplotlib/seaborn (data visualization)
- Sklearn (Machine Learning)
- KMeans (Community detection)


## Project Outcomes
1000 posts each from r/machinelearning and r/statistics were scrapped. After some data cleaning, each post (title and contents) was tokenized and saved. A KMeans model is then fitted on the word tokens, and the KMeans model with the lowest inertia value was chosen.<br><br>

Clusters with majority (>90%) r/statistics or majority r/machinelearning posts were identified, and the common words were analyzed:<br><br>
| **Cluster** | **Value Counts** | **Predominant Topic** | **Common Words**                                                                        |
|-------------|------------------|-----------------------|-----------------------------------------------------------------------------------------|
| 2           | 0 - 49<br>1 - 2  | r/statistics          | grad,year,school,course,algebra,class,im,statistic,stats,math                           |
| 4           | 0 - 26<br>1- 0   | r/statistics          | sample,data,test,concentration,group,control,spss,treatment,different,anova             |
| 9           | 0 - 41<br>1 - 0  | r/statistics          | value,standard,error,disease,variance,mean,population,size,deviation,sample             |
| 13          | 0 - 21           | r/statistics          | test,speed,list,analysis,mediation,two,relationship,dependent,independent,variable      |
| 14          | 0 - 43<br>1 - 3  | r/statistics          | effect,data,hi,linear,categorical,analysis,dataset,regression,model,variable            |
| 15          | 0 - 49<br>1 - 5  | r/statistics          | skewed,time,data,one,probability,sample,mean,value,normal,distribution                  |
| 17          | 0 - 20<br>1 - 0  | r/statistics          | jamovi,question,function,wilcoxon,answer,differenced,welchs,student,interval,confidence |
| 26          | 0 - 31<br>1 - 1  | r/statistics          | mean,table,reject,test,hypothesis,variable,change,null,cell,survival                    |
| 29          | 0 - 62<br>1 - 1  | r/statistics          | appropriate,anova,significant,im,variable,ttest,data,sample,group,test                  |
| 32          | 0 - 15<br>1 - 0  | r/statistics          | costeffectiveness,two,analysis,randomized,cea,effect,conduct,itc,trial,treatment        |
| 33          | 0 - 32<br>1 - 2  | r/statistics          | green,one,anova,marble,compare,test,box,red,subject,group                               |
| 34          | 0 - 27<br>1 - 1  | r/statistics          | control,education,covariate,data,comparison,year,variable,effect,salary,age             |
| 5           | 0 - 3<br>1 - 36  | r/machinelearning     | yet,project,token,text,model,gpt4,code,api,openai,prompt                                |
| 7           | 0 - 0<br>1 - 71  | r/machinelearning     | one,language,dataset,question,user,could,task,like,model,llm                            |
| 8           | 0 - 4<br>1 - 52  | r/machinelearning     | research,paper,chatgpt,ai,state,diffusion,llama,language,generative,model               |
| 12          | 0 - 8<br>1 - 79  | r/machinelearning     | architecture,one,ml,new,like,dataset,data,train,training,model                          |
| 18          | 0 - 1<br>1 - 36  | r/machinelearning     | text,like,segmentation,similarity,network,input,output,task,model,image                 |
| 31          | 0 - 3<br>1 - 29  | r/machinelearning     | link,post,text,code,window,transformer,model,finetuning,context,token                   |



### 1. Datasets Used
**_To be populated_**

---

### 2. Data Analysis & Key Insights

1. **There has been a steady decline of total voters in IMDb**
**_To be populated_**

---

### 3. Machine Learning

**_To be populated_**

---

### 4. Conclusion and Future Work

**_To be populated_** 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
