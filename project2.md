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
