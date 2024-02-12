
**Project description:** This project explores movie analytics through a Streamlit web app, leveraging IMDb and Movie Lens datasets to showcase the application of recommender systems and uncover trends in the film industry from 2013 to 2023.
<br><br>
**Project Objectives**: 
1. To conduct data analysis on the IMDB dataset
2. To build recommender engines to provide recommendations of movies based on user inputs

**Tech Used**:
- requests/beautifulsoup (data scraping)
- numpy/pandas (data manipulation)
- matplotlib/plotly/seaborn (data visualization)
- sklearn/tensorflow (machine learning)

## Project Outcomes
Please refer to the [following deck](/pdf/ppt1.pdf) for the presentation deck that summarizes all the details stated below.

### 1. Datasets Used
**_To be populated_**

1. [IMDB non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
2. Movie Revenue Dataset (df_revenue - scraped from www.boxofficemojo.com)
3. Movie Ratings Dataset (df_ratings - scraped from https://www.imdb.com/)
4. [Movie Lens 1m Dataset](https://grouplens.org/datasets/movielens/1m/)
<br><br>
For the scraping of ratings, due to the limitations in computational resources, I was not able to complete the scraping of the revenues and ratings of all the entries in dataset #1. Hence, to train the recommender engine, I used the Movie Lens 1m Dataset, which was more readily available.<br><br>
The following shows a fact table of the datasets used (for source 1 to 3):
<br><br>

<img src="images/fact_table.png?raw=true"/>

### 2. Data Analysis & Key Insights

1. **There has been a steady decline of total voters in IMDb**
<img src="images/P1_insight1.png?raw=true"/>

2. **Preference for certain genres have changed over the years**
<img src="images/P1_insight2_1.png?raw=true"/>
<img src="images/P1_insight2_2.png?raw=true"/>

3. Revenue figures for successful diretors can rake up to billions of dollars
<img src="images/P1_insight3.png?raw=true"/>

4. Revenues took a hit during covid, but some movie genres see rebounds whereas some stayed flat

<img src="images/P1_insight4.png?raw=true"/>

### 3. Recommender Engines

 
- Collaborative Filtering model with Tensorflow<br>
This code segment builds a collaborative filtering neural network using Tensorflow/Keras for
recommendation systems. It utilises user and movie embeddings, concatenated and processed through dense layers with ReLU activations and dropout regularization.<br>
This model was trained to predict movie ratings and aims to minimize the mean squared error loss.
 
```python3
K = 10
mu = df_train['rating'].mean()
epochs=100
reg = 0.1

# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u) # (N, 1, K)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
x = Concatenate()([u_embedding, m_embedding]) # (N, 2K)

# the neural network
x = Dense(1000)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(500)(x)
x = Dropout(0.5)(x)
x = Dense(250)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(1)(x)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  optimizer='adam',
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.user_id.values, df_train.movie_id.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  callbacks = [early_stopping],
  batch_size=128,
  validation_data=(
    [df_test.user_id.values, df_test.movie_id.values],
    df_test.rating.values - mu
  )
)
```

<img src="images/model.jpg?raw=true"/>

- Cosine-similarity model<br>
This second cosine-similarity model takes inputs from users through a Streamlit interface of the movies they enjoyed watching, and saves it in variable __watched_features__. <br>
It then compares the selected features with the features in the IMDB dataframe, and then shortlists the top 20 movies most similar to the movies inputted. These 20 movies are then sorted in accordance to descending average ratings and recommendation propensity (a metric engineered from average ratings and the age of the movie. The higher the recommendation propensity, the more recent and also popular the movie is). The user will then be recommended with the top 3 selections.

```python3
watched_features = rec_df[rec_df.index.isin(watched_movie_ids)].drop(['primaryTitle', 'popularity_score', 'recommendation_propensity'], axis=1)

features = rec_df.drop(['primaryTitle', 'popularity_score', 'recommendation_propensity'], axis=1)

similarity_to_watched = cosine_similarity(features, watched_features)
average_similarity = similarity_to_watched.mean(axis=1)

# Set the similarity of watched movies to a low value
average_similarity[rec_df.index.isin(watched_movie_ids)] = -1

# Get the indices of the movies with the top 5 highest average similarity
recommended_indices = average_similarity.argsort()[-20:][::-1]

# Get the movie IDs of the recommended movies
recommended_movie_ids = rec_df.index[recommended_indices]
recommended_movies = rec_df.loc[recommended_movie_ids].sort_values(['recommendation_propensity','averageRating'],ascending=False)
recommended_movies = recommended_movies[recommended_movies['averageRating']>=5]
recommended_movies = recommended_movies.iloc[:3]
```

### 4. Conclusion and Future Work
**_To be populated_**

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
