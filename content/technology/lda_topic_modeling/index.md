---
title: "LDA and Optuna"
date: 2021-02-02T06:00:00+01:00
draft: false
hideLastModified: true
summary: "What are topics, really, but the undercurrents \ 
of conversation, hints of broader themes that \ 
pepper throughout our speech like spices..."
summaryImage: "ratatouille_taste.png"
tags: ["tutorial", "nlp", "advanced", "intermediate", "topic modeling", "lda", "hyperparameter", "optuna"]
---


What are topics, really, but the undercurrents of conversation, hints of broader themes that pepper throughout our speech like spices wafting from a richly flavored stew... 

We are going to flip the script and look at recipes through the lenses of topic modeling to see what general categories arise. In this blog, we'll cover the Latent Dirichlet Algorithm (LDA), hyperparameter tuning, and other technical topics.

If you just need a coding example, you can skip down to [here](#modeling).


## What is Topic Modeling?

Topic modeling is, at its simplest, the process of finding words that frequently show up together. To a person, these co-occurring words usually end up suggesting a theme. When you hear "bark", "nose", and "leash", what comes to mind? Even without "dog" appearing in there, you may have recognized that it ties these words together - thus, voila, "dog" is your topic.

![A dog says hello](https://tenor.com/view/hi-hello-dog-there-oh-gif-11420655.gif)

One of the most common ways to do topic modeling is **Latent Dirichlet Algorithm (LDA)**. This is an **unsupervised** algorithm, meaning that it does not require ground truth information, but instead arises patterns found inherently in the data. This can be really cool to identify hidden themes among a bunch of documents, which can give you a high-level understanding without having to go through and read every article. 

A cautionary note: words get used in a variety of contexts. "Bark" could be a dog's bark or the bark of a tree. Since language is tricky this way, we don't want to say the presence of a word definitely indicates a topic - instead, what we'll do is say that there's a certain probability of a word representing each topic; we can then roll these up to get a probability for each document/recipe as to whether it contains each topic.

In a recipe sense, think of this as onions - onions show up everywhere, so they're not a great indicator for, say, Italian vs. Mexican vs. Indian food. On the other hand, penne, chipotle peppers, and turmeric are more specific - they'll indicate a higher probability of those individual cuisines. A recipe with onions and chipotle peppers will have a very small but not-0 probability of belonging to the Italian topic (thanks to onions) and a very high probability of belonging to the Mexican topic. 

I recognize this is a lot to take in. If you're interested and want to learn more, [this](http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/) is a good resource for explaining topic modeling and its strengths and weaknesses. [This article](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158) goes into additional depth on LDA so you can understand what's happening under the hood. 

## Hyperparameter Optimization

Next, to get our best possible set of modelled topics, we need to do some hyperparameter tuning. A **hyperparameter** is a parameter, or option, that you must decide upon when training your machine learning model. Think of it this way - you train a model so it can make 'decisions' (e.g. classifications, detections) without you. Hyperparameters help define the form of this training, and thus what kind of model you're going to make. If you're into Pokemon, a tangible comparison to this is how Eevee can evolve into a multitude of different variations based on the conditions under which you train it.  Eevee's environmental conditions are metaphorically like hyperparameters, in that they affect its training and eventual outcomes.

<!-- ![Eevee's different evolutionary paths](eevee_evolutions.png#center) -->

{{% figure src="eevee_evolutions.png" width="500" caption="Eevee's different evolutionary paths are similarly influenced by its training and environmental conditions. Image originally from LevelSkip" %}}

LDA has three hyperparameters that we'll need to tune:

* **n_topics** - the LDA algorithm requires the number of topics upfront. This can be tricky since the whole reason you're running LDA is to learn what topics exist.

* **alpha** -  topic-document density; the larger alpha, the more topics you expect to be in a document, and vice versa.

* **eta** - topic-word density; the larger beta, the more words from your documents are part of topics , and vice versa

[This page](https://www.thoughtvector.io/blog/lda-alpha-and-beta-parameters-the-intuition/#:~:text=LDA%20Alpha%20and%20Beta%20Parameters%20-%20The%20Intuition.,via%20an%20open%20source%20implementation%20like%20Python%E2%80%99s%20gensim) explains alpha and eta in a really nice and coherent way.

You can spend a lot of time tinkering with different values for your hyperparameters to try to find what gives you the best result. However, not today! Today we use Optuna. I'll show you how to do so below.


-----
## The Code

For this example, I will use this open-source dataset of 250,000 recipes: [https://eightportions.com/datasets/Recipes/]().

You will need Python and the following libraries, installable from pip or conda: **gensim**, **optuna**, **pyarrow**, and **pandas**. The full example code from file ingestion through modeling is available [here](https://github.com/abargar/optuna_example). 

### Feature/Token Creation

[This](https://github.com/abargar/optuna_example/blob/main/process_ingredients.py) is the script where I process features from files. In general, this is what I do:

1.  I iterate through each file/recipe to pull out the recipe name and ingredient list (lines 38-47)

2. I combine the name and ingredients into a single string (lines 50-52)

3. I get rid of punctuation and numbers (lines 29-33)

4. I get rid of [useless words](https://github.com/abargar/optuna_example/blob/main/useless_words.txt), like "a", "into", "advertisement", and "fresh" (I'm cynical. Also, line 34)

5. I save the resulting tokens to a file. 


### Modeling

[This](https://github.com/abargar/optuna_example/blob/main/model.py) is the script to generate the topic model.

First, we need to prepare the corpus - the list of tokens - into a format readable by the model. Because we cannot pass corpora_dict and corpus directly into optuna's optimize function, we define them here as global variables. 


```python
token_df = pd.read_parquet("data/tokenized_ingredients.parquet")
tokens_list = token_df.tokens.values
corpora_dict = corpora.Dictionary(tokens_list)
corpus = [corpora_dict.doc2bow(tokens) for tokens in tokens_list]
```

In order to optimize performance, we need something to optimize towards. [Coherence](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_coherence_model_selection.ipynb) measures well the topics are represented by their highest probability words - how frequently do these words actually appear together - and is correlated with how human-interpretable the end topics are. This function handles the coherence calculation.

```python
def compute_coherence(model, corpus, corpora_dict):
    coherence_model_lda = CoherenceModel(
        model=model,
        texts=corpus,
        corpus=None,
        dictionary=corpora_dict,
        coherence="c_v",
    )
    return coherence_model_lda.get_coherence()
```

This is the crux of Optuna- defining the objective that we will be optimizing. Note that here we define uniform distributions for alpha, eta, and the number of topics (ntopics) for Optuna to sample from. Next, we define our most ideal outcome score. Maximum coherence is 1, but this can be tricky as it usually indicates an underlying problem with the data. Thus, we set our ideal score to a more realistic 0.8. The model is then created with these parameters and tested over the dataset. Post-creating the model, the coherence score is calculated and the model outputs are stored via the function *write_model_results*. The output of this function is the difference between the computed coherence score and the ideal 0.8 score.

```python
def objective(trial):
    alpha = trial.suggest_uniform("alpha", 0.01, 1)
    eta = trial.suggest_uniform("eta", 0.01, 1)
    ntopics = trial.suggest_uniform("num_topics", 10, 50)
    ideal_score = 0.8
    model = gensim.models.LdaMulticore(
        workers=7,
        corpus=corpus,
        id2word=corpora_dict,
        num_topics=ntopics,
        random_state=100,
        passes=3,
        alpha=alpha,
        eta=eta,
        per_word_topics=True,
    )
    coherence_score = compute_coherence(model, tokens_list, corpora_dict)
    print(f"Trial {trial.number} coherence score: {round(coherence_score,3)}")
    write_model_results(trial, model, coherence_score)
    coherence_score_diff = abs(ideal_score - coherence_score)
    return coherence_score_diff
```

Lastly - outside of any function, in *main* - we create the study. Here we pass the objective function we want to optimize and how many trials we want to run for. As it runs, it will print the ongoing best trial and best trial parameters. Don't freak out if the score looks different in this logger, as optuna tracks the difference between calculated and ideal coherence scores. 

```python
study = optuna.create_study()
study.optimize(objective, n_trials=100)
Path(f"models").mkdir(exist_ok=True)

logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"Best trial info: {study.best_trial}")
```

At the end, open the file "model_results.csv". Your top model will be the one with the highest coherence score. All of the accompanying goodies, e.g. top words per topic, will be stored in the file location "models/trial_{trial_number}."


-----
## The Results

Finally we have completed the modeling process. Let's see what we've got!

![Drumroll please!](https://tenor.com/view/sing-sing-universal-drum-roll-buster-moon-gif-7389474.gif)

My best trial was the 61st, with 10 topics, an alpha of 0.45 ( fairly average topic-document density), and an eta of 0.06 (few words co-occur frequently). My coherence was 0.59, decent given the limited amount of preprocessing. 

### Top Words

**0 -** **Italian or French?**: chopped, oil, pepper, olive, salt, finely, leaves, black, lemon, garlic, red, parsley,...

**1 - Savory bread & pizza making**: bread, slices, salt, egg, oil, flour, sliced, sugar, water, cheese, 

**2 -** I think this one may be **Mexican**: sauce, cilantro, minced, beef, large, red, chile, cloves, corn, peeled, 

**3 -** **Also Italian?**:  chopped, ounce, pepper, garlic, salt, can, oil, taste, olive, dried, ....

**4 -** **Potato time**: cheese, cream, ounce, chopped, shredded, potatoes, pepper, salt, butter, cheddar,

**5 -** **Baking**: sugar, butter, flour, vanilla, salt, extract, baking, cream, allpurpose, chocolate,

**6 -** **"Asian"**: sauce, chicken, pepper, chopped, oil, salt, garlic, rice, powder, onion, minced, black, pork, soy, 

**7 -** **Cocktails**: juice, ounce, orange, sugar, lemon, lime, as, water, ice, fluid, pineapple, white, frozen, garnish

**8 -** **Salads**: sliced, thinly, peeled, cut, oil, salad, vinegar, white, salt, juice, chopped, apple, pepper, green, 

**9 -** **???:** pepper, chopped, salt, shrimp, diced, oil, black, peeled, butter, large, red, sliced,

As anticipated, some topics are more coherent, and others are made clearer with access to a larger set. [Here](https://github.com/abargar/optuna_example/blob/main/example_50topwords.txt) is my full list with up to 50 words per topic if you'd like to peruse. 

### Most Topical Recipes

Next, I'd like to see which recipes are most representative of these topics. This might shed some light on the topics that are harder to crack (looking at you, 1, 4, and 10).

For brevity, I'll just include the top 3, but the full list is [here](https://github.com/abargar/optuna_example/blob/main/topical_recipes.txt) :

**0 -** Marinated Lamb Shoulder Chops, Red Wine-Rosemary Grilled...

**1 -** Scali, Eggplant Parmigiana, Fougasse, Italian Loaf, Challah

**2 -** Dos Toros Quesidilias, Fish Tacos, Chorizo Burger

**3 -** Chili, Chili, Stew 

**4 -** 3 Cheese Enchilada, Potato Casserole, Mac and Cheese

**5 -** Chocolate Cake, Chocolate Chip Cookies

**6 -** Chicken Lettuce Wraps, Chicken Chow Mein, Sweet and Spicy Pork

**7 -** Punch, Sangria, Terrine

**8 -** Apple and Raisin Slaw, Sush-Roll Rice Salad, Tuna Maki

**9 -** Fettucini with Rabbit, Roasted Turkey, Spitfire Shrimp

----

### Take Aways

From an experimental perspective - hey, that wasn't bad! This approach found coherent groupings that make sense on both a word-based and document/recipe-based perspective.

From a data perspective - this is a great example of how your dataset impacts what you find. My Mexican partner was deeply unimpressed by cluster number 3 and what they did to jicama in the salad section. We would certainly find different groupings had we worked with a dataset that was less focused on an English-speaking audience, or at least had broader representation.

Even if I don't cook from this dataset. I'm simply satisfied we made it through a complex problem, and found something interesting at the end. That's a wrap, folks! Bon appetit! 

![Bon appetit!](https://media.gifs.nl/ratatouille-gifs-eC41Kp.gif)