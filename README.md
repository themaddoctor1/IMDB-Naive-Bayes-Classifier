# IMDB Rating Predictor

This application is a machine learning model that given a set of descriptive
labels will attempt to predict the rating of a movie with those attributes.

## About

The program is designed to predict movie ratings using a Naive Bayes model.
The model was trained on the IMDB 5000 Movie Dataset available, which can be found
[here](https://data.world/popculture/imdb-5000-movie-dataset), and computes results by
measuring the frequency of each label. For instance, the program would measure
the probability of an actor being in a movie (given by ```actor_(.*)_name```) given
its rating. For numerical quantities (continuous variables), the program uses
a normal distribution to measure the probability density of a given variable.
For instance, the ```duration``` variable, which measures the length of a
film, assumes that the length is normally distributed.

## Dependencies

The current implementation uses Numpy for computations on lists, as well as Matplotlib
for displaying statistical charts. These can be installed via

```
pip install numpy matplotlib
```

## Usage

To use the program, one can run:

```
python class.py <csv_file> (<label> <value>)*
```

Where ```(<label> <value>)*``` represents an arbitrary number of uses of label-value
pairs. For instance, to predict based on duration and director name, one would run:

```
python class.py ./movie_metadata.csv duration 162 director_name "Christopher Nolan"
```

As of the latest test, this result should yield the same rating as was given
to "The Dark Knight Rises", which was a 162 minute movie directed by 
Christopher Nolan.

