# -*- coding: utf-8 -*-


import logging
import typer
from typing import Optional
from pathlib import Path
from Recommendations.model import Recommender

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode='a')

logger = logging.getLogger(__name__)

app = typer.Typer()

rec = Recommender()

@app.command()
def train(dataset: Optional[Path] = Path("./data/ratings_train.dat")):
	rec.train(dataset)
	logging.info(f"Model created!!!")

@app.command()
def predict(user_id: Optional[str], m: Optional[str] = 5):
	movie_id,movie_rating = rec.predict(int(user_id),num_recommendations=int(m))
	logging.info(f"Movie_MAS: {movie_id}\nMovie_rating: {movie_rating}")
	print(f"Movie_MAS: {movie_id}\nMovie_rating: {movie_rating}")
	return movie_id, movie_rating

@app.command()
def reload():
    rec.warmup()
    pass

if __name__ == '__main__':
    app()