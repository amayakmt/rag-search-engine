# helper function to load movies to memory
import json
from config import MOVIES

def load_movies() -> dict:
    with open(MOVIES, "r") as m:
        data = json.load(m)

        return data["movies"]
    