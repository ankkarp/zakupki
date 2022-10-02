from joblib import load
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', type=str)
# parser.add_argument('-c', '--country', type=str)
# parser.add_argument('-l', '--min_price', type=int)
# parser.add_argument('-h', '--max_price', type=int)
#
args = parser.parse_args()

if __name__ == "__main__":
    tfid = load('tfid.joblib')
    data = pd.read_csv("clean_data.csv")
    products = pd.read_csv("clean_products.csv")
    products.to_csv("clean_products.csv")
    embeddings = tfid.transform(data["product_characteristics"].values)
    similarity_matrix = cosine_similarity(tfid.transform([args.query]), embeddings)
    output = data.iloc[np.argsort(similarity_matrix)[:, ::-1][0]].head(100)
    output.drop(output.columns[output.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    output.to_csv("output/recommendations.csv")