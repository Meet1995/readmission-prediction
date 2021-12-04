import json
import pandas as pd
from tqdm import tqdm 
from pep_modules import patientai as pai

dim_lst = json.loads(input("Enter the embedding dimension list >>"))

cross_df = pd.read_pickle(r'../data/cross_data.pkl')

for dim in tqdm(dim_lst):
    word2vec = pai.train_word2vec(cross_df['journey'],size=dim)
    word2vec.save(f'../saved_models/word2vec_{dim}.model')

print("All the models saved to project home saved_models directory!")