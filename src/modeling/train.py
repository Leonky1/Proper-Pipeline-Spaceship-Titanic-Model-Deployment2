import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import get_model_pipeline

def train_model():
    df = pd.read_csv('data/raw/train.csv')
    
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0].astype(int)
    
    df = df.fillna(method='ffill') 

    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 
                'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
                'Group', 'Deck', 'Side']
    X = df[features]
    y = df['Transported'].astype(int)

    pipeline = get_model_pipeline()
    pipeline.fit(X, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/model.pkl')

if __name__ == "__main__":
    train_model()