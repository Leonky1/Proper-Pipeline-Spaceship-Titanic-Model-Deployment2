import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/train.csv") 

    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0].astype(int)

    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 
                'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
                'Group', 'Deck', 'Side']
    
    #Handling Missing Values
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    mappings = {
        'HomePlanet': {'Earth': 0, 'Europa': 1, 'Mars': 2},
        'CryoSleep': {False: 0, True: 1},
        'Destination': {'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2},
        'VIP': {False: 0, True: 1},
        'Deck': {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7},
        'Side': {'P': 0, 'S': 1}
    }
    for col, m in mappings.items():
        df[col] = df[col].map(m).astype(int)

    X = df[features]
    y = df['Transported'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

    joblib.dump(scaler, "artifacts/preprocessor.pkl")

    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)

    return train_df, test_df

if __name__ == "__main__":
    preprocess()
