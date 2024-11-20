import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import load_model
def detect(wallet):
    BLOCKCHAIN_API_URL = "https://api.blockchain.com/v3/wallet"
    response = requests.get(f"{BLOCKCHAIN_API_URL}/{wallet}/transactions")
    if response.status_code == 200:
        return response.json()
    return None
    # response = requests.get(f'https://dirtyapi.replit.app/api/{wallet}')
    # dtf1 = response.json()
    del dtf1["ERC20MostSentTokenType"]
    del dtf1["ERC20MostRecTokenType"]
    df1 = pd.DataFrame([dtf1])  
    if 'Address' in df1.columns:
        df1.drop(['Address'], axis=1, inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df1)
    loaded_model=load_model(r'/content/DMD.h5')
    predictions1 = loaded_model.predict(df_scaled)
    binary_predictions = (predictions >= 0.5).astype(int)
    return accuracy_score([1,0],binary_predictions)
