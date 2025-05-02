import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df_villaft = pd.read_csv("/content/drive/MyDrive/Scraped Data Bukitvista/data_rentals_bukitvista.csv")

# Pastikan kolom harga berbentuk float
df_villaft['Price_per_day_usd'] = pd.to_numeric(df_villaft['Price_per_day_usd'], errors='coerce')

# Load vectorizer
with open("location_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

location_vectors = vectorizer.transform(df_villaft['Location'].astype(str))

def recommend_properties(user_location, user_budget_usd, top_n=5):
    try:
        user_location_vec = vectorizer.transform([user_location.lower().strip()])
        loc_sim = cosine_similarity(user_location_vec, location_vectors).flatten()

        price_diff = np.abs(df_villaft['Price_per_day_usd'] - user_budget_usd)
        price_score = 1 / (1 + price_diff)

        final_score = (0.6 * loc_sim) + (0.4 * price_score)

        df_villaft['score'] = final_score
        top_properties = df_villaft.sort_values(by='score', ascending=False).head(top_n)

        return top_properties[['Villa', 'Location', 'Price_per_day_usd', 'Bedrooms', 'Bathrooms', 'Url']].to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}
