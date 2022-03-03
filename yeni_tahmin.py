"""Ornek tahmin scripti.

Yazacaginiz kod, arguman olarak bir dosya yolu almali ve sonuclari
iki kolon ['id', 'SalePrice'] formatinda "output/predictions.csv"
dosyasina kaydetmelidir. Asagida rastgele bir tahmin yapip hedef dosyayi kaydeden
ornek bir kod bulunmaktadir.
"""

import sys
import numpy as np
import pandas as pd
import joblib

OUTPUT_PATH = 'output/predictions.csv'
MODEL_YOLU = "yeni_tahminci.joblib"



def preprocesing(data):
  data = data.drop(["Alley", "PoolQC", "MiscFeature", "Fence", "FireplaceQu"], axis=1)
  data = data.fillna(0)
  data = pd.get_dummies(data, drop_first=True)
  kolonlar = joblib.load("yeni_kolonlar")
  data = data.reindex(kolonlar, axis=1)
  data = data.fillna(0)

  return data

if __name__ == "__main__":
  filepath = sys.argv[1]
  df = pd.read_csv(filepath)
  model = joblib.load(MODEL_YOLU)
  X = preprocesing(df)
  y_pred = model.predict(X)
  pd.Series(y_pred,  index=df["Id"], name="SalePrice").to_csv(OUTPUT_PATH, index=True)