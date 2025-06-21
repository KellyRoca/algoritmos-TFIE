import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.api import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
from prophet import Prophet
from math import ceil
import matplotlib.pyplot as plt
import os
import warnings
from fpdf import FPDF
from PIL import Image
import time
start_time = time.time()

warnings.filterwarnings("ignore")

def calc_mape(real, pred):
    real = pd.Series(real).reset_index(drop=True)
    pred = pd.Series([max(0, round(p)) for p in pred]).reset_index(drop=True)
    df_eval = pd.DataFrame({'real': real, 'pred': pred})
    df_eval = df_eval[df_eval['real'] != 0].dropna()
    if df_eval.empty or len(df_eval) != len(real):
        return np.inf
    return mean_absolute_percentage_error(df_eval['real'], df_eval['pred']) * 100

def ses_forecast_opt(serie):
    from scipy.optimize import minimize_scalar

    def forecast(alpha, data):
        pred = [np.nan]
        prev = data[0]
        for t in range(1, len(data)):
            val = alpha * data[t - 1] + (1 - alpha) * prev
            val = ceil(val)
            pred.append(val)
            prev = val
        return pred

    def objective(alpha):
        pred = forecast(alpha, serie)
        return calc_mape(serie[1:], pred[1:])

    res = minimize_scalar(objective, bounds=(1e-10, 1 - 1e-10), method='bounded')
    alpha_opt = round(res.x, 10)
    pred_hist = forecast(alpha_opt, serie)

    forecast_fut = []
    prev = pred_hist[-1]
    for _ in range(6):
        val = alpha_opt * serie[-1] + (1 - alpha_opt) * prev
        val = ceil(val)
        forecast_fut.append(val)
        prev = val

    return pred_hist, forecast_fut, alpha_opt, calc_mape(serie[1:], pred_hist[1:])

Tk().withdraw()
ruta = askopenfilename(title="Selecciona tu archivo Excel", filetypes=[("Excel files", "*.xlsx")])
df = pd.read_excel(ruta)
df['Mes'] = pd.to_datetime(df['Mes'], errors='coerce')
df.dropna(subset=['Mes'], inplace=True)
df.rename(columns={'Producto': 'SKU'}, inplace=True)
df = df.sort_values(by=['SKU', 'Mes'])

if not os.path.exists("imagenes_modelos"):
    os.makedirs("imagenes_modelos")

resumen = []
mejores_modelos = []

for sku, grupo in df.groupby('SKU'):
    serie = grupo['Demanda real'].dropna().tolist()
    fechas = grupo['Mes'].tolist()
    if len(serie) < 12:
        continue

    resultados = {}
    parametros_usados = {}
    pronosticos_futuros = {}
    historicos_modelos = {}

    try:
        adf_pvalue = round(adfuller(serie)[1], 5)
    except:
        adf_pvalue = np.nan

    try:
        stl = STL(serie, period=12, robust=True).fit()
        seasonal_strength = round(1 - (np.var(stl.resid) / (np.var(stl.resid + stl.seasonal))), 3)
    except:
        seasonal_strength = np.nan

    estacionalidad_fuerte = seasonal_strength is not np.nan and seasonal_strength > 0.5
    es_estacionaria = adf_pvalue < 0.05

    try:
        hist = [np.mean(serie[i-3:i]) for i in range(3, len(serie))]
        mm3_forecast = [np.mean(serie[-3:])] * 6
        resultados['MM3'] = calc_mape(serie[3:], hist)
        parametros_usados['MM3'] = 'ventana=3'
        pronosticos_futuros['MM3'] = mm3_forecast
        historicos_modelos['MM3'] = [np.nan]*3 + hist
    except:
        resultados['MM3'] = np.inf

    try:
        hist = [np.mean(serie[i-4:i]) for i in range(4, len(serie))]
        mm4_forecast = [np.mean(serie[-4:])] * 6
        resultados['MM4'] = calc_mape(serie[4:], hist)
        parametros_usados['MM4'] = 'ventana=4'
        pronosticos_futuros['MM4'] = mm4_forecast
        historicos_modelos['MM4'] = [np.nan]*4 + hist
    except:
        resultados['MM4'] = np.inf

    try:
        ses_pred, ses_fut, alpha, mape = ses_forecast_opt(serie)
        resultados['SES'] = mape
        parametros_usados['SES'] = f'alpha={alpha}'
        pronosticos_futuros['SES'] = ses_fut
        historicos_modelos['SES'] = ses_pred
    except:
        resultados['SES'] = np.inf

    try:
        model = Holt(serie).fit()
        pred = model.predict(start=1, end=len(serie)-1)
        forecast = model.forecast(6)
        resultados['Holt'] = calc_mape(serie[1:], pred)
        parametros_usados['Holt'] = 'default'
        pronosticos_futuros['Holt'] = forecast
        historicos_modelos['Holt'] = [np.nan] + list(pred)
    except:
        resultados['Holt'] = np.inf

    if estacionalidad_fuerte:
        try:
            model = ExponentialSmoothing(serie, trend='add', seasonal='add', seasonal_periods=12).fit()
            pred = model.predict(start=1, end=len(serie)-1)
            forecast = model.forecast(6)
            resultados['Holt-Winters'] = calc_mape(serie[1:], pred)
            parametros_usados['Holt-Winters'] = 'trend=add, seasonal=add'
            pronosticos_futuros['Holt-Winters'] = forecast
            historicos_modelos['Holt-Winters'] = [np.nan] + list(pred)
        except:
            resultados['Holt-Winters'] = np.inf

    try:
        arima_model = auto_arima(serie, seasonal=False, d=0 if es_estacionaria else None, suppress_warnings=True)
        forecast = arima_model.predict(n_periods=6)
        pred = arima_model.predict_in_sample()
        resultados[f'ARIMA{arima_model.order}'] = calc_mape(serie, pred)
        parametros_usados[f'ARIMA{arima_model.order}'] = str(arima_model.order)
        pronosticos_futuros[f'ARIMA{arima_model.order}'] = forecast
        historicos_modelos[f'ARIMA{arima_model.order}'] = list(pred)
    except:
        resultados['ARIMA'] = np.inf

    if estacionalidad_fuerte:
        try:
            d_value = 0 if es_estacionaria else None
            sarima_model = auto_arima(
                serie, start_p=0, start_q=0, d=d_value,
                seasonal=True, m=12,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2, D=1,
                trace=False, error_action='ignore',
                suppress_warnings=True, stepwise=True
            )
            forecast = sarima_model.predict(n_periods=6)
            pred = sarima_model.predict_in_sample()
            resultados[f'SARIMA{sarima_model.order}x{sarima_model.seasonal_order}'] = calc_mape(serie, pred)
            parametros_usados[f'SARIMA{sarima_model.order}x{sarima_model.seasonal_order}'] = str(sarima_model.order) + " x " + str(sarima_model.seasonal_order)
            pronosticos_futuros[f'SARIMA{sarima_model.order}x{sarima_model.seasonal_order}'] = forecast
            historicos_modelos[f'SARIMA{sarima_model.order}x{sarima_model.seasonal_order}'] = list(pred)
        except:
            resultados['SARIMA'] = np.inf

    try:
        prophet_df = pd.DataFrame({'ds': fechas, 'y': serie})
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=6, freq='MS')
        forecast = m.predict(future)
        pred = forecast['yhat'][:-6].tolist()
        fut = forecast['yhat'][-6:].tolist()
        resultados['Prophet'] = calc_mape(serie, pred)
        parametros_usados['Prophet'] = 'default'
        pronosticos_futuros['Prophet'] = fut
        historicos_modelos['Prophet'] = pred
    except:
        resultados['Prophet'] = np.inf

    for modelo in resultados:
        resumen.append({
            'SKU': sku,
            'Modelo': modelo,
            'MAPE (%)': resultados[modelo],
            'Parámetros': parametros_usados.get(modelo, ''),
            'Pronóstico Próximos 6': [max(0, round(x)) for x in pronosticos_futuros.get(modelo, [])],
            'Estacionalidad': seasonal_strength,
            'Estacionaridad (ADF)': adf_pvalue
        })

    mejor_modelo = min(resultados.items(), key=lambda x: x[1])
    mejores_modelos.append({
        'SKU': sku,
        'Mejor Modelo': mejor_modelo[0],
        'MAPE (%)': mejor_modelo[1],
        'Parámetros': parametros_usados.get(mejor_modelo[0], ''),
        'Pronóstico Próximos 6': [max(0, round(x)) for x in pronosticos_futuros.get(mejor_modelo[0], [])],
        'Estacionalidad': seasonal_strength,
        'Estacionaridad (ADF)': adf_pvalue
    })

modelos_simple = ["MM3", "MM4", "SES"]
final_modelos = []

for mejor in mejores_modelos:
    if mejor["Mejor Modelo"] in modelos_simple:
        sku_modelos = [r for r in resumen if r["SKU"] == mejor["SKU"] and r["Modelo"] not in modelos_simple]
        if sku_modelos:
            segundo_mejor = min(sku_modelos, key=lambda x: x["MAPE (%)"])
            final_modelos.append({
                "SKU": mejor["SKU"],
                "Modelo Seleccionado": segundo_mejor["Modelo"],
                "MAPE (%)": segundo_mejor["MAPE (%)"],
                "Parámetros": segundo_mejor["Parámetros"],
                "Pronóstico Próximos 6": segundo_mejor["Pronóstico Próximos 6"],
                "Modelo Preferible": mejor["Mejor Modelo"]
            })
    else:
        final_modelos.append({
            "SKU": mejor["SKU"],
            "Modelo Seleccionado": mejor["Mejor Modelo"],
            "MAPE (%)": mejor["MAPE (%)"],
            "Parámetros": mejor["Parámetros"],
            "Pronóstico Próximos 6": mejor["Pronóstico Próximos 6"],
            "Modelo Preferible": "-"
        })

# Da excel de todos los modelos comparados así como resultados de prueba de estacionalidad y estacionaridad
pd.DataFrame(resumen).to_excel("visso_resultados_totales_de_modelos.xlsx", index=False)
# Da excel para cada uno de los skus, sus pronosticos y el mejor modelo elegible
pd.DataFrame(final_modelos).to_excel("visso_modelos_seleccionados.xlsx", index=False)

# Gráficos solo del modelo final preferible
df_final = pd.DataFrame(final_modelos)
for sku in df_final['SKU'].unique():
    grupo = df[df['SKU'] == sku]
    serie = grupo['Demanda real'].dropna().tolist()
    modelo_final = df_final[df_final['SKU'] == sku]['Modelo Seleccionado'].values[0]

    # Históricos y futuros
    historico = None
    futuro = None
    for r in resumen:
        if r['SKU'] == sku and r['Modelo'] == modelo_final:
            futuro = r['Pronóstico Próximos 6']
            break
    for r in resumen:
        if r['SKU'] == sku and r['Modelo'] == modelo_final:
            historico = r.get('Histórico') or None

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(serie)), serie, label='Demanda real', marker='o')
    if historico:
        plt.plot(range(len(historico)), historico, linestyle='--', label=f"{modelo_final} (hist)")
    if futuro:
        x = range(len(serie), len(serie) + len(futuro))
        plt.plot(x, [max(0, round(x)) for x in futuro], label=f"{modelo_final} (futuro)")
    plt.title(f"{sku} - Modelo seleccionado: {modelo_final}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"imagenes_modelos/{sku.replace('/', '_')}.png")
    plt.close()

# Exportar imágenes a PDF
pdf = FPDF()
img_folder = "imagenes_modelos"
img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    image = Image.open(img_path)
    width, height = image.size
    width, height = width * 0.264583, height * 0.264583
    pdf.add_page()
    pdf.image(img_path, x=10, y=10, w=min(width, 190))
pdf.output("visso_forecast_graficas.pdf")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ Tiempo total de ejecución: {elapsed_time:.2f} segundos")
