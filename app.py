import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Mumbai AQI Estimation System",
    page_icon="🌫️",
    layout="wide"
)

# ── LOAD MODELS ──
@st.cache_resource
def load_models():
    regressor    = joblib.load('xgboost_regressor.pkl')
    classifier   = joblib.load('xgboost_classifier.pkl')
    le_pollutant = joblib.load('pollutant_encoder.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return regressor, classifier, le_pollutant, feature_cols

regressor, classifier, le_pollutant, feature_cols = load_models()

season_map = {'monsoon': 0, 'post_monsoon': 1, 'summer': 2, 'winter': 3}

# ── ALERT FUNCTION ──
def get_alert(aqi):
    if aqi <= 50:
        return 'Good', '#2ecc71', 'Air quality is satisfactory. Enjoy outdoor activities freely.', 'No precautions needed.'
    elif aqi <= 100:
        return 'Satisfactory', '#a8e063', 'Air quality is acceptable for most people.', 'Sensitive individuals should limit prolonged outdoor activity.'
    elif aqi <= 200:
        return 'Moderate', '#f39c12', 'Sensitive groups may experience health effects.', 'Wear a mask outdoors. Children and elderly should limit exertion.'
    elif aqi <= 300:
        return 'Poor', '#e74c3c', 'Everyone may begin to experience health effects.', 'Avoid outdoor activity. Keep windows closed. Use air purifier if available.'
    elif aqi <= 400:
        return 'Very Poor', '#8e44ad', 'Health warnings of emergency conditions for entire population.', 'Stay indoors. Wear N95 mask if going out. Avoid all physical outdoor activity.'
    else:
        return 'Severe', '#2c3e50', 'HEALTH EMERGENCY — Entire population likely to be affected.', 'Do NOT go outside. Seal windows. Seek medical attention if experiencing symptoms.'

# ── HEADER ──
st.title('🌫️ Mumbai AQI Estimation & Alert System')
st.markdown('**ME228 Course Project** | Prakrati · Snehal · Prashil · Jishan | IIT Bombay')
st.markdown('---')

# ── INPUTS ──
st.header('📋 Enter Current Conditions')
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('🕐 Time & Context')
    year        = st.selectbox('Year', [2022, 2023, 2024, 2025, 2026], index=3)
    hour        = st.slider('Hour of Day (0-23)', 0, 23, 8)
    day         = st.slider('Day of Month', 1, 31, 15)
    day_of_week = st.selectbox('Day of Week', [0,1,2,3,4,5,6],
                    format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    is_weekend  = 1 if day_of_week >= 5 else 0
    is_peak     = 1 if (7 <= hour <= 10) or (17 <= hour <= 21) else 0
    season_name = st.selectbox('Season', ['monsoon','post_monsoon','summer','winter'],
                    format_func=lambda x: x.replace('_',' ').title())
    season      = season_map[season_name]
    festival    = int(st.checkbox('Festival Period (e.g. Diwali)?'))
    crop_burn   = int(st.checkbox('Crop Burning Season Active?'))

with col2:
    st.subheader('🌤️ Weather')
    temp_c        = st.slider('Temperature (°C)', 10.0, 45.0, 28.0, 0.5)
    humidity      = st.slider('Humidity (%)', 10, 100, 70)
    dew_point     = st.slider('Dew Point (°C)', 0.0, 35.0, 20.0, 0.5)
    wind_speed    = st.slider('Wind Speed (km/h)', 0.0, 60.0, 12.0, 0.5)
    wind_dir      = st.slider('Wind Direction (degrees)', 0, 360, 180)
    wind_gusts    = st.slider('Wind Gusts (km/h)', 0.0, 80.0, 18.0, 0.5)
    precipitation = st.slider('Precipitation (mm)', 0.0, 50.0, 0.0, 0.5)
    is_raining    = int(st.checkbox('Currently Raining?'))
    heavy_rain    = int(st.checkbox('Heavy Rain?'))
    pressure      = st.slider('Pressure (hPa)', 990.0, 1025.0, 1010.0, 0.5)
    cloud_cover   = st.slider('Cloud Cover (%)', 0, 100, 30)

with col3:
    st.subheader('🏭 Pollutant Readings (µg/m³)')
    pm2_5 = st.slider('PM2.5', 0.0, 500.0, 45.0, 1.0)
    pm10  = st.slider('PM10',  0.0, 600.0, 80.0, 1.0)
    co    = st.slider('CO',    0.0, 5000.0, 500.0, 10.0)
    no2   = st.slider('NO2',   0.0, 300.0, 40.0, 1.0)
    so2   = st.slider('SO2',   0.0, 300.0, 30.0, 1.0)
    o3    = st.slider('O3',    0.0, 300.0, 60.0, 1.0)
    dust  = st.slider('Dust',  0.0, 200.0, 5.0,  1.0)

    st.subheader('📊 Recent AQI History')
    st.caption('Enter readings from previous hours')
    aqi_lag1   = st.slider('AQI 1 hour ago',        0.0, 500.0, 100.0, 1.0)
    aqi_lag2   = st.slider('AQI 2 hours ago',        0.0, 500.0, 95.0,  1.0)
    aqi_lag24  = st.slider('AQI same hour yesterday',0.0, 500.0, 90.0,  1.0)
    pm25_lag1  = st.slider('PM2.5 1 hour ago',       0.0, 500.0, 42.0,  1.0)
    pm25_lag24 = st.slider('PM2.5 same hr yesterday',0.0, 500.0, 40.0,  1.0)
    o3_lag1    = st.slider('O3 1 hour ago',           0.0, 300.0, 58.0,  1.0)
    o3_lag24   = st.slider('O3 same hour yesterday',  0.0, 300.0, 55.0,  1.0)

# ── ROLLING AVERAGES (auto computed) ──
aqi_roll3  = (aqi_lag1 + aqi_lag2 + aqi_lag24) / 3
aqi_roll6  = aqi_roll3
aqi_roll24 = aqi_lag24
pm25_roll3  = (pm25_lag1 + pm25_lag24) / 2
pm25_roll24 = pm25_lag24

# ── PREDICT BUTTON ──
st.markdown('---')
predict_btn = st.button('🔮 PREDICT AQI', use_container_width=True)

if predict_btn:
    input_data = {
        'year': year, 'day': day, 'hour': hour,
        'day_of_week': day_of_week, 'is_weekend': is_weekend,
        'season': season, 'temp_c': temp_c,
        'humidity_percent': humidity, 'dew_point_c': dew_point,
        'wind_speed_kmh': wind_speed, 'wind_dir_deg': wind_dir,
        'wind_gusts_kmh': wind_gusts, 'precipitation_mm': precipitation,
        'is_raining': is_raining, 'heavy_rain': heavy_rain,
        'pressure_msl_hpa': pressure, 'cloud_cover_percent': cloud_cover,
        'pm2_5_ugm3': pm2_5, 'pm10_ugm3': pm10, 'co_ugm3': co,
        'no2_ugm3': no2, 'so2_ugm3': so2, 'o3_ugm3': o3, 'dust_ugm3': dust,
        'festival_period': festival, 'crop_burning_season': crop_burn,
        'AQI_lag1': aqi_lag1, 'AQI_lag2': aqi_lag2, 'AQI_lag24': aqi_lag24,
        'pm2_5_lag1': pm25_lag1, 'pm2_5_lag24': pm25_lag24,
        'o3_lag1': o3_lag1, 'o3_lag24': o3_lag24,
        'AQI_roll3': aqi_roll3, 'AQI_roll6': aqi_roll6, 'AQI_roll24': aqi_roll24,
        'pm2_5_roll3': pm25_roll3, 'pm2_5_roll24': pm25_roll24,
        'is_peak_hour': is_peak
    }

    input_df = pd.DataFrame([input_data])[feature_cols]

    predicted_aqi       = float(regressor.predict(input_df)[0])
    predicted_aqi       = max(0, min(500, predicted_aqi))
    predicted_class     = classifier.predict(input_df)[0]
    predicted_pollutant = le_pollutant.inverse_transform([predicted_class])[0]
    category, color, health_msg, advice = get_alert(predicted_aqi)

    # ── OUTPUT ──
    st.markdown('---')
    st.header('🎯 Estimation Results')

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div style='background-color:{color};padding:30px;border-radius:12px;text-align:center;'>
            <h1 style='color:white;margin:0;font-size:64px;'>{predicted_aqi:.0f}</h1>
            <h2 style='color:white;margin:0;'>{category}</h2>
            <p style='color:white;margin:0;font-size:14px;'>Estimated AQI</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div style='background-color:#2c3e50;padding:30px;border-radius:12px;text-align:center;'>
            <h1 style='color:white;margin:0;font-size:48px;'>🏭</h1>
            <h2 style='color:white;margin:0;'>{predicted_pollutant}</h2>
            <p style='color:white;margin:0;font-size:14px;'>Dominant Pollutant</p>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style='background-color:#e8f4f8;padding:20px;border-radius:12px;'>
            <h3 style='color:#1F4E79;margin-top:0;'>⚕️ Health Advisory</h3>
            <p style='color:#333;'><b>{health_msg}</b></p>
            <p style='color:#555;'>{advice}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── AQI SCALE ──
    st.markdown('---')
    st.subheader('📊 AQI Scale Reference')
    scale_cols = st.columns(6)
    scale_data = [
        ('Good', '0-50', '#2ecc71'),
        ('Satisfactory', '51-100', '#a8e063'),
        ('Moderate', '101-200', '#f39c12'),
        ('Poor', '201-300', '#e74c3c'),
        ('Very Poor', '301-400', '#8e44ad'),
        ('Severe', '401-500', '#2c3e50'),
    ]
    for col, (cat, rng, clr) in zip(scale_cols, scale_data):
        border = '4px solid black' if cat == category else 'none'
        col.markdown(f"""
        <div style='background:{clr};padding:10px;border-radius:8px;text-align:center;border:{border}'>
            <b style='color:white;font-size:12px;'>{cat}</b><br>
            <span style='color:white;font-size:11px;'>{rng}</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info('👆 Fill in the conditions above and click PREDICT AQI to get the estimation.')

# ── FOOTER ──
st.markdown('---')
st.markdown(
    '<p style="text-align:center;color:gray;font-size:12px;">ME228 | IIT Bombay | Mumbai AQI Estimation Project | 2026</p>',
    unsafe_allow_html=True
)
