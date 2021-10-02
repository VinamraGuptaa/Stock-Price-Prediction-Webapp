#Stock Price Prediction webapp using Facebook Prophet

#Imports
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet 
import streamlit as st
from datetime import date
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Prediction App🚀')
stocks = ('TCS', 'RIL', 'ICICI Bank', 'Bajaj finserv')
selected_stock = st.selectbox('Select dataset for prediction ', stocks)

n_years = st.slider('Years of prediction:📆', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    if(ticker=="TCS"):
        data = yf.download("TCS.NS",START,TODAY)
        data.reset_index(inplace=True)
        return data

    elif(ticker=="RIL"):
        data = yf.download("RELIANCE.NS", START, TODAY)
        data.reset_index(inplace=True)
        return data
    elif(ticker=="ICICI Bank"):
        data = yf.download("ICICIBANK.NS", START, TODAY)
        data.reset_index(inplace=True)
        return data   
    elif(ticker=="Bajaj finserv"):
        data = yf.download("BAJFINANCE.NS", START, TODAY)
        data.reset_index(inplace=True)
        return data 


             
    

	
data_load_state = st.text('Loading data...⏳')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!✔️')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider📊', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data📏')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years 📅 ')
fig1 = plot_plotly(m, forecast,xlabel="Date",ylabel="Close")
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.markdown("***")
st.subheader("Let's discuss and analyse the drop in trend in March 2020 due to Lockdown☣️")
option = st.selectbox('Select the stock you want to analyse',('TCS', 'RIL', 'ICICI Bank','Bajaj finserv'))
get_data=load_data(option)
if(option=="TCS"):
    
    #Setting the range of base plot
    fig = px.line(get_data, x='Date', y='High',title="TCS: Day's High Price during Phase 1 Lockdown(RED)(25 March – 14 April) and Phase 2 Lockdown (GREEN)(15 April – 3 May💯)", range_x=['2020-01-01','2020-06-30'])

    # Adding the shape in the dates
    fig.update_layout(
        shapes=[
            # First phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-03-23",y0=0,x1="2020-04-14",y1=1,fillcolor="Red",opacity=0.5,layer="below",line_width=0,),
            # Second phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-04-15",y0=0,x1="2020-05-03",y1=1,fillcolor="Green",opacity=0.5,layer="below",line_width=0,)
                ])
    st.plotly_chart(fig)
elif(option=="RIL"):
   
    #Setting the range of base plot
    fig = px.line(get_data, x='Date', y='High',title="RIL: Day's High Price during Phase 1 Lockdown(RED)(25 March – 14 April) and Phase 2 Lockdown (GREEN)(15 April – 3 May💯)", range_x=['2020-01-01','2020-06-30'])

    # Adding the shape in the dates
    fig.update_layout(
        shapes=[
            # First phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-03-23",y0=0,x1="2020-04-14",y1=1,fillcolor="Red",opacity=0.5,layer="below",line_width=0,),
            # Second phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-04-15",y0=0,x1="2020-05-03",y1=1,fillcolor="Green",opacity=0.5,layer="below",line_width=0,)
                ])
    st.plotly_chart(fig)
elif(option=="ICICI Bank"):
   
    #Setting the range of base plot
    fig = px.line(get_data, x='Date', y='High',title="ICICI Bank: Day's High Price during Phase 1 Lockdown(RED)(25 March – 14 April) and Phase 2 Lockdown (GREEN)(15 April – 3 May💯)", range_x=['2020-01-01','2020-06-30'])

    # Adding the shape in the dates
    fig.update_layout(
        shapes=[
            # First phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-03-23",y0=0,x1="2020-04-14",y1=1,fillcolor="Red",opacity=0.5,layer="below",line_width=0,),
            # Second phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-04-15",y0=0,x1="2020-05-03",y1=1,fillcolor="Green",opacity=0.5,layer="below",line_width=0,)
                ])
    st.plotly_chart(fig)
elif(option=="Bajaj finserv"):
    #Setting the range of base plot
    fig = px.line(get_data, x='Date', y='High',title="Bajaj Finserv: Day's High Price during Phase 1 Lockdown(RED)(25 March – 14 April) and Phase 2 Lockdown (GREEN)(15 April – 3 May💯)", range_x=['2020-01-01','2020-06-30'])

    # Adding the shape in the dates
    fig.update_layout(
        shapes=[
            # First phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-03-23",y0=0,x1="2020-04-14",y1=1,fillcolor="Red",opacity=0.5,layer="below",line_width=0,),
            # Second phase Lockdown
            dict(type="rect",xref="x",yref="paper",x0="2020-04-15",y0=0,x1="2020-05-03",y1=1,fillcolor="Green",opacity=0.5,layer="below",line_width=0,)
                ])
    st.plotly_chart(fig) 

st.markdown("***")
#Footer
st.write("Made with ❤️by Vinamra Gupta")



col1, col2,col3 = st.columns(3)

with col1:
    st.markdown('''
    <a href="https://github.com/VinamraGuptaa">
        <img src="https://cdn.jsdelivr.net/gh/dmhendricks/signature-social-icons/icons/round-flat-filled/50px/github.png" />
    </a>''',
    unsafe_allow_html=True
)
with col2:
    st.markdown('''
    <a href="https://www.linkedin.com/in/vinamra-gupta-6908001a8/">
        <img src="https://cdn.jsdelivr.net/gh/dmhendricks/signature-social-icons/icons/round-flat-filled/50px/linkedin.png" />
    </a>''',
    unsafe_allow_html=True
)
with col3:
    st.markdown('''
    <a href="https://medium.com/@vinamragupta98">
        <img src="https://cdn1.iconfinder.com/data/icons/social-media-circle-7/512/Circled_Medium_svg5-48.png" />
    </a>''',
    unsafe_allow_html=True
)
    
  
    


