import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pmdarima as pm
import h2o
from tpot import TPOTRegressor
from flaml import AutoML
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import copy
import plotly.express as px

def auto_arima_predictions(df, test_percentage, num_predictions):
    df = df.copy()
    df = df["Adj Close"]
    test_size = int(test_percentage / 100 * len(df))
    train_data = df[:-test_size]
    test_data = df[-test_size:]
    model = pm.auto_arima(train_data, error_action='warn', trace=True, suppress_warnings=True, stepwise=True,
                          random_state=42, n_fits=100)
    model.fit(train_data)
    forecast = model.predict(n_periods=len(test_data))
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    model.fit(df)
    predictions = model.predict(n_periods=num_predictions)
    predictions = pd.DataFrame(predictions, columns=['Prediction'])
    best_model_name = model.order
    return rmse, predictions, best_model_name

def make_lag_df(df, lag):
    df_copy = df.copy()
    for i in range(1, lag + 1):
        df_copy[f'Adj Close_lag_{i}'] = df_copy['Adj Close'].shift(i)
    return df_copy.dropna()

def h2o_predictions(df, lag, test_percentage, num_predictions, seconds_model):
    df = df.copy()
    df = df[["Adj Close"]]
    df = make_lag_df(df, lag)
    train_size = int(len(df) * (1 - test_percentage / 100))
    train, test = df[:train_size], df[train_size:]
    h2o.init()
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)
    x = train.columns.drop('Adj Close').tolist()
    y = "Adj Close"
    aml = H2OAutoML(max_runtime_secs=seconds_model, seed=42)
    aml.train(x=x, y=y, training_frame=train_h2o)
    best_model_name = aml.leaderboard[0, "model_id"]
    perf = aml.leader.model_performance(test_data=test_h2o)
    rmse = perf.rmse()
    predictions = []
    h2o_predictions_df = df.copy()
    for _ in range(num_predictions):
        last_entry = h2o_predictions_df.iloc[-1]
        h2o_predictions_df = pd.DataFrame([last_entry], columns=df.columns)
        h2o_predictions_df = h2o_predictions_df.shift(1, axis=1)
        h2o_predictions_df.iloc[:, 0] = 0
        h2o_predictions_df.drop("Adj Close", axis=1, inplace=True)
        prediction = aml.predict(h2o.H2OFrame(h2o_predictions_df)).as_data_frame().iloc[0, 0]
        predictions.append(prediction)
        h2o_predictions_df.insert(0, 'Adj Close', prediction)
    h2o.cluster().shutdown()
    predictions = pd.DataFrame(predictions, columns=['Prediction'])
    return rmse, predictions, best_model_name

def tpot_predictions(df, lag, test_percentage, num_predictions, seconds_model):
    df = df.copy()
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].astype('int64')
    df = df[['Adj Close']]
    df = make_lag_df(df, lag)
    X = df.drop(columns=['Adj Close'])
    y = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage/100, random_state=42)
    tpot = TPOTRegressor(max_time_mins=seconds_model/60, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    predictions_test = tpot.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    last_row = X.iloc[[-1]].copy()
    predictions = []
    for i in range(num_predictions):
        next_day_prediction = tpot.predict(last_row)[0]
        predictions.append(next_day_prediction)
        last_row.iloc[:, :-1] = last_row.iloc[:, 1:].values
        last_row.iloc[:, -1] = next_day_prediction
    predictions = pd.DataFrame(predictions, columns=['Prediction'])
    best_model_name = tpot.fitted_pipeline_.steps[-1][1].__class__.__name__
    return rmse, predictions, best_model_name

def flaml_predictions(df, lag, test_percentage, num_predictions, seconds_model):
    df = df.copy()
    df = df[['Adj Close']]
    df = make_lag_df(df, lag)
    X = df.drop(columns=['Adj Close']).values
    y = df['Adj Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage/100, random_state=42)
    automl = AutoML()
    settings = {
        "time_budget": seconds_model,
        "metric": 'rmse',
        "task": 'regression'
    }
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    predictions_test = automl.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    predictions = []
    flaml_predictions_df = df.copy()
    for _ in range(num_predictions):
        last_entry = flaml_predictions_df.iloc[-1]
        flaml_predictions_df = pd.DataFrame([last_entry], columns=df.columns)
        flaml_predictions_df = flaml_predictions_df.shift(1, axis=1)
        flaml_predictions_df.iloc[:, 0] = 0
        flaml_predictions_df.drop("Adj Close", axis=1, inplace=True)
        prediction = automl.predict(flaml_predictions_df.values.reshape(1, -1))[0]
        predictions.append(prediction)
        flaml_predictions_df.insert(0, 'Adj Close', prediction)
    predictions = pd.DataFrame(predictions, columns=['Prediction'])
    best_model_name = automl.model.estimator
    return rmse, predictions, best_model_name

def next_weekdays(start_date, n):
    weekdays = []
    current_date = start_date + timedelta(days=1)
    while len(weekdays) < n:
        if current_date.weekday() < 5:
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    return weekdays

def predictia_preturilor_actiunilor(companii):
    st.subheader("Predicția prețurilor acțiunilor")
    companie_selectata = st.selectbox("Selectați o companie:", companii["Ticker"])

    option = st.checkbox("OPȚIONAL: Selectați perioada pe care doriți să o analizați (DEFAULT: toate datele disponibile despre companie)⌛")
    if companie_selectata and option:
        start_date_stock = st.date_input("Selectați data de start🕘:", datetime.today())
        end_date_stock = st.date_input("Selectați data de sfârșit🕔:", datetime.today())

        # validare date
        stock_data = yf.Ticker(companie_selectata).history(period="max")
        earliest_date = stock_data.index.min().date()
        if start_date_stock < earliest_date:
            st.error("Data de start trebuie să fie ulterioară datei {}".format(earliest_date))

        if start_date_stock > end_date_stock:
            st.error("Data de start trebuie să fie anterioară datei de sfârșit")

        today = datetime.now().date()
        if end_date_stock >= today:
            st.error("Data de sfârșit nu trebuie să fie ulterioară zilei de azi")

        diferenta_zile = (end_date - start_date).days
        if diferenta_zile < 7:
            st.error("Datele nu sunt la cel puțin o săptămână distanță")

        if start_date_stock >= earliest_date and start_date_stock <= end_date_stock and end_date_stock < today and diferenta_zile >= 7:
            df = yf.download(companie_selectata, start=start_date_stock, end=end_date_stock + timedelta(1))

    elif companie_selectata:
        df = yf.download(companie_selectata)

    num_predictions = st.slider("Selectați numărul de zile pentru care doriți să primiți predicții🔮:", min_value=1, max_value=30, step=1, value=1)
    test_percentage = st.slider("Selectați procentul de date care să fie folosite pentru testare💯:", min_value=1, max_value=50, step=1, value=20)
    lag = st.slider("Selectați lagul🍭:", min_value=1, max_value=10, step=1, value=5)
    seconds = st.slider("Selectați timpul acordat antrenării modelelor de AutoML⏰:", min_value=100, max_value=10000, step=100, value=300)
    seconds_model = int(seconds / 3)

    if st.button("Obțineți predicții!💸") and companie_selectata:

        st.header(f"Graficul acțiunilor {companie_selectata}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Prețuri reale', line=dict(color='blue')))
        fig.update_layout(title='Prețurile acțiunilor',
                          xaxis_title='Dată',
                          yaxis_title='Preț de închidere ajustat ($)')
        st.plotly_chart(fig)

        with st.status("Realizare predicții...", expanded=True) as status:

            st.header("AutoARIMA")
            auto_arima_rmse, auto_arima_predictions_df, auto_arima_best_model_name = auto_arima_predictions(df, test_percentage, num_predictions)
            if option:
                dates = next_weekdays(end_date, num_predictions)
            else:
                yesterday = datetime.now().date() - timedelta(1)
                dates = next_weekdays(yesterday, num_predictions)
            auto_arima_predictions_df['Date'] = dates
            auto_arima_predictions_df.set_index('Date', inplace=True)
            st.write(f"RMSE: {auto_arima_rmse}")
            st.write("Predicții")
            st.write(auto_arima_predictions_df)
            st.write(f"Cel mai bun model: {auto_arima_best_model_name}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Prețuri reale', line=dict(color='blue')))
            fig.add_trace(
                go.Scatter(x=auto_arima_predictions_df.index, y=auto_arima_predictions_df['Prediction'], mode='lines',
                           name='Prețuri previzionate', line=dict(color='green')))
            fig.update_layout(title='Prețurile acțiunilor',
                              xaxis_title='Dată',
                              yaxis_title='Preț de închidere ajustat ($)')
            st.plotly_chart(fig)

            st.header("H2O")
            h2o_rmse, h2o_predictions_df, h2o_best_model_name = h2o_predictions(df, lag, test_percentage, num_predictions, seconds_model)
            if option:
                dates = next_weekdays(end_date, num_predictions)
            else:
                yesterday = datetime.now().date() - timedelta(1)
                dates = next_weekdays(yesterday, num_predictions)
            dates = pd.to_datetime(dates)
            h2o_predictions_df.index = dates
            st.write(f"RMSE: {h2o_rmse}")
            st.write("Predicții")
            st.write(h2o_predictions_df)
            st.write(f"Cel mai bun model: {h2o_best_model_name}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Prețuri reale', line=dict(color='blue')))
            fig.add_trace(
                go.Scatter(x=h2o_predictions_df.index, y=h2o_predictions_df['Prediction'], mode='lines',
                           name='Prețuri previzionate', line=dict(color='green')))
            fig.update_layout(title='Prețurile acțiunilor',
                              xaxis_title='Dată',
                              yaxis_title='Preț de închidere ajustat ($)')
            st.plotly_chart(fig)

            st.header("TPOT")
            tpot_rmse, tpot_predictions_df, tpot_best_model_name = tpot_predictions(df, lag, test_percentage, num_predictions, seconds_model)
            if option:
                dates = next_weekdays(end_date, num_predictions)
            else:
                yesterday = datetime.now().date() - timedelta(1)
                dates = next_weekdays(yesterday, num_predictions)
            tpot_predictions_df['Date'] = dates
            tpot_predictions_df.set_index('Date', inplace=True)

            st.write(f"RMSE: {tpot_rmse}")
            st.write("Predicții")
            st.write(tpot_predictions_df)
            st.write(f"Cel mai bun model: {tpot_best_model_name}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Prețuri reale', line=dict(color='blue')))
            fig.add_trace(
                go.Scatter(x=tpot_predictions_df.index, y=tpot_predictions_df['Prediction'], mode='lines',
                           name='Prețuri previzionate', line=dict(color='green')))
            fig.update_layout(title='Prețurile acțiunilor',
                              xaxis_title='Dată',
                              yaxis_title='Preț de închidere ajustat ($)')
            st.plotly_chart(fig)

            st.header("FLAML")
            flaml_rmse, flaml_predictions_df, flaml_best_model_name = flaml_predictions(df, lag, test_percentage, num_predictions, seconds_model)
            if option:
                dates = next_weekdays(end_date, num_predictions)
            else:
                yesterday = datetime.now().date() - timedelta(1)
                dates = next_weekdays(yesterday, num_predictions)
            flaml_predictions_df['Date'] = dates
            flaml_predictions_df.set_index('Date', inplace=True)

            st.write(f"RMSE: {flaml_rmse}")
            st.write("Predicții")
            st.write(flaml_predictions_df)
            st.write(f"Cel mai bun model: {flaml_best_model_name}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Prețuri reale', line=dict(color='blue')))
            fig.add_trace(
                go.Scatter(x=flaml_predictions_df.index, y=flaml_predictions_df['Prediction'], mode='lines',
                           name='Prețuri previzionate', line=dict(color='green')))
            fig.update_layout(title='Prețurile acțiunilor',
                              xaxis_title='Dată',
                              yaxis_title='Preț de închidere ajustat ($)')
            st.plotly_chart(fig)

            status.update(label="Predicții realizate", state="complete", expanded=False)

        data = {
            'Nume algoritm': ['AutoARIMA', 'H2O', 'TPOT', 'FLAML'],
            'RMSE': [auto_arima_rmse, h2o_rmse, tpot_rmse, flaml_rmse],
            'Cel mai bun model': [str(auto_arima_best_model_name), h2o_best_model_name, tpot_best_model_name, flaml_best_model_name]
        }
        df_algoritmi = pd.DataFrame(data)
        df_algoritmi = df_algoritmi.sort_values(by='RMSE')
        df_algoritmi.index += 1
        st.header("Leaderboard")
        st.write(df_algoritmi)

def plot_efficient_frontier_and_max_sharpe(mu, S, selected_risk_free_rate):
    ef = EfficientFrontier(mu, S)
    ef_max_sharpe = copy.deepcopy(ef)
    ef_max_sharpe.max_sharpe(risk_free_rate=selected_risk_free_rate)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    st.header("Frontiera eficientă a portofoliilor")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stds, y=rets, mode='markers', marker=dict(color=sharpes, colorscale='viridis_r', size=5),
                             name='Portofolii aleatoare'))
    fig.add_trace(go.Scatter(x=[std_tangent], y=[ret_tangent], mode='markers', marker=dict(color='red', symbol='star', size=10),
                             name='Max Sharpe'))
    fig.update_layout(title='Frontiera eficientă a portofoliilor',
                      xaxis_title='Deviație standard',
                      yaxis_title='Randament așteptat',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig)

def problema_portofoliului_optim(companii):
    st.subheader("Problema portofoliului optim")
    tickers = st.multiselect('Selectați acțiunile pe care le doriți în portofoliul dvs📊:', companii["Ticker"])
    start_date_po = st.date_input("Selectați data de start:", datetime.today())
    end_date_po = st.date_input("Selectați data de sfârșit:", datetime.today())
    if start_date_po > end_date_po:
        st.error("Data de start trebuie să fie anterioară datei de sfârșit")
    selected_risk_free_rate = st.slider("Selectați rata fără risc (risk free rate)🧊:", min_value=0.01, max_value=0.05, value=0.02,
                                        step=0.01)
    if st.button("Analizați!🤓"):
        stocks_df = yf.download(tickers, start=start_date_po, end=end_date_po)['Adj Close']
        st.write(stocks_df.head())
        fig_price = px.line(stocks_df, title='Prețurile acțiunilor')
        st.plotly_chart(fig_price)
        daily_returns = stocks_df.pct_change().dropna()
        st.header("Rentabilitățile zilnice")
        st.write(daily_returns.head())
        corr_df = stocks_df.corr().round(2)
        st.header("Corelația dintre acțiuni")
        fig_corr = px.imshow(corr_df, text_auto=True, title='Corelația dintre acțiuni')
        st.plotly_chart(fig_corr)

        mu = expected_returns.mean_historical_return(stocks_df)
        S = risk_models.sample_cov(stocks_df)
        st.header("Randamentul așteptat al activelor din portofoliul dvs")
        st.write(mu)
        plot_efficient_frontier_and_max_sharpe(mu, S, selected_risk_free_rate)

        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=0.02)
        weights = ef.clean_weights()
        st.header("Ponderi")
        weights_df = pd.DataFrame.from_dict(weights, orient='index')
        weights_df.columns = ['weights']
        st.write(weights_df)

        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        st.header("Performanța portofoliului")
        st.write('Randament anual așteptat (expected annual return): {}%'.format((expected_annual_return * 100).round(2)))
        st.write('Volatilitate anuală (annual volatility): {}%'.format((annual_volatility * 100).round(2)))
        st.write('Rata Sharpe (Sharpe ratio): {}'.format(sharpe_ratio.round(2)))

        st.header("Portofoliu optimizat")
        stocks_df['Optimized Portfolio'] = 0
        for ticker, weight in weights.items():
            stocks_df['Optimized Portfolio'] += stocks_df[ticker] * weight
        st.write(stocks_df.head())

def main():
    st.set_page_config(page_title="Aplicație", page_icon=":moneybag:")

    # Sidebar
    st.sidebar.title("Predicția prețurilor acțiunilor și problema portofoliului optim")
    st.sidebar.image('stock-market.jpg')
    st.sidebar.subheader("Sectoare principale de activitate și companiile de vârf")
    companii = pd.read_csv("companii.tsv", sep="\t")
    companii.index = range(1, len(companii) + 1)
    st.sidebar.table(companii)

    st.sidebar.subheader("Descrierea bibliotecilor Python folosite pentru predicții")
    biblioteci = {
    'Biblioteci': ['AutoARIMA', 'H2O', 'TPOT', 'FLAML'],
    'Descriere': [
        'Implementează modele automatice ARIMA pentru prognoza seriilor de timp.',
        'Platformă de machine learning scalabilă, open-source, cu algoritmi distribuiți.',
        'Librărie Python de AutoML, bazată pe optimizare evolutivă.',
        'Bibliotecă de machine learning, furnizând soluții eficiente pentru problemele de regresie și clasificare.'
    ]
    }
    biblioteci = pd.DataFrame(biblioteci)
    biblioteci.index = range(1, len(biblioteci) + 1)
    st.sidebar.table(biblioteci)

    # Main screen
    tab1, tab2 = st.tabs(["Predicția prețurilor acțiunilor📈", "Problema portofoliului optim🗂️"])
    with tab1:
      predictia_preturilor_actiunilor(companii)
    with tab2:
      problema_portofoliului_optim(companii)

if __name__ == '__main__':
    main()
