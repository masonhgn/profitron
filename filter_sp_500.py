from collect_sp_500_data import collect_data

tickers = collect_data()

results = []


import yfinance as yf

def get_last_closing_price(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period="1d")
        last_closing_price = history['Close'].iloc[-1]
        return last_closing_price
    except:
        return None

for ticker in tickers:
    price = get_last_closing_price(ticker)
    if price is not None and price <= 50:
        print(ticker, price)



