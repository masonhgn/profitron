import pandas as pd
from ftplib import FTP
from io import StringIO

# Connect to the NASDAQ FTP server
ftp = FTP('ftp.nasdaqtrader.com')
ftp.login()

# Navigate to the SymbolDirectory
ftp.cwd('SymbolDirectory')

# Initialize a list to store the lines of the file
lines = []

# Retrieve the 'otherlisted.txt' file and append each line to the list
ftp.retrlines('RETR otherlisted.txt', lines.append)

# Close the FTP connection
ftp.quit()

# Join the list into a single string separated by newlines
data = '\n'.join(lines)

# Remove the last two lines (footer information)
data = '\n'.join(data.split('\n')[:-2])

# Read the data into a pandas DataFrame using the pipe '|' as a separator
df = pd.read_csv(StringIO(data), sep='|')

# Filter the DataFrame for NYSE listings ('N' denotes NYSE)
nyse_df = df[df['Exchange'] == 'N']

# Get the list of tickers
nyse_tickers = nyse_df['ACT Symbol'].tolist()

# Print the list of NYSE tickers
print(nyse_tickers)
print(len(list(set(nyse_tickers))))
nyse_df.to_csv('nyse_tickers.csv', index=False)