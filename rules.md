Respect the following guidelines when writing your code:
For loading data loader = dt.DataLoader('REE', '2021-02-01','2021-04-02', data_source='cafe', minimal=True)
data = loader.download()
st.write(data)
For visualization:
from vnquant import plot as pl
pl.vnquant_candle_stick(
    data='VND',
    title='VND symbol from 2019-09-01 to 2019-11-01',
    xlab='Date', ylab='Price',
    start_date='2019-09-01',
    end_date='2019-11-01',
    data_source='CAFE',
    show_advanced=['volume', 'macd', 'rsi']
)
For loading multiple stickers use:
loader = dt.DataLoader(['VCB', 'TCB'], '2021-02-01','2021-04-02', data_source='CAFE', minimal=True, table_style='prefix')
data = loader.download()

Any Calculation involve stock performance or portfolio performance should use adjusted price. 

@2025-03-01T14-04_export.csv look at to understand data structure and column names


