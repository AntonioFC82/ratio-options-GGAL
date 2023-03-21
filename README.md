# ratio-options-GGAL
### Backtesting ratio option strategy for GGAL-AR stock.

This code allows for backtesting on an options exercise considering the 'ratio spread' strategy was employed.
The information for each OpEx is in an Excel file, which is processed to return a DataFrame for the rest of the operations.
The file 'data_by_opex.py' contains all the code. From this, .py files are generated to backtest each OpEx. In each file, only two values need to be set:

1.  n = number of days elapsed since the beginning of the OpEx (the program assumes that the strategy was built throughout those n days).
2.  qty_atm = number of bought lots (defaulted to 100).

The backtesting report looks as follows:
![image](https://user-images.githubusercontent.com/87617614/226752551-3a535e07-a591-4a75-a990-57fc194add0d.png)

The main graph shows the daily PnL (expressed in 100 k$):
![image](https://user-images.githubusercontent.com/87617614/226752838-4c1ee06b-0c8b-43a3-88bf-26e40e647909.png)
