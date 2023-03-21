'''=======================================================================
I. IMPORTS
======================================================================='''

from data_by_opex import *

#-------------------------------------------------------------------------
df_suby = get_data_suby()

book = '$GGAL - Info Opex 2021-08.xlsx'
sheet = 'Lotes-Agosto 2021'
df_opt = get_data_opt(book, sheet)

tipo = 'call'
df_call = call_put(df_opt, tipo)

df_ATM = base_ATM(df_call, df_suby)

n = 6
info_bases = get_bases(df_ATM, df_call, n)

b_atm = get_bases(df_ATM, df_call, n)['Base_ATM']
b_otm = get_bases(df_ATM, df_call, n)['Base_OTM']
px_atm = get_bases(df_ATM, df_call, n)['Px_ATM']
px_otm = get_bases(df_ATM, df_call, n)['Px_OTM']
qty_atm = 100
info_ratio = ratio(df_call, b_atm, b_otm, px_atm, px_otm, n, qty_atm)
break_even = ratio(df_call, b_atm, b_otm, px_atm, px_otm, n)[0]['break_even']

info_PnL_day = PnL_day(df_ATM, df_call, b_atm, b_otm, n)


opex_report = html_report(df_call, df_ATM, b_atm, b_otm, px_atm, px_otm, n, break_even, 'AG21')

#-------------------------------------------------------------------------
