'''=======================================================================
I. IMPORTS
======================================================================='''
from math import floor

import pandas as pd
import numpy as np
import ta
#import pyxlsb
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

pio.renderers.default = "browser"


'''=======================================================================
II. GET & ORDER DATA
======================================================================='''
#1. Histórico Cotizaciones Subyacente.
def get_data_suby(start='2020-12-01'):
    ''' Extrae data del Excel con históricos
    del Subyacente y devuelve un DF ordenado.'''
    df_hist = pd.read_csv('cotizaciones_historicas.csv')
    df_hist = df_hist.drop(['especie', 'timestamp'], axis=1)
    df_hist.set_index('fecha', inplace=True)
    df_hist.index.name = 'Date'
    df_hist.index = pd.to_datetime(df_hist.index)
    df_hist.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df_hist.loc[start:,:].copy().round(2)
    df['daily_return'] = ta.others.daily_return(df['Close'])
    df['volat_40'] = df['daily_return'].rolling(40).std()*np.sqrt(250)
    df['volat_10'] = df['daily_return'].rolling(10).std()*np.sqrt(250)
    
    return df


#2. Cotizaciones Opciones por OpEx.
def get_data_opt(book, sheet):
    ''' Extrae data del Excel con históricos
    de Opciones y devuelve un DF ordenado.'''
    df = pd.read_excel(book, sheet)
    
    # Quitar columnas que no se utilizarán:
    df = df.drop(['ESPECIE', 'MONTO', 'HORA', 'C. ANT.', 'NOMINAL',
        'TLR', 'DÍAS AL VTO.', 'PLAZO (años)'],
        axis=1).dropna()
    
    # Renombrar columnas:
    old_names = df.columns
    new_names = ['Fecha', 'Base', 'Tipo', 'Close', 'pct_opt', 'Open',
        'High', 'Low', 'Px_GGAL', 'pct_suby', 'VI', 'VE', 'Paridad']
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    
    # Cambiar a % las columnas pct y volatilidades:
    df['pct_opt'] = (df['pct_opt']*100).round(2)
    df['pct_suby'] = (df['pct_suby']*100).round(2)
    df['VI'] = (df['VI']*100).round(2)
    df['VE'] = (df['VE']*100).round(2)

    return df


#3. Separar DF según tipo de Opción.
def call_put(df, tipo):
    '''Devuelve un DF sólo de Calls o Put
    y agrega columna con el ratio entre 2 bases
    (con 1 base intermedia libre).
    df: return get_data_opt(),
    tipo: str, 'call' o 'put'.'''
    if tipo == 'call':
        df_call = df.loc[df['Tipo'] == 'Call']
        # df_call = df[df['Tipo'].isin(['Call'])]
        df_call['ratio_2B'] = (df_call['Close']/df_call['Close'].shift(-2)).round(2)
        return df_call
    if tipo == 'put':
        df_put = df.loc[df['Tipo'] == 'Put']
        df_put['ratio_2B'] = (df_put['Close']/df_put['Close'].shift(-2)).round(2)
        return df_put
    

#4. Obtener un DF con la base ATM de cada día del OpEx.
def base_ATM(df1, df2):
    '''Devuelve un DF con la base ATM del día y agrega 2 columnas
    con las volatilidades de 10 y 40 días del subyacente.
    df1: return de call_put(),
    df2: return de get_data_suby().'''
    fecha_uni = set(df1['Fecha'])

    df_atm = pd.DataFrame()
    for fecha in fecha_uni:
        df = df1.groupby(['Fecha']).get_group(fecha).nsmallest(1, 'Paridad') # Paridad --> diferencia entre Px_GGAL y Base.
        df_atm = df_atm.append(df)
    
    df_atm.set_index('Fecha', inplace=True)
    df_atm = df_atm.sort_values('Fecha', ascending=True)

    # Combinando el DF con la VH de 40d y 10d del Subyacente:
    df_atm = pd.merge(df_atm, df2[['volat_40', 'volat_10']], right_index=True, left_index=True)
    
    return df_atm


'''=======================================================================
III. GET RATIO
======================================================================='''
def get_bases(df1, df2, n):
    '''Retorna el par de bases ATM y OTM con sus precios, en función
    del Px_Suby medio de los primeros n días del OpEx.
    df1: return base_ATM(),
    df2: return call_put(),
    n: int, días desde iniciado el OpEx.'''
    indice = list(set(df2['Fecha']))
    indice.sort()
    # Cálculo de las Bases ATM & OTM
    #px_medio = df1['Px_GGAL'].head(n).mean()
    #b_atm = df1.head(n).loc[(df1['Base'] > px_medio)]['Base'].values[0]
    b_atm = df1[n-1:]['Base'].values[0]

    bases_full = list(set(df2['Base']))
    bases_full.sort()
    indexATM = bases_full.index(b_atm)
    b_otm = bases_full[indexATM + 2]

    px_atm = df2[(df2['Fecha'] == indice[n-1]) & (df2['Base'] == b_atm)]['Close'].values[0]
    px_otm = df2[(df2['Fecha'] == indice[n-1]) & (df2['Base'] == b_otm)]['Close'].values[0]

    fecha_ratio = df2[(df2['Fecha'] == indice[n-1])]['Fecha'].values[0]

    bases = {'Base_ATM': b_atm,
        'Px_ATM': px_atm,
        'Base_OTM': b_otm,
        'Px_OTM': px_otm,
        'Fecha_Ratio': fecha_ratio}

    return bases


def ratio(df, b_atm, b_otm, px_atm, px_otm, n, qty_atm=100):
    '''Retorna un DF con resultados PnL a Finish del OpEx
    correspondiente a la estrategia de opciones: 'ratio'.
    df: return call_put(),
    b_atm: int, return get_bases()['Base_ATM'],
    b_otm: int, return get_bases()['Base_OTM'],
    px_atm: float, return get_bases()['Px_ATM'],
    px_otm: float, return get_bases()['Px_OTM'],
    n: int, días desde iniciado el OpEx,
    qty_atm: int, cantidad de lotes comprados.'''
    index = list(set(df['Fecha']))
    index.sort()
    # Precios de las bases ATM y OTM para armar Ratio en una fecha específica:
    px_hoy = df[(df['Fecha'] == index[n-1])]['Px_GGAL'].values[0]
    # px_atm = df[(df['Fecha'] == index[n-1]) & (df['Base'] == b_atm)]['Close'].values[0]
    # px_otm = df[(df['Fecha'] == index[n-1]) & (df['Base'] == b_otm)]['Close'].values[0]
    
    # Armar DF con valores por debajo y por encima del Px_Suby:
    paso = 3
    lista_1 = [i for i in range(floor(0.9*(floor(px_hoy))), floor(px_hoy), paso)]
    lista_2 = [i for i in range(floor(px_hoy), floor(1.25*(floor(px_hoy))), paso)]
    df_finish = pd.DataFrame([lista_1 + lista_2]).T
    df_finish.columns = ['Px_Finish']

    dif_P = 2*px_otm - px_atm # diferencia de primas
    dif_B = b_otm - b_atm # diferencia de bases
    
    # Diccionario con ganancia debajo de la b_atm, ganancia máx y breakeven:
    PnL = {
        'under_atm':round(dif_P*100*qty_atm, 2),
        'max_profit':round((dif_P + dif_B)*100*qty_atm, 2),
        'break_even':round(dif_P + 2*b_otm - b_atm, 2)
        }
    
    # Obtener DF con el PnL en función del Px_Suby a Finish:
    conditions = [
        (df_finish['Px_Finish'] < b_atm),
        (df_finish['Px_Finish'] >= b_atm) & (df_finish['Px_Finish'] < b_otm),
        (df_finish['Px_Finish'] == b_otm),
        (df_finish['Px_Finish'] > b_otm),
    ]
    values = [
        dif_P*qty_atm*100,
        (df_finish['Px_Finish'] + dif_P - b_atm)*100*qty_atm,
        (dif_P + dif_B)*100*qty_atm,
        (-df_finish['Px_Finish'] + dif_P + 2*b_otm - b_atm)*100*qty_atm
    ]
    df_finish['Result'] = np.select(conditions, values)
    
    return [PnL, df_finish.set_index('Px_Finish').Result]


def PnL_day(df1, df2, b_atm, b_otm, n):
    '''Devuelve un DF con el PnL de cada día
    si se hubiera desarmado el Ratio.
    df1: return base_ATM(),
    df2: return call_put(),
    b_atm: int, return get_bases()['Base_ATM'],
    b_otm: int, return get_bases()['Base_OTM'],
    n: int, días transcurridos desde el inicio del OpEx.'''
    list_px_atm = []
    list_px_otm = []
    indice = list(set(df2['Fecha']))
    indice.sort()

    for i in range(1, len(indice)+1):
        px_atm = df2[(df2['Fecha'] == indice[i-1]) & (df2['Base'] == b_atm)]['Close'].values[0]
        px_otm = df2[(df2['Fecha'] == indice[i-1]) & (df2['Base'] == b_otm)]['Close'].values[0]

        list_px_atm.append(px_atm)
        list_px_otm.append(px_otm)
    
    PnL = pd.DataFrame([list_px_atm, list_px_otm]).T
    PnL.set_index(df1.index, inplace=True)
    PnL.columns = ['Px_ATM', 'Px_OTM']

    credit_ratio = round(2*list_px_otm[n-1] - list_px_atm[n-1], 2)
    PnL['Day_result'] = round(credit_ratio + (PnL['Px_ATM'] - 2*PnL['Px_OTM']), 2)
    PnL = PnL[n:]

    return credit_ratio, PnL


def vi_smile(df, n):
    '''Devuelve un DF con las VI's de cada base
    para un día determinado del OpEx.
    df: return call_put(),
    n: int, días transcurridos desde el inicio del OpEx.'''
    index = list(set(df['Fecha']))
    index.sort()
    df_aux = df.set_index('Base')

    return df_aux.loc[df_aux['Fecha'] == index[n-1], ['VI']]


'''=======================================================================
IV. OTHERS
======================================================================='''
# ===== Tables HTML Report =====:
def info_tables(df1, df2, table, n):
    '''
    Retorna 2 columnas con información para armar
    las siguientes tablas:
    * table1 = precios del subyacente.
    * table2 = VH del subyacente.
    * table3 = VI de la base ATM.
    * table4 = bases y precios del Raatio.
    ----------------------------------------------
    df1: return base_ATM(),
    df2: return call_put(),
    table: str, 'tableN' (con N de 1 a 4),
    n: int, días desde el inicio del OpEx.
    ----------------------------------------------
    '''
    if table == 'table1':
        px_medio_8d = round(df1['Px_GGAL'].head(n).mean(), 2)
        px_min_opex = df1['Px_GGAL'].min()
        px_max_opex = df1['Px_GGAL'].max()
        dif_px = round((px_max_opex - px_min_opex), 2)
        values_1 = [['Ítem'], ['Value']]
        values_2 = [['Px Medio - 8 días', 'Px Mínimo', 'Px Máximo', 'Delta Px'],
                    [px_medio_8d, px_min_opex, px_max_opex, dif_px]]
        return values_1, values_2
    
    if table == 'table2':
        VH_medio_8d = df1['volat_40'].head(n).mean().round(2)
        VH_min_opex = df1['volat_40'].min().round(2)
        VH_max_opex = df1['volat_40'].max().round(2)
        dif_VH = round((VH_max_opex - VH_min_opex), 2)
        values_1 = [['Ítem'], ['Value']]
        values_2 = [['VH Media - 8 días', 'VH Mínima', 'VH Máxima', 'Delta VH'],
                    [VH_medio_8d, VH_min_opex, VH_max_opex, dif_VH]]
        return values_1, values_2

    if table == 'table3':
        VI_medio_8d = df1['VI'].head(n).mean().round(2)
        VI_min_opex = df1['VI'].min()
        VI_max_opex = df1['VI'].max()
        dif_VI = VI_max_opex - VI_min_opex
        values_1 = [['Ítem'], ['Value']]
        values_2 = [['VI Media - 8 días', 'VI Mínima', 'VI Máxima', 'Delta VI'],
                    [VI_medio_8d, VI_min_opex, VI_max_opex, dif_VI]]
        return values_1, values_2

    if table == 'table4':
        b_atm = get_bases(df1, df2, n)['Base_ATM']
        px_atm = get_bases(df1, df2, n)['Px_ATM']
        b_otm = get_bases(df1, df2, n)['Base_OTM']
        px_otm = get_bases(df1, df2, n)['Px_OTM']

        values_1 = [['Ítem'], ['Value']]
        values_2 = [['Bases ATM & OTM', 'Px ATM & OTM', 'Ratio', 'Breakeven'],
                    [[b_atm, b_otm], [px_atm, px_otm],
                    round((px_atm/px_otm), 3), round((2*px_otm - px_atm + 2*b_otm - b_atm), 2)]]
        return values_1, values_2


'''=======================================================================
V. HTML REPORT
======================================================================='''
#1. Auxiliares:
# ===== Make Subplots =====:

def html_report(df_call, df_ATM, b_atm, b_otm, px_atm, px_otm, n, break_even, opex):
    '''
    Devuelve el reporte con la información del OpEx y el detalle
    de PnL del Ratio armado durante los primeros n días.
    ---------------------------------------------------------------
    df_call: return call_put(),
    df_ATM: return base_ATM(),
    b_atm: int, return get_bases()['Base_ATM'],
    b_otm: int, return get_bases()['Base_OTM'],
    n: int, días desde iniciado el OpEx,
    break_even: int, return ratio()[0]['break_even'],
    title: str, mm-yy (mes/año vencimiento OpEx).
    ---------------------------------------------------------------
    '''
    rows = 4
    cols = 8
    row_heights = [0.2, 0.8/3, 0.8/3, 0.8/3]
    column_widths = [0.125]*8
    specs=[[{"colspan": 2, "type": "table"}, None, {"colspan": 2, "type": "table"}, None, {"colspan": 2, "type": "table"}, None, {"colspan": 2, "type": "table"}, None],
            [{"colspan": 4},        None        ,        None       ,       None        , {"colspan": 4, "secondary_y": True},    None     ,     None    ,     None   ],
            [{"colspan": 4, "secondary_y": True},     None    ,   None    ,    None     , {"colspan": 4, "secondary_y": True},    None     ,     None    ,     None   ],
            [{"colspan": 4},        None        ,        None       ,       None        , {"colspan": 4},    None     ,     None    ,     None   ]]
    subplot_titles=('Precios Subyacente', 'VH Subyacente', 'VI base ATM', 'Info Ratio', # fila 1
                    'VH 10 y 40d GGAL & VI Base ATM', 'Close Prices GGAL & Base ATM (diaria)', # fila 2
                    'VI Bases ATM/OTM y Ratio', 'PnL Diario', # fila 3
                    'VI Smile - Inicio, Medio y Fin OpEx', 'PnL Ratio')#, # fila 4


    # ===== Set lines and markers =====:
    line1 = dict(color='royalblue', width=2)
    line2 = dict(color='aquamarine', width=2)
    line3 = dict(color='gray', width=2)
    line4 = dict(color='white', width=1.5, dash='dot')
    line5 = dict(color='red', width=2, dash='dot')
    line6 = dict(color='salmon', width=2)

    # marker1 = dict(color=line1['color'], size=8)
    marker1 = dict(color='royalblue', size=8)
    marker2 = dict(color='aquamarine', size=8)
    marker3 = dict(color='gray', size=8)
    marker4 = dict(color='salmon', size=8)


    # ===== Índice fechas únicas del df_call (row=3, cols=1-4) =====:
    index = list(set(df_call['Fecha']))
    index.sort()


    #2. Plot:
    fig = go.Figure(make_subplots(
                                rows=rows, cols=cols, row_heights=row_heights, column_widths=column_widths,
                                vertical_spacing=0.12, horizontal_spacing=0.075,
                                specs=specs, subplot_titles=subplot_titles))

    # ===== Fila 1 - Columna 1-2: Table 1 =====:
    fig.add_trace(go.Table(
                            header=dict(
                                values=info_tables(df_ATM, df_call, 'table1', n)[0],
                                font=dict(size=14),
                                align="center", height=35),
                            cells=dict(
                                values=info_tables(df_ATM, df_call, 'table1', n)[1],
                                font={'size':12},
                                height=30, align="center")
                            ),
                row=1, col=1)

    # ===== Fila 1 - Columna 3-4: Table 2 =====:
    fig.add_trace(go.Table(
                            header=dict(
                                values=info_tables(df_ATM, df_call, 'table2', n)[0],
                                font=dict(size=14),
                                align="center", height=35),
                            cells=dict(
                                values=info_tables(df_ATM, df_call, 'table2', n)[1],
                                font={'size':12},
                                height=30, align="center")
                            ),
                row=1, col=3)

    # ===== Fila 1 - Columna 5-6: Table 3 =====:
    fig.add_trace(go.Table(
                            header=dict(
                                values=info_tables(df_ATM, df_call, 'table3', n)[0],
                                font=dict(size=14),
                                align="center", height=35),
                            cells=dict(
                                values=info_tables(df_ATM, df_call, 'table3', n)[1],
                                font={'size':12},
                                height=30, align="center")
                            ),
                row=1, col=5)

    # ===== Fila 1 - Columna 7-8: Table 4 =====:
    fig.add_trace(go.Table(
                            header=dict(
                                values=info_tables(df_ATM, df_call, 'table4', n)[0],
                                font=dict(size=14),
                                align="center", height=35),
                            cells=dict(
                                values=info_tables(df_ATM, df_call, 'table4', n)[1],
                                font={'size':12},
                                height=30, align="center")
                            ),
                row=1, col=7)


    # ===== Fila 2 - Columnas 1 a 4 =====:
    fig.add_trace(go.Scatter(x=df_ATM.index, y=df_ATM['VI'],
                            mode='lines+markers',
                            marker_symbol='diamond', marker=marker1,
                            line=line1, name='VI Base ATM', legendgroup='1'),
                    row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=df_ATM['volat_40'], 
                            mode='lines+markers',
                            marker_symbol='pentagon', marker=marker2,
                            line=line2,
                            name='Volatilidad 40d', legendgroup='1'),
                    row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=df_ATM['volat_10'],
                            mode='lines+markers',
                            marker_symbol='circle', marker=marker3,
                            line=line3,
                            name='Volatilidad 10d', legendgroup='1'),
                    row=2, col=1)

    # Región correspondiente a los primeros 10 días del OpEx:
    fig.add_trace(go.Scatter(x=df_ATM.index[:10].to_list(), y=[df_ATM['VI'].max()]*10,
                            mode='lines', line_width=0,
                            fill='tozeroy', fillcolor="rgba(200, 230, 230, 0.1)",
                            name='Primeros 10d', legendgroup='1'),
                    row=2, col=1)

    # Líneas Horizontales correspondientes a VI baja, media y alta:
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[35 for i in df_ATM.index],
                            line=line4, name='VI Baja', legendgroup='1'),
                    row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[55 for i in df_ATM.index],
                        line=line4, name='VI Media', legendgroup='1'),
                row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[75 for i in df_ATM.index],
                        line=line4, name='VI ALta', legendgroup='1'),
                row=2, col=1)

    # ===== Fila 2 - Columnas 5 a 8 =====:
    fig.add_trace(go.Scatter(x=df_ATM.index, y=df_ATM['Px_GGAL'],
                            mode='lines+markers', yaxis='y1',
                            marker_symbol='diamond',
                            marker=marker1,
                            line=line1, name='Px GGAL', legendgroup='2'),
                    secondary_y=False, row=2, col=5)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=df_ATM['Close'],
                            mode='lines+markers',
                            marker_symbol='pentagon',
                            marker=marker2,
                            line=line2,
                            name='Px Base ATM', legendgroup='2'),
                    secondary_y=True, row=2, col=5)

    # Región correspondiente a los primeros 10 días del OpEx:
    lim1 = df_ATM['Px_GGAL'].max() + 5
    lim2 = break_even + 5
    lim_max = lambda x : lim1 if (df_ATM['Px_GGAL'].max() > break_even) else lim2
    fig.add_trace(go.Scatter(x=df_ATM.index[:10].to_list(), y=[lim_max(lim1)]*10, # == [lim1 if lim1 > lim2 else lim2]*10
                            mode = 'lines', line_width=0,
                            fill='tozeroy', fillcolor="rgba(200, 230, 230, 0.1)",
                            name='Primeros 10d', legendgroup='2'),
                    row=2, col=5)

    # Líneas Horizontales correspondientes a las Bases ATM y OTM:
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[b_atm for i in df_ATM.index],
                            line=line4, name='Base ATM', legendgroup='2'),
                    row=2, col=5)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[b_otm for i in df_ATM.index],
                            line=line4, name='Base OTM', legendgroup='2'),
                    row=2, col=5)
    fig.add_trace(go.Scatter(x=df_ATM.index, y=[break_even for i in df_ATM.index],
                            line=line5, name='Breakeven', legendgroup='2'),
                    row=2, col=5)


    # ===== Fila 3 - Columnas 1 a 4 =====:
    # index[:-8]  --> para quitar los últimos días del OpEx, donde la VI varía mucho.
    fig.add_trace(go.Scatter(x=index[:-8], y=df_call.loc[df_call['Base'] == b_atm]['VI'][:-8],
                            mode='lines+markers',
                            marker_symbol='diamond',
                            marker=marker1,
                            line=line1,
                            name='VI Base ATM del Ratio', legendgroup='3'),# yaxis="y1"),
                    secondary_y=False, row=3, col=1)
    fig.add_trace(go.Scatter(x=index[:-8], y=df_call.loc[df_call['Base'] == b_otm]['VI'][:-8],
                            mode='lines+markers',
                            marker_symbol='pentagon',
                            marker=marker2,
                            line=line2,
                            name='VI Base OTM del Ratio', legendgroup='3'),#  yaxis="y1"),
                    secondary_y=False, row=3, col=1)
    fig.add_trace(go.Scatter(x=index[:-8], y=df_call.loc[df_call['Base'] == b_atm]['ratio_2B'][:-8],
                            mode='lines+markers',
                            marker_symbol='circle',
                            marker=marker3,
                            line=line3,
                            name='Ratio', legendgroup='3'),
                    secondary_y=True, row=3, col=1)

    # ===== Fila 3 - Columnas 5 a 8 =====:
    lista = [i+1 for i in PnL_day(df_ATM, df_call, b_atm, b_otm, n)[1].reset_index().index]
    lista.insert(0, 0)
    lista1 = list(PnL_day(df_ATM, df_call, b_atm, b_otm, n)[1]['Day_result'])
    lista1.insert(0, PnL_day(df_ATM, df_call, b_atm, b_otm, n)[0])
    colors = lista1.copy()
    colors[0] = 'cyan'
    fechas = list(PnL_day(df_ATM, df_call, b_atm, b_otm, n)[1].index)
    fecha_ratio = get_bases(df_ATM, df_call, n)['Fecha_Ratio']
    fechas.insert(0, fecha_ratio)
    fig.add_trace(go.Bar(x=lista, y=lista1,
                        name='PnL Diario', text=fechas,
                        texttemplate='%{text|%d/%m}', textposition='outside',
                        marker_color=colors, legendgroup='4'),
                    row=3, col=5)


    # ===== Fila 4 - Columnas 1 a 4 =====:
    n1 = 8
    n2 = 20
    n3 = 35
    fig.add_trace(go.Scatter(x=vi_smile(df_call, n1).index, y=vi_smile(df_call, n1)['VI'],
                            mode='lines+markers',
                            marker_symbol='diamond',
                            marker=marker1,
                            line=line1,
                            name='VI Smile Inicio OpEx', legendgroup='5'),
                    row=4, col=1)
    fig.add_trace(go.Scatter(x=vi_smile(df_call, n2).index, y=vi_smile(df_call, n2)['VI'],
                            mode='lines+markers',
                            marker_symbol='pentagon',
                            marker=marker2,
                            line=line2,
                            name='VI Smile Mitad OpEx', legendgroup='5'),
                    row=4, col=1)
    fig.add_trace(go.Scatter(x=vi_smile(df_call, n3).index, y=vi_smile(df_call, n3)['VI'],
                            mode='lines+markers',
                            marker_symbol='circle',
                            marker=marker3,
                            line=line3,
                            name='VI Smile Fin OpEx', legendgroup='5'),
                    row=4, col=1)

    # ===== Fila 4 - Columnas 5 a 8 =====:
    fig.add_trace(go.Scatter(x=ratio(df_call, b_atm=b_atm, b_otm=b_otm, px_atm=px_atm, px_otm=px_otm, n=n, qty_atm=100)[1].index,
                            y=ratio(df_call, b_atm=b_atm, b_otm=b_otm, px_atm=px_atm, px_otm=px_otm, n=n, qty_atm=100)[1],
                            mode='lines+markers',
                            marker_symbol='diamond',
                            marker=marker4,
                            line=line6,
                            name='Ratio PnL', legendgroup='6'),
                    row=4, col=5)


    # ===== Layout =====:
    fig.update_layout(title=f'OPEX REPORT {opex}',
                    title_font=dict(family='Gravitas One', size=26, color='springgreen'),
                    template='plotly_dark',
                    legend_tracegroupgap = 120,
                    showlegend=True,
                    height=1800)

    # ===== X Selectors =====:
    fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                                    buttons=list([
                                                dict(count=10, label="10d", step="day", stepmode="todate"),
                                                dict(count=20, label="20d", step="day", stepmode="backward"),
                                                dict(step="all")]),
                                    bgcolor='royalblue'))

    fig.update_xaxes(rangeslider_thickness = 0.035)

    # some adjustments for yaxis
    fig.update_layout(
            yaxis2=dict(title="$ GGAL",
                titlefont=dict(color='royalblue'),
                tickfont=dict(color='royalblue'),
                range=[df_ATM['Px_GGAL'].min()-5, lim_max(lim1)]),
            yaxis3=dict(title="$ b_ATM",
                titlefont=dict(color='aquamarine'),
                tickfont=dict(color='aquamarine')),
            yaxis5=dict(title="Ratio",
                titlefont=dict(color='grey'),
                tickfont=dict(color='grey'))
            )

    # style all the traces
    fig.update_traces(hoverinfo="name+x+y")

    return fig.show()#, pio.write_image(fig, f"Report_{opex}.pdf", engine="kaleido")


