# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:56:36 2024

@author: TedLove
"""

import pandas as pd
import numpy as np

def clean_up_order_book(order_book, LOB):
        
    order_book['price'] = order_book['price'] * 1e-4

    order_book.index = order_book.index - order_book.index.min()
    
    LOB.index = order_book.index
    
    order_book['midPrice'] = LOB['midPrice']
    order_book_filtered = order_book.loc[order_book['type']>=4]

  
    order_book_filtered = order_book_filtered.loc[order_book_filtered['type']>=4]
    
    
    X = -order_book_filtered['direction'] *( order_book_filtered['price']-order_book_filtered['midPrice'] )
    X = X.rename('X')
    
    order_book_filtered = pd.concat([order_book_filtered,X],axis=1)
    return order_book_filtered, LOB
    
    
def clean_up_LOB(LOB):
    
    col_names = []
    for i in range(1,3):
        col_names.append('ask ' + str(i))
        col_names.append('ask vol ' + str(i))
        col_names.append('bid ' + str(i))
        col_names.append('bid vol ' + str(i))
  
    LOB.columns = col_names
    
    for column in LOB.columns:
        if 'vol' not in column:
            LOB[column] =  LOB[column] * 1e-4
    
    LOB['midPrice'] = ((LOB['ask 1'] +  LOB['bid 1']) / 2  )
    
    LOB.index = (LOB.index - LOB.index[0])
    

    return LOB
    
    
    
    
    
        
        
        
    
    