# -*- coding: utf-8 -*-
# 使用內建的 urllib.request 裡的 urlopen 這個功能來送出網址
import requests
from bs4 import BeautifulSoup
import pandas as pd
import tkinter as tk
def main():
    global df01
    df01 = pd.DataFrame(columns=["型態",'範例'])
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',}
    sch = ''
    html = requests.get(url='https://www.rakuya.com.tw/search/rent_search/index?con=eJyrVkrOLKlUsopWMjJQitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJekFHE', headers=headers).content
    soup = BeautifulSoup(html,'lxml')
    df01 = pd.DataFrame(columns=["名稱","資訊"])
    
    
    
    
    
    
    
    nams = soup.find_all('div',class_='obj-info')
    for nam in nams:
        nmes = nam.find('h6')
        dtnam = nmes.text
        ifo=[]
        datas = nam.find_all('li',class_='clearfix')
        for data in datas:
            rs = data.text.replace('\n', '')  # 替换换行符
            ifo.append(rs)
        gstr = ''.join(ifo)
        print(dtnam+' : '+gstr)
        s01 = pd.Series([nmes.text,gstr], index=['名稱','資訊'])
        df01 = df01.append(s01, ignore_index=True)
        df01.index = df01.index+1
    #print(df01.to_string())
if __name__ == '__main__':
    main()
