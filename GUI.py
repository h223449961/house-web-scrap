# -*- coding: utf-8 -*-
# 使用內建的 urllib.request 裡的 urlopen 這個功能來送出網址
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import requests
from styleframe import StyleFrame, Styler
from bs4 import BeautifulSoup
import pandas as pd
import tkinter as tk
import re
import numpy as np
import math
class NeuralNetwork():
    def __init__(self):
        # 随机数生成的种子随机数生成的种子
        np.random.seed(1)
        # 将权重转换为值为-1到1且平均值为0的3乘1矩阵
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # 定义signoid函数的导数
    def sigmoid(self, x):
        x = x.astype(float)
        return (1 / (1 + np.exp(-x)))
    # 计算sigmoid函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 训练
    def train(self, train_inputs, train_outputs,i): # 输入 输出 迭代次数
        # 训练模型在不断调整权重的同时做出准确预测
        for iteration in range(i):
            # 通过神经元提取训练数据
            output = self.think(train_inputs)
            # 反向传播错误率
            error = train_outputs - output
            # 进行权重调整
            adjustments = np.dot(train_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights = self.synaptic_weights + adjustments
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
# 外島
lian = 'eJyrVkrOLKlUsopWMjJQitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJekFHE'
# 東
tung = 'eJyrVkrOLKlUsopWMrRQitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJkdFHg'
hua = 'eJyrVkrOLKlUsopWMrRUitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJlTFHk'
il = 'eJyrVkrOLKlUsopWMlaK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAilgUQg'
# 臺北
taip = 'eJyrVkrOLKlUsopWMlCK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAibYUPw'
# 基隆
ki = 'eJyrVkrOLKlUsopWMlSK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAiewUQA'
# 桃園
yuan = 'eJyrVkrOLKlUsopWMlGK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAio4UQw'
# 新北
newp = 'eJyrVkrOLKlUsopWMlKK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAiiIUQQ'
# 新竹市
shn = 'eJyrVkrOLKlUsopWMlWK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAisQURA'
# 新竹縣
shnn = 'eJyrVkrOLKlUsopWMlOK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAivoURQ'
# 苗栗
mia = 'eJyrVkrOLKlUsopWMleK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAizAURg'
# 南投
to = 'eJyrVkrOLKlUsopWMjRQitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJdtFHA'
# 彰化
cha = 'eJyrVkrOLKlUsopWslSK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAi5wUSA'
# 臺中
taic = 'eJyrVkrOLKlUsopWslCK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAi2YURw'
# 嘉義市
chia = 'eJyrVkrOLKlUsopWMjRSitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJfZFHI'
# 嘉義縣
chiaa = 'eJyrVkrOLKlUsopWMjRWitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJgPFHM'
# 屏東
pin = 'eJyrVkrOLKlUsopWMjRXitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJjnFHc'
# 高雄
kao = 'eJyrVkrOLKlUsopWMjRVitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJh7FHU'
# 臺南
nan = 'eJyrVkrOLKlUsopWMjRRitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJhFFHQ'
# 雲林
yun = 'eJyrVkrOLKlUsopWMjRUitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJejFHE'
root = tk.Tk()
root.geometry("1600x800")
radioValue = tk.IntVar()
exfile_frame = tk.Frame(root)
exfile_frame.pack(side=tk.TOP)
exfile_label = tk.Label(exfile_frame, text='快速點擊爬取',font=(24),fg = "#30aabc")
exfile_label.pack(side=tk.LEFT)
def validate():
    value = radioValue.get()
    global mode
    if (value == 1):
        mode = tung
    if( value == 2):
        mode = lian
    if( value == 3):
        mode = hua
    if (value == 4):
        mode = mia
    if( value == 6):
        mode = pin
    if (value == 7):
        mode = kao
    if( value == 8):
        mode = shn
    if( value == 9):
        mode = shnn
def oas():
    global mode
    df01 = pd.DataFrame(columns=["型態",'範例'])
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',}
    sch = ''
    html = requests.get(url='https://www.rakuya.com.tw/search/rent_search/index?con='+mode,headers=headers).content
    soup = BeautifulSoup(html,'lxml')
    df01 = pd.DataFrame(columns=['名稱','類型','格局','樓層加權分'])
    nams = soup.find_all('div',class_='obj-info')
    for nam in nams:
        nmes = nam.find('h6')
        ifo=[]
        datas = nam.find_all('li',class_='clearfix')
        for data in datas:
            rs = data.text.replace('\n', '')  # 替换换行符
            ifo.append(rs)
        gstr = ''.join(ifo)
        a,b,c = gstr.split('/')
        characters = "-"
        b = ''.join( x for x in b if x not in characters)
        if(c=='-'):
            c = '0.2樓'
        #print(a+' and '+b+' and '+c)
        num = [float(s) for s in re.findall(r'-?\d+\.?\d*', b)]
        #print(num,sum(num))
        floors = [float(s) for s in re.findall(r'-?\d+\.?\d*', c)]
        flor = float(''.join(str(i) for i in floors))
        staweighs = (num[-1]/flor)
        if(staweighs>1.2):
            staweighs = 0.02
        s01 = pd.Series([nmes.text,a,b,sum(num),staweighs], index=['名稱','類型','格局','格局總分','樓層加權分'])
        df01 = df01.append(s01, ignore_index=True)
        df01.index = df01.index+1
    strongmap =  {'雅房':10,'分租套房':10.9,'獨立套房':12,'整層住家':40,'住宅':39,'土地':264,'廠房':1589,
                  '其他':118,'商用':112,'住辦':103,'店面':99}
    df01['類型加權分'] = df01['類型'].map(strongmap)
    vgrab = []
    djrab = []
    average = 0
    adjust = 0
    for ind in df01.index:
        average = df01['格局總分'][ind]/df01['類型加權分'][ind]
        adjust = df01['格局總分'][ind]*df01['樓層加權分'][ind]
        vgrab.append(average)
        djrab.append(adjust)
    sigdrab = [1/(1+math.exp(-i)) for i in djrab]
    df01['格局加權分'] = vgrab
    df01['校正加權分'] = sigdrab
    weightsum = 0
    ave = 0
    for ind in df01.index:
        weightsum = weightsum+df01['格局加權分'][ind]
    ave = weightsum/len(df01.index) 
    print(ave)
    result = []
    for ind in df01.index:
        if(df01['格局加權分'][ind]>ave):
            result.append("推薦")
        else:
            result.append("普通")
    df01["公式計算"] = result
    dft = df01[['樓層加權分','格局加權分','校正加權分']]
    dftt = df01[['樓層加權分','格局加權分','校正加權分']]
    rab = []
    for ind in df01.index:
        if(df01['公式計算'][ind]=='推薦'):
            rab.append(1)
        else:
            rab.append(0)
    dft['rabel'] = rab
    num = dftt.to_numpy()
    numt = num.reshape(1,-1).T
    ranu = dft['rabel'].to_numpy()
    ranut = ranu.reshape(1,-1).T
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    def clo(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    print(num)
    print(ranut)
    b = NeuralNetwork()
    b.train(num, ranut,90000)
    classif = softmax(b.think(num))
    print(classif)
    cround = np.round(classif, 2)
    cro = pd.DataFrame(cround,columns = ['機率'])
    print(cro)
    print('總機率： '+str(np.sum(cround)))
    print('最接近總機率的值： '+str(clo(cround,1)))
    flg = clo(cro,1)
    nlearn = []
    for ind in cro.index:
        if(cro['機率'][ind]==flg):
            nlearn.append('推薦')
        else:
            nlearn.append('普通')
    df01["神經網路"] = nlearn
    print(df01)
    acoun = 0
    bcoun = 0
    altrue = 0
    preci = 0
    for i in df01.index:
        truflg = []
        if(df01['公式計算'][i]==df01['神經網路'][i]):
            acoun = acoun+1
        else:
            bcoun = bcoun+1
        if(df01['神經網路'][i]=='推薦'):
            altrue = altrue +1
        if(df01['公式計算'][i]=='推薦'):
            truflg = df01['公式計算'][i]
        if(truflg==df01['神經網路'][i]):
            preci= preci+1
    print((acoun/(acoun+bcoun))*100,(preci/altrue)*100)
    acpre_label.configure(text ='準確： '+str((acoun/(acoun+bcoun))*100)+' %')
    sf = StyleFrame(df01)
    sf.set_column_width_dict(col_width_dict={("名稱"): 60,("類型"): 13,("格局"): 48,("樓層加權分"): 29,("格局加權分"): 27,("校正加權分"): 27})
    sname = 'hog.xlsx'
    output = sf.to_excel(sname).save()
    df = pd.read_excel(sname)
    cols = list(df.columns)
    def treeview_sort_column(tv, col, reverse):
        try:
            l = [float((tv.set(k, col)), k) for k in tv.get_children('')]
        except:
            l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)

    # rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

    # reverse sort next time
        tv.heading(col, command=lambda:
            treeview_sort_column(tv, col, not reverse))
    tree = ttk.Treeview(root)
    tree.pack()
    tree["columns"] = cols
    tree.column("#0", width=0, stretch=False)
    for i in cols:
        # 店名
        tree.column('# 1',width =157,anchor="center")
        # 類型
        tree.column('# 2',width =22,anchor="center")
        # 格局
        tree.column('# 3',width =70,anchor="center")
        # 總分
        tree.column('# 4',width =0,anchor="center")
        # 素食種類
        tree.column('# 5',width =22,anchor="center")
        # 可配合素食加權分
        tree.column('# 6',width =48,anchor="center")
        # 休息時間
        tree.column('# 7',width =22,anchor="center")
        # 全勤加權分
        tree.column('# 8',width =3,anchor="center")
        # 可手機聯絡加權分
        tree.column('# 9',width =39,anchor="center")
        # 地址
        tree.column('# 10',width =139,anchor="center")
        # 推薦
        #tree.column('# 11',width =0,anchor="center")
        # 推薦
        #tree.column('# 12',width =10,anchor="center")
        tree.heading(i,text=i,anchor='center')
        tree.heading(i, text=i,command=lambda c=i: treeview_sort_column(tree, c, False))
    for index, row in df.iterrows():
        tree.insert("",'end',text = index,values=list(row))
    tree.place(relx=0,rely=0.4,relheight=0.5,relwidth=1)
r1 = tk.Radiobutton(root,text = "臺東",font=(24),variable=radioValue, value=1,command = validate).pack()
r2 = tk.Radiobutton(root,text = "外島",font=(24),variable=radioValue, value=2,command = validate).pack()
r3 = tk.Radiobutton(root,text = "花蓮",font=(24),variable=radioValue, value=3,command = validate).pack()
r4 = tk.Radiobutton(root,text = "苗栗",font=(24),variable=radioValue, value=4,command = validate).pack()
r6 = tk.Radiobutton(root,text = "屏東",font=(24),variable=radioValue, value=6,command = validate).pack()
r7 = tk.Radiobutton(root,text = "高雄",font=(24),variable=radioValue, value=7,command = validate).pack()
r8 = tk.Radiobutton(root,text = "新竹縣",font=(24),variable=radioValue, value=8,command = validate).pack()
r9 = tk.Radiobutton(root,text = "新竹市",font=(24),variable=radioValue, value=9,command = validate).pack()
def com(*args): # 處理事件， *args 表示可變引數  
    global mode
    mode = comboxlist.get()
    if(mode == '桃園'):
        mode = 'eJyrVkrOLKlUsopWMlGK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAio4UQw'
    if(mode == '雲林'):
        mode = 'eJyrVkrOLKlUsopWMjRUitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJejFHE'
    if(mode == '臺南'):
        mode = 'eJyrVkrOLKlUsopWMjRRitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJhFFHQ'
    if(mode == '基隆'):
        mode = 'eJyrVkrOLKlUsopWMlSK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAiewUQA'
    if(mode == '臺北'):
        mode = 'eJyrVkrOLKlUsopWMlCK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAibYUPw'
    if(mode == '新北'):
        mode = 'eJyrVkrOLKlUsopWMlKK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAiiIUQQ'
    if(mode == '臺中'):
        mode = 'eJyrVkrOLKlUsopWslCK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAi2YURw'
    if(mode == '嘉義縣'):
        mode = 'eJyrVkrOLKlUsopWMjRWitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJgPFHM'
    if(mode == '嘉義市'):
        mode = 'eJyrVkrOLKlUsopWMjRSitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJfZFHI'
    if(mode == '宜蘭'):
        mode = 'eJyrVkrOLKlUsopWMlaK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAilgUQg'
    if(mode == '南投'):
        mode = 'eJyrVkrOLKlUsopWMjRQitVRSsksLshJBAoo5WQWlyjpKOUnZWXmpYAUBIHki1MTi5IzQFywvthaAJdtFHA'
    if(mode == '彰化'):
        mode = 'eJyrVkrOLKlUsopWslSK1VFKySwuyEkE8pVyMotLlHSU8pOyMvNSQPJBIPni1MSi5AwQF6wtthYAi5wUSA'
comvalue=tk.StringVar()
comboxlist=ttk.Combobox(root,textvariable=comvalue)
comboxlist["values"]=("桃園","雲林","臺南",'基隆','臺北','新北','臺中','嘉義縣','嘉義市','宜蘭','南投','彰化')  
comboxlist.current(0)
comboxlist.bind("<<ComboboxSelected>>",com) # 繫結事件，（下拉列表框被選中時，繫結綁定的函式）  
comboxlist.pack()
acpre_frame = tk.Frame(root)
acpre_frame.pack(side=tk.TOP)
b1 = tk.Button(acpre_frame, text="爬取（若無特別設定將爬取上個暫存的縣市記錄）",font=(24),command = oas).pack(side=tk.LEFT)
acpre_label = tk.Label(acpre_frame, text='精準度',font=(24),fg = "#C13E43")
acpre_label.pack(side=tk.RIGHT)
root.mainloop()
