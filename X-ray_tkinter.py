# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:16:39 2023

@author: D01
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
from PIL import Image, ImageTk, ImageOps 
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import imageio



#一般分有病沒病		
def upload_and_recognize():
    # 顯示選擇文件對話框
    filepath = filedialog.askopenfilename(filetypes=[('jpeg', '*.jpeg'),('png', '*.png'),('jpg', '*.jpg'),('gif', '*.gif')]) 
    if not filepath:
        return

    # 讀取圖片文件並使用 PIL 將其轉換為 PhotoImage 對象
    image1 = Image.open(filepath)
    image1 = image1.resize((500,500))
    w, h = image1.size
    image = ImageTk.PhotoImage(image1)
    canvas.delete('all')                 # 清空 Canvas 原本內容
    canvas.config(scrollregion=(0,0,w,h))   # 改變捲動區域
    canvas.create_image(0, 0, anchor='nw', image=image)   # 建立圖片
    canvas.image = image               # 修改属性更新畫面
    try:
        # 讀取圖像並預處理
        data = np.ndarray(shape=(1, 255, 255, 3), dtype=np.float32)
        image1 = image1.resize((255,255))
		#turn the image into a numpy array
        image_array = np.stack((image1,)*3, axis=-1)
        #pretrained前處理
        preprocessing_funct = tf.keras.applications.efficientnet_v2.preprocess_input
        image_array = preprocessing_funct(image_array)
        normalized_image_array = (image_array.astype(np.float32)/255.0)
        data[0] = normalized_image_array
        # 使用模型進行圖像辨識
        result = model.predict(data)
		# 清除之前的結果
        text.delete(1.0, tk.END)
        prob_str = []
        for prob in result[0]:
            if prob > 0.3:
                data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
                image1 = image1.resize((256,256))
        		#turn the image into a numpy array
                image_array = np.stack((image1,)*3, axis=-1)
                #pretrained前處理
                normalized_image_array = (image_array.astype(np.float32)/255.0)
                data[0] = normalized_image_array
                # 使用模型進行圖像辨識
                result2 = model2.predict(data)
                prob_str = []
                for prob in result2[0]:
                    if prob > 0.5:
                        prob_str.append("virus:{:.2f}%".format(prob * 100))
                    else:
                        prob_str.append("bacteria:{:.2f}%".format(prob * 100))
            else:
                prob_str.append("normal:{:.2f}%".format(prob * 100))

          
		# 檢查機率值是否大於30% 
        '''for prob in result[0]:
            if prob > 0.3:
                button1.config(state=tk.NORMAL)
                break
            else:
                button1.config(state=tk.DISABLED)'''
        '''for prob in result[0]:
            if prob > 0.3:
                tk.messagebox.showinfo("恭喜!獲得顏值測試服務", "機率值大於30%")
                break'''
		# 將結果顯示在文本框中
        text.insert('end',filepath.split('/')[-1]+"\n")
        text.insert('end',', '.join(prob_str))
    except Exception as e:
        # 將異常信息顯示在文本框中
        text.insert('end', str(e))

# 建立主窗口
root = tk.Tk()
root.title('PNEUMONIA Test')

#建立label
label1 = tk.Label(root, 
                  pady=20,
                  text='X-RAY NORMAL/PNEUMONIA',
                  font=('DIN Alternate Light',20),
                  fg='#BAF8FF',
                  bg='#2D4D67')
#'bold'#8B4513#F5DEB3
label1.pack(fill='x',side='top')

#建立畫布
frame = tk.Frame(root, bd=10,relief="sunken",bg='#2D4D67')                  
frame.pack()



canvas = tk.Canvas(frame, width=500, height=450, bg='#fff')
#建立捲軸
'''scrollX = tk.Scrollbar(frame, orient='horizontal')
scrollX.pack(side='bottom', fill='x')
scrollX.config(command=canvas.xview)

scrollY = tk.Scrollbar(frame, orient='vertical')
scrollY.pack(side='right', fill='y')
scrollY.config(command=canvas.yview)'''

#canvas.config(xscrollcommand=scrollX.set, yscrollcommand=scrollY.set)
canvas.pack(side='left')

# 建立按鈕
button = tk.Button(root, text='上傳圖片及辨識', command=upload_and_recognize)
button.pack()

'''button1 = tk.Button(root, text='顏值測試', command=upload_and_recognize)
button1.pack()'''

# 建立 Label 對象
label = tk.Label(root)
label.pack()

# 創建文本框
text = tk.Text(root)
text.pack()

# 加載訓練好的模型
model = load_model("C:\\Users\\88691\\Downloads\\ChestX-Ray2_2313121_acc95_rec97_prec95.h5")
model2 = load_model("C:\\Users\\88691\\Downloads\\ChestX-Ray2BV_23141627_acc92_rec89_prec89.h5")

# 顯示主窗口
root.mainloop()
