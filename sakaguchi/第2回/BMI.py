# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:45:04 2018

@author: yu
"""

import pdb

class BMICalc(object):
    def __init__(self,height,weight):
        self.height = height
        self.weight = weight
        print("create class")

    def bmi_calc(self):
        h = self.height / 100
        self.bmi = self.weight / (h**2)
        print(self.bmi)

    def classifier(self):
        # BMI計算した結果を判断
        if self.bmi < 18.5:
            print("やせ型")
        elif self.bmi < 25:
            print("標準")
        elif self.bmi < 30:
            print("やや肥満")
        else:
            print("肥満")



if __name__ == "__main__":
    # Taro
    height = 30
    weight = 180
    #インスタンス化
    taro = BMICalc(height,weight)
    taro.bmi_calc()
    taro.classifier()

    # Sato
    height = 60
    weight = 170
    sato = BMICalc(height,weight)
    sato.bmi_calc()
    sato.classifier()

    # Kobayashi
    height = 40
    weight = 168
    kobayashi = BMICalc(height,weight)
    kobayashi.bmi_calc()
    kobayashi.classifier()

## パラメータ
# height:身長
# weight:体重
# bmi:BMI計算

## 判定する3人 ##



    
