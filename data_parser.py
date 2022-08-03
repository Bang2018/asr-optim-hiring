# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 22:58:15 2022
Amadeus
@author: XYZ
"""

import pandas as pd
import numpy as np
from mip import *
from pathlib import Path
import os
import time

DIR = "E:/DeepLrningCode/Amadeus/data/"
FILE = "instance"

No_Machines = 0
Budget = 0
Restruct_Period = 0
BigM = 99999
Strategy1 = "Buy and Sale on Same Day"
Strategy2 = "Buy First and Sale Later"

def load_data(dir_path,fname,strategy,_machine=0,budget=0,restruct=0):
    START = time.time()
    day=[]
    price=[]
    resale=[]
    daily_profit=[]
    N=[]
    C=[]
    D=[]
    Des=[]
    case_no=0
    result_logic=[]
    fpath = Path(os.path.join(dir_path,fname+".txt"))
    if fpath.exists():
        print(f"{fname} is available")
        with open(os.path.join(dir_path,fname+".txt"),"r") as f:
            contents = f.readlines()
            line_ptr=0
            for line_no,line in enumerate(contents,start=1):
                value = line.replace("\n"," ").strip().split(" ")
                if len(value)==3:
                   case_no +=1
                   n_machine=int(value[0])
                   N.append(n_machine)
                   line_ptr =line_no + n_machine
                   print("=="*10)
                   print(f"STRATEGY : {strategy}")
                   print("=="*10)
                   print(f"\nCase No: {case_no}\n")
                   print(f"No. of Machines {n_machine}")
                   
                   budget = int(value[1])
                   print(f"Budget {budget}")
                   C.append(budget)
                   restruct = int(value[2])+1 #for D+1 period
                   D.append(restruct-1)
                   print(f"Restruct Period (D+1) {restruct}")
                   #print(f"Line No {line_no} and {line_ptr}\n")
                if len(value)==4:
                       value = line.replace("\n"," ").strip().split()
                       
                       day.append(int(value[0]))
                       price.append(int(value[1]))
                       resale.append(int(value[2]))
                       daily_profit.append(int(value[3]))
                if line_no ==line_ptr:
                   data = pd.DataFrame({"Day":day,"Price":price,"Resale":resale,"Daily_Profit":daily_profit}) 
                   print(data.head())
                   res = mip_solver(data,n_machine,budget,restruct,case_no,strategy)
                   res_logic = logic(data,n_machine,budget,restruct,case_no,strategy) 
                   result_logic.append(f"Case {case_no} : {res_logic}")
                   Des.append(f"Case {case_no} : {res}")
                   day=[]
                   price=[]
                   resale=[]
                   daily_profit=[]
                
        f.close()
        #data = pd.DataFrame({"Day":day,"Price":price,"Resale":resale,"Daily_Profit":daily_profit}) 
        #print(data.head())
        result = pd.DataFrame({"N":N,"C":C,"D":D,"ROI(Exact)":Des,"ROI(Simuation)":result_logic})
        END = time.time()
        print("=="*20)
        print(f"\nSTRATEGY : {strategy}")
        print("\nSolution Using Exact and Simulation Method")
        print("=="*20)
        print(result)
        print(f"\nCompleted in {round(END-START,2)} secs")
    


def mip_solver(df:pd.DataFrame,No_Machines:int,Budget:int,Restruct_Period:int,Case_No:int,Strategy:str):
    print("\nExact Solution:\n")
    if len(df)==1:
        print("Not using solver..")
    objective=0
    if len(df)==1:
        if df["Price"].iloc[0] - df["Resale"].iloc[0]<= Budget:
            sale = df["Price"].iloc[0] - df["Resale"].iloc[0]
            return df["Daily_Profit"].iloc[0]*(Restruct_Period-df["Day"].iloc[0])-sale
    sold_machine=[]
    resale = 0
    START = time.time()
    model = Model()
    available_days = [df["Day"].iloc[i] for i in range(len(df))]
    x_bin = [model.add_var(var_type=BINARY,name=f"x_{i}") for i in range(Restruct_Period)]
    
    y_bin = [[model.add_var(var_type=BINARY,name=f"y_{i}_{j}") for j in range(No_Machines)] for i in range(Restruct_Period)]
    
    indx = {}
    for i in range(Restruct_Period):
        if i in available_days:
           #print(i)
           indx[i] = df[df["Day"]==i].index[0]
    
    expr=LinExpr()
    for i in range(Restruct_Period): 
        if i in available_days:
            if available_days.count(i) >1:
               temp = list(df[df["Day"]==i].index)
               for j in range(len(temp)):
                   if Strategy=="Buy and Sale on Same Day":
                      expr += y_bin[i][j]*df["Daily_Profit"].iloc[j]*(Restruct_Period-df["Day"].iloc[j])-y_bin[i][j]*(df["Price"].iloc[j]-df["Resale"].iloc[j])
                   if Strategy=="Buy First and Sale Later":
                       #print(expr)
                       expr += y_bin[i][j]*df["Daily_Profit"].iloc[j]*(Restruct_Period-df["Day"].iloc[j])-y_bin[i][j]*(df["Price"].iloc[j])
               model.objective = maximize(expr)
            else:
                if Strategy=="Buy and Sale on Same Day":
                    model.objective = maximize(xsum(x_bin[i]*df["Daily_Profit"].iloc[indx[i]]*(Restruct_Period-df["Day"].iloc[indx[i]])for i in range(Restruct_Period) if i in available_days)-\
                           xsum(x_bin[i]*(df["Price"].iloc[indx[i]]-df["Resale"].iloc[indx[i]]) for i in range(Restruct_Period) if i in available_days))
                if Strategy=="Buy First and Sale Later":
                    model.objective = maximize(xsum(x_bin[i]*df["Daily_Profit"].iloc[indx[i]]*(Restruct_Period-df["Day"].iloc[indx[i]])for i in range(Restruct_Period) if i in available_days)-\
                           xsum(x_bin[i]*(df["Price"].iloc[indx[i]]) for i in range(Restruct_Period) if i in available_days))
    expr2 = mip.LinExpr()
    for i in range(Restruct_Period): 
        if i in available_days:
            if available_days.count(i) >1:
               temp = list(df[df["Day"]==i].index)
               for j in range(len(temp)):
                   if Strategy=="Buy and Sale on Same Day":
                       expr2 += y_bin[i][j]*(df["Price"].iloc[j]-df["Resale"].iloc[j])
                   if Strategy=="Buy First and Sale Later":
                       expr2 += y_bin[i][j]*(df["Price"].iloc[j])
               model += expr2 <= Budget
            else:   
                 if Strategy=="Buy and Sale on Same Day":
                    model += xsum(x_bin[i]*(df["Price"].iloc[indx[i]]-df["Resale"].iloc[indx[i]]) for i in range(Restruct_Period) if i in available_days) <= Budget
                 if Strategy=="Buy First and Sale Later":
                    model += xsum(x_bin[i]*(df["Price"].iloc[indx[i]]) for i in range(Restruct_Period) if i in available_days) <= Budget 
    status = model.optimize()
    fresult = DIR[:-5] + "/result/result.txt"
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
           print('Result:\n')
           objective = model.objective_value
           print(f'Model Objective: {objective}')
           with open(fresult,"w") as fres:
                for v in model.vars:
           
                     if abs(v.x) > 1e-6: # only printing non-zeros
                         print('{} = {}'.format(v.name, v.x))
                         machine = v.name
                         machine = machine[-1]
                         sold_machine.append(int(machine))
                         fres.write(f"{v.name} = {v.name}")
                         fres.write("\n")
           fres.close()
           print(f"List of machines sold on days : {sold_machine}\n")
           print(df)
           for i in range(len(df)):
               if Strategy=="Buy and Sale on Same Day":
                  if df["Day"].iloc[i] in sold_machine:
                   pass
                  else:
                      pr = df["Price"].iloc[i]
                      re = df["Resale"].iloc[i]
                      resale += pr-re 
               if Strategy=="Buy First and Sale Later":
                  re = df["Resale"].iloc[i]
                  print(re)
                  resale+=re
           print(f"Earning from resale of remaining machines $ {resale} ")        
           
    fpath = DIR[:-5] + "/LP" + "/" +f"Buy_Sale_{Case_No}.lp"
    model.write(fpath)
    END=time.time()
    if Strategy=="Buy and Sale on Same Day":
       r = objective-resale
    if Strategy=="Buy First and Sale Later":
        r = objective+resale
    print(f"ROI : {r}")
    print(f"\nComputational Time {round(END-START,2)} secs\n")
    
    return r

def logic(df:pd.DataFrame,No_Machines:int,Budget:int,Restruct_Period:int,Case_No:int,Strategy:str):
    print("\nSimulation:\n")
    sold_machine = []
    remaining_machines=[]
    resale1=0
    earnings=0
    cum =0
    if Strategy=="Buy and Sale on Same Day":
       df["Pay"] = df["Price"]-df["Resale"]
    if Strategy=="Buy First and Sale Later":
        df["Pay"] = df["Price"]
    #Sort the dataframe by Pay
    df = df.sort_values(by="Pay",ascending=True,kind="quicksort")
    print(df)
    for i in range(len(df)):
        if df["Pay"].iloc[i] <= Budget-cum:
            cum += df["Pay"].iloc[i]
            #print(f"Cumulative {cum}")
            profit = df["Daily_Profit"].iloc[i]
            day = df["Day"].iloc[i]
            price = df["Price"].iloc[i]
            resale = df["Resale"].iloc[i]
            if Strategy=="Buy and Sale on Same Day":
               earnings += df["Daily_Profit"].iloc[i]*(Restruct_Period-df["Day"].iloc[i]) -\
                       (df["Pay"].iloc[i])
            if Strategy=="Buy First and Sale Later":
                earnings += df["Daily_Profit"].iloc[i]*(Restruct_Period-df["Day"].iloc[i]) -\
                       (df["Price"].iloc[i])
            sold_machine.append(i)
    for i in range(len(df)):
        if Strategy=="Buy and Sale on Same Day":
           if i not in sold_machine:
              remaining_machines.append(i)
              pay1 = df["Pay"].iloc[i] 
              #print(f"{pay1}")
              resale1 += pay1
        if Strategy=="Buy First and Sale Later":
            remaining_machines.append(i)
            pay1 = df["Resale"].iloc[i] 
            #print(f"{pay1}")
            resale1 += pay1
    sold = [df["Day"].iloc[i] for i in sold_machine]
    remaining = [df["Day"].iloc[i] for i in remaining_machines]
    print(f"Machines are sold on days {sold}, Earnings: $ {earnings}")
    print(f"Machines not sold on days {remaining}, Resale: $ {resale1}")
    if Strategy=="Buy and Sale on Same Day":
       r2 = earnings-resale1 
       print(f"ROI : {r2}")
       return r2               
    if Strategy=="Buy First and Sale Later":
       r2=earnings+resale1
       return r2
            




load_data(DIR,FILE,Strategy1)


