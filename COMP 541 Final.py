import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import tkinter as tk
from tkinter import *



#All ML stuff
data= pd.read_csv("dataset.csv")
tester= pd.read_csv("Test.csv")
tester.head()
data['BEDS'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')

Tester=tester.drop(['ADDRESS', 'CITY','PRICE'],axis=1)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = data['PRICE']

train1 = data.drop(['ADDRESS','CITY', 'PRICE'],axis=1)
train1.head()
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


model = GradientBoostingRegressor()

parameters = {'learning_rate': [0.01,0.1,0.5,0.7],
                  
                  'n_estimators' : [50,250,500,1000],
                  'max_depth'    : [6,8,10,20] 
                 }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(x_train, y_train)    
from sklearn import ensemble
params = {'n_estimators' : 500, 'max_depth' : 6, 'min_samples_split' : 2,
          'learning_rate' : 0.1, 'loss' : 'ls'}
regression = ensemble.GradientBoostingRegressor(**params)

regression.fit(x_train, y_train)
regression.score(x_test,y_test)
t_sc = np.zeros((params['n_estimators']),dtype=np.float64)

testsc = np.arange((params['n_estimators']))+1


#All UI Stuff


def address_info(Tester):
	window2=tk.Tk()
	window2.title("Address Info")
	window2.geometry("500x600")
	my_frame=Frame(window2, width=500, height= 600)
	
	Mainlabel= tk.Label(my_frame,text= "Enter Home Details Below:")
	Mainlabel.config(font=("Courier", 15,"bold"))
	Mainlabel.place(x= 50, y = 15)

	

	#Creation of labels
	Address_label= tk.Label(my_frame,text= "Address:",font=("Courier", 10,"bold"))
	City_label= tk.Label(my_frame,text= "City:",font=("Courier", 10,"bold"))
	Zip_label=tk.Label(my_frame,text= "Zip Code:",font=("Courier", 10,"bold"))
	Price_Label=tk.Label(my_frame,text= "Current Price:",font=("Courier", 10,"bold"))
	Beds_label = tk.Label(my_frame,text= "Beds:",font=("Courier", 10,"bold"))
	Baths_label = tk.Label(my_frame,text= "Baths:",font=("Courier", 10,"bold"))
	SQFT_Label= tk.Label(my_frame,text= "Square Feet:",font=("Courier", 10,"bold"))
	Lot_label=tk.Label(my_frame,text= "Lot Size:",font=("Courier", 10,"bold"))
	Year_label= tk.Label(my_frame,text= "Year Built:",font=("Courier", 10,"bold"))
	Days_label=tk.Label(my_frame,text= "Days on Market:",font=("Courier", 10,"bold"))
	SQFT_price_label=tk.Label(my_frame,text= "Price Per Square Foot:",font=("Courier", 10,"bold"))
	HOA_label=tk.Label(my_frame,text= "Enter HOA (If no HOA put 0):",font=("Courier", 10,"bold"))



	#Creation of Text boxes
	address_entry= tk.Entry(my_frame,textvariable=address)
	city_entry= tk.Entry(my_frame,textvariable=city)
	zip_entry= tk.Entry(my_frame,textvariable=zipcode)
	price_entry= tk.Entry(my_frame,textvariable=price)
	beds_entry= tk.Entry(my_frame,textvariable=beds)
	baths_entry= tk.Entry(my_frame,textvariable=baths)
	sqft_entry= tk.Entry(my_frame,textvariable=sqft)
	lot_entry= tk.Entry(my_frame,textvariable=lot)
	Year_entry= tk.Entry(my_frame,textvariable=year_bulit)
	days_entry= tk.Entry(my_frame,textvariable=days_market)
	sqft_price_entry= tk.Entry(my_frame,textvariable=sqft_price)
	HOA_entry= tk.Entry(my_frame,textvariable=HOA)

	#Submit Button
	Submit_bttn= tk.Button(my_frame, text='Submit', command=lambda:Save(Tester))
	def Save(Tester):
		Address_string=address_entry.get()
		City_string= city_entry.get()
		Tester['ZIP OR POSTAL CODE']= zip_entry.get()
		Tester['BEDS']=beds_entry.get()
		Tester['BATHS']=baths_entry.get()
		Tester['SQUARE FEET']=sqft_entry.get()
		Tester['LOT SIZE']=lot_entry.get()
		Tester['YEAR BUILT']=Year_entry.get()
		Tester['DAYS ON MARKET']=days_entry.get()
		Tester['$/SQUARE FEET']=sqft_price_entry.get()
		Tester['HOA/MONTH']=HOA_entry.get()
		y_pred=regression.predict(Tester)
		Estimated_price.configure(text= "Estimated Price: "+ str(y_pred),font=("Courier", 12,"bold") )
		Address_label_main.configure(text=Address_string+City_string,font=("Courier", 12,"bold"))
		window2.destroy()
		
	#Placing of all objects
	Address_label.place(x=50, y=45)
	City_label.place(x=50,y=65)
	Zip_label.place(x=50,y=85)
	Price_Label.place(x=50,y=105)
	Beds_label.place(x=50,y=125)
	Baths_label.place(x=50,y=145)
	SQFT_Label.place(x=50,y=165)
	Lot_label.place(x=50, y=185)
	Year_label.place(x=50,y=205)
	Days_label.place(x=50,y=225)
	SQFT_price_label.place(x=50,y=245)
	HOA_label.place(x=50,y=265)
	address_entry.place(x=325,y=45)
	city_entry.place(x=325,y=65)
	zip_entry.place(x=325,y=85)
	price_entry.place(x=325,y=105)
	beds_entry.place(x=325,y=125)
	baths_entry.place(x=325,y=145)
	sqft_entry.place(x=325,y=165)
	lot_entry.place(x=325,y=185)
	Year_entry.place(x=325,y=205)
	days_entry.place(x=325,y=225)
	sqft_price_entry.place(x=325,y=245)
	HOA_entry.place(x=325,y=265)
	Submit_bttn.place(x=250, y=300)



	
	my_frame.pack()
	
window= tk.Tk()


# Global Vars
address=tk.StringVar()
city=tk.StringVar()
zipcode=tk.IntVar()
price=tk.IntVar()
beds=tk.StringVar()
baths=tk.IntVar()
sqft=tk.IntVar()
lot=tk.IntVar()
year_bulit=tk.IntVar()
days_market=tk.IntVar()
sqft_price=tk.IntVar()
HOA=tk.IntVar()

window.geometry("500x300")
my_frame2=Frame(window, width=500, height= 300)
my_frame2.pack()

label= tk.Label(text= "Housing Price predictor")
label.config(font=("Courier", 13,"bold"))
label.place(x=250,anchor='n')
Enter_info= tk.Button(my_frame2,text="Click here to enter housing information",command= lambda:address_info(Tester))
Enter_info.place(x=250, y=100, width=220, height=50, anchor = 'center')
Address_label_main= tk.Label(my_frame2,text="")
Address_label_main.place(x=250, y=215, anchor='center')
Estimated_price=tk.Label(my_frame2,text="Estimated Price:")
Estimated_price.place(x=250,y=250,anchor='center')




window.mainloop()
