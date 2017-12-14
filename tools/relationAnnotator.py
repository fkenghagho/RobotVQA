import numpy as np
import Tkinter as t
from Tkinter import *
import tkMessageBox

class ObjectRelationAnnotator(object):
	def __init__(self,datasetGenerator):
		
		#globale Variablen##########################################
		self.datasetGenerator=datasetGenerator
		#window construction
		self.topWindow = Tk()
		# Add a grid
		self.tkvarF=StringVar()
		self.tkvarF.set('Objects Relationships')
		self.labelF=Label(textvariable=self.tkvarF)
		self.firstWindow = LabelFrame(self.topWindow,labelwidget=self.labelF)
		self.firstWindow.grid(column=0,row=0, sticky=(N,W,E,S) )
		self.firstWindow.columnconfigure(0, weight = 1)
		self.firstWindow.rowconfigure(0, weight = 1)
		self.firstWindow.pack(pady = 100, padx = 100)
	
	
		#Einloggen
		self.createLogin()
		#createMenu()
		
		self.topWindow.mainloop()	
				

	def createLogin(self):
		self.topWindow.title("Relationships-Annotator")
		# Create a Tkinter variable
		self.tkvarF.set('Objects-Relationships')
		
		
		# Dictionary for relation with options
		self.relationTkvar = StringVar(self.topWindow)
		self.relationChoices = { 'left','right','front','back','over','under','on','in','has'}
		self.relationTkvar.set('left') # set the default option
		
		# Dictionary for obejct 1 with options
		self.objectTkvar1 = StringVar(self.topWindow)
		self.objectChoices1 = {''}
		self.objectTkvar1.set('') # set the default option
		
		# Dictionary for obejct 2 with options
		self.objectTkvar2 = StringVar(self.topWindow)
		self.objectChoices2 = {''}
		self.objectTkvar2.set('') # set the default option
		
		#label Relation
		self.relationLabel=Label(self.firstWindow, text="Relation", padx=10, pady=10)
		self.relationLabel.grid(row = 1, column = 1,sticky=t.W)
		
		#pop menu for Relation
		self.relationMenu = OptionMenu(self.firstWindow, self.relationTkvar, *self.relationChoices)
		self.relationMenu.grid(row = 1, column =4,sticky=t.E)
		
		#label  object1
		self.objectLabel1=Label(self.firstWindow, text="Object1", padx=10, pady=10)
		self.objectLabel1.grid(row = 2, column = 1,sticky=t.W)
	
		#label object2
		self.objectLabel2=Label(self.firstWindow, text="Object2", padx=10)
		self.objectLabel2.grid(row = 3, column = 1,sticky=t.W)
		
		#pop menu for object1
		self.objectMenu1 = OptionMenu(self.firstWindow, self.objectTkvar1, *self.objectChoices1)
		self.objectMenu1.grid(row = 2, column =4,sticky=t.E)
		
		
		
		#pop menu for object2
		self.objectMenu2 = OptionMenu(self.firstWindow, self.objectTkvar2, *self.objectChoices2)
		self.objectMenu2.grid(row = 3, column =4,sticky=t.E)
		
		#button save
		self.saveButton=Button(self.firstWindow,text='Save', justify=RIGHT,relief=FLAT,overrelief=RAISED,activebackground='#6f6f6f',fg='#0000ff',highlightcolor='#6f6f6f',command=self.save,cursor='hand1')
		self.saveButton.grid(row = 4, column = 4,pady=30,sticky=t.E)
	
		
		#button cancel
		self.cancelButton=Button(self.firstWindow,text='Cancel', justify=RIGHT,relief=FLAT,overrelief=RAISED,activebackground='#6f6f6f',fg='#ff0000',highlightcolor='#6f6f6f',command=self.cancel,cursor='hand1')
		self.cancelButton.grid(row = 4, column = 3,pady=30,sticky=t.E)
		
		#button add
		self.addButton=Button(self.firstWindow,text='Add', justify=RIGHT,relief=FLAT,overrelief=RAISED,activebackground='#6f6f6f',fg='#00820f',highlightcolor='#6f6f6f',command=self.add,cursor='hand1')
		self.addButton.grid(row = 4, column = 2,pady=30,sticky=t.W)
		
		#button show
		self.showButton=Button(self.firstWindow,text='Show', justify=RIGHT,relief=FLAT,overrelief=RAISED,activebackground='#6f6f6f',highlightcolor='#6f6f6f',command=self.show,cursor='hand1')
		self.showButton.grid(row = 4, column = 1,pady=30,sticky=t.W)
	
	
	def save(self):
		if(self.tkvar.get()=='Verwalter'):
			pass
			
	
	def add(self):
		self.objectMenu1['menu'].add_command(label="Home")
		self.objectMenu1['menu'].add_command(label="Hello")
		self.objectTkvar1.set('Hello')
	
	def cancel(self):
		self.objectMenu1['menu'].delete(0,'end')
		self.objectTkvar1.set('')
	def show(self):
		if(self.tkvar.get()=='Verwalter'):
			pass
