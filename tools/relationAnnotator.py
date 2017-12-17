import numpy as np
import Tkinter as t
from Tkinter import *
import tkMessageBox
from tkFileDialog import askopenfilename

class ObjectRelationAnnotator(object):
	def __init__(self,datasetGenerator):
		
		#globale Variablen##########################################
		self.listRelation=[]
		self.annotFile=""
		self.imageFile=""
		self.outputImage="D:/dataset1/annotImage.jpg"
		if datasetGenerator!=None:
			self.datasetGenerator=datasetGenerator
		else:
			self.datasetGenerator=Dataset('D:/dataset1',1,0,mode='offline')
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
		filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
		print(filename)
					
	
	def add(self):
		
		self.objectMenu1['menu'].add_command(label="Hello",command=t._setit(self.objectTkvar1, "Hello"))
		self.objectTkvar1.set('Hello')
		if self.imageFile=="":
			tkMessageBox.showinfo("Input Image-Selection","First choose an input image for annotation!!!")
			self.imageFile=askopenfilename() # show an "Open" dialog box and return the path to the selected file
			print(self.imageFile)
			if not os.path.isfile(self.imageFile): 
				tkMessageBox.showerror("Relationships-Creation","Wrong input Image!!!")
			else:
				filename, file_extension = os.path.splitext(self.imageFile)
				if file_extension not in ['.JPG','.jpg','.png','.png','.bmp','.BMP'] :
						tkMessageBox.showerror("Relationships-Creation","Wrong input Image type: "+str( file_extension)+"!!!")
				else:
					item=self.datasetGenerator.ImageParser(filename+'.json',self.imageFile,
					self.outputImage,0.5,mode="indirect")
					self.objectTkvar1.set('')
					self.objectTkvar2.set('')
					for i in range(len(item)):
						obj=str(i)+':'+item[i]
						self.objectMenu1['menu'].add_command(label=obj,command=t._setit(self.objectTkvar1, obj))
						self.objectMenu2['menu'].add_command(label=obj,command=t._setit(self.objectTkvar2, obj))
						self.objectTkvar1.set(obj)
						self.objectTkvar2.set(obj)
					tkMessageBox.showinfo("Relationships-Creation","Image loaded. You can now build a relation!!!")
		else:
			if(len(self.objectTkvar1.get())>0 and len(self.objectTkvar2.get())>0):
				rel=self.relationTkvar.get()
				obj1=self.objectTkvar1.get().split(':')[1]
				obj2=self.objectTkvar2.get().split(':')[1]
				self.listRelation.append([obj1,rel,obj2])
			else:
				tkMessageBox.showerror("Relationships-Creation","Wrong input objects!!!")
				
	
	def cancel(self):
		#clear menus
		self.objectMenu1['menu'].delete(0,'end')
		self.objectTkvar1.set('')
		self.objectMenu2['menu'].delete(0,'end')
		self.objectTkvar2.set('')
		#reset main variables
		self.annotFile=""
		self.imageFile=""
		del self.listRelation[:]
		tkMessageBox.showinfo("Relationships-Deletion","System reset!!!")
	def show(self):
		tkMessageBox.showinfo("Relationships-Preview","No relationship has been established yet!!!")
