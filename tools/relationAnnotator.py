"""
#@Author:   Frankln Kenghagho
#@Date:     04.04.2019
#@Project:  RobotVA
"""


import numpy as np
import Tkinter as t
from Tkinter import *
import tkMessageBox
from generateDataset import Dataset
from tkFileDialog import askopenfilename
import os
import json

class ObjectRelationAnnotator(object):
	def __init__(self,datasetGenerator):

		"""Spatial Relation Annotator
		"""
		
		#globale Variablen##########################################
		self.listRelation=[]
		self.index=13
		self.step=1
		self.annotFile=""
		self.imageFile=""
		self.annotationFile=""
		self.FILEOPENOPTIONS1 = dict(defaultextension='.jpg',filetypes=[('All files','*.*'), ('Image file','*litImage*.jpg')])
		self.FILEOPENOPTIONS2 = dict(defaultextension='.json',filetypes=[('All files','*.*'), ('Image file','*.json')])
		
		if datasetGenerator!=None:
			self.datasetGenerator=datasetGenerator
		else:
			self.datasetGenerator=Dataset()
		self.outputImage=self.datasetGenerator.folder+'ModifieD'+'/'+'modified.'+self.datasetGenerator.extension
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
		self.relationChoices = { 'left','front','under','on','in','has','valign'}
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
		if( self.annotationFile==""):
			tkMessageBox.showerror("Relationships-Save","No relationship to save!!!")
		else:
			#self.annotFile= askopenfilename(initialdir = self.datasetGenerator.folder,**self.FILEOPENOPTIONS2) # show an "Open" dialog box and return the path to the selected file
			#if not os.path.isfile(self.annotFile):
			#	tkMessageBox.showerror("Relationships-Save","Invalid selected output file!!!")
			#else:
				with open(self.annotationFile,"r") as f:
					annot=json.load(f)
				f.close()
				#empty list
				"""uncomment the statement below if you want to ignore any existing relational map
				"""
				#del annot['objectRelationship'][:]
				for elt in self.listRelation:
					annot['objectRelationship'].append(json.loads('{"object1":"'+elt[0]+'","relation":"'+elt[1]+'","object2":"'+elt[2]+'"}'))
				with open(self.annotationFile,"w") as f:
					json.dump(annot,f)
				f.close()
				tkMessageBox.showinfo("Relationships-Save","Relationships saved successfully!!!")
				del self.listRelation[:]
				self.cancel()#reset
				self.add()#next image
	def add(self):
		if self.imageFile=="":
			#tkMessageBox.showinfo("Input Image-Selection","First choose an input image for annotation!!!")
			#self.imageFile=askopenfilename(initialdir = self.datasetGenerator.folder,**self.FILEOPENOPTIONS1 ) # show an "Open" dialog box and return the path to the selected file
			self.index=self.index+self.step
			self.imageFile=self.datasetGenerator.folder+'/'+self.datasetGenerator.litImage+str(self.index)+'.'+self.datasetGenerator.extension
			print(self.imageFile)
			if not os.path.isfile(self.imageFile): 
				tkMessageBox.showerror("Relationships-Creation","Wrong input Image!!!")
			else:
				filename, file_extension = os.path.splitext(self.imageFile)
				if file_extension not in ['.'+self.datasetGenerator.extension] :
						tkMessageBox.showerror("Relationships-Creation","Wrong input Image type: "+str( file_extension)+"!!!")
				else:
					self.annotationFile=self.datasetGenerator.folder+'/'+self.datasetGenerator.annotation+filename.split(self.datasetGenerator.litImage).pop()+'.'+self.datasetGenerator.annotExtension
					item,tetax=self.datasetGenerator.ImageParser(self.annotationFile,self.imageFile,
					self.outputImage,0.5,mode="indirect")
					self.objectTkvar1.set('')
					self.objectTkvar2.set('')
					for i in range(len(item)):
						obj=str(i)+':'+item[i]
						self.objectMenu1['menu'].add_command(label=obj,command=t._setit(self.objectTkvar1, obj))
						self.objectMenu2['menu'].add_command(label=obj,command=t._setit(self.objectTkvar2, obj))
						self.objectTkvar1.set(obj)
						self.objectTkvar2.set(obj)
					self.topWindow.title(self.imageFile+' *** '+tetax)
					tkMessageBox.showinfo("Relationships-Creation","Image loaded. You can now build a relation!!!")
		else:
			if(len(self.objectTkvar1.get())>0 and len(self.objectTkvar2.get())>0):
				rel=self.relationTkvar.get()
				obj1=self.objectTkvar1.get().split(':')[1]
				obj2=self.objectTkvar2.get().split(':')[1]
				self.listRelation.append([obj1,rel,obj2])
				tkMessageBox.showinfo("Relationships-Creation","Relationship created!!!")
			else:
				tkMessageBox.showerror("Relationships-Creation","Wrong input objects!!!")
				
	
	def cancel(self):
		#resset index
		self.annotationFile=""
		#clear menus
		self.objectMenu1['menu'].delete(0,'end')
		self.objectTkvar1.set('')
		self.objectMenu2['menu'].delete(0,'end')
		self.objectTkvar2.set('')
		#reset main variables
		self.annotFile=""
		self.imageFile=""
		del self.listRelation[:]
		if os.path.isfile(self.outputImage):
			os.unlink(self.outputImage)
		self.topWindow.title("Relationships-Annotator")
		tkMessageBox.showinfo("Relationships-Deletion","System reset!!!")
	def show(self):
		if len(self.listRelation)<=0:
			tkMessageBox.showinfo("Relationships-Preview","No relationship has been established yet!!!")
		else:
			message=''
			for elt in self.listRelation:
				message=message+'{'+elt[0]+', '+elt[1]+', '+elt[2]+'}\n'
			tkMessageBox.showinfo("Relationships-Preview",message)
