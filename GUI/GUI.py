import tkinter as tk
from tkinter import ttk
from tkinter import messagebox,filedialog
from formScanner import *

import os

mainApplication=tk.Tk()
mainApplication.title('Form Scanner')
mainApplication.geometry('480x360')

mainMenu=tk.Menu()

file=tk.Menu(mainMenu,tearoff=False)

queryUrl=''
dataUrl=''

openIcon1 = tk.PhotoImage(file='icons/open.png')
openIcon2=tk.PhotoImage(file='icons/open.png')

def openQueryFile(event=None):
    global queryUrl
    queryUrl=filedialog.askopenfilename(initialdir=os.getcwd(),title='Select File',filetypes=(('Image file','*.png'),('All files','*.*')))
    print(queryUrl)

def openDataFile(event=None):
    global dataUrl
    dataUrl=filedialog.askopenfilename(initialdir=os.getcwd(),title='Select File',filetypes=(('Image file','*.png'),('All files','*.*')))
    print(os.path.dirname(dataUrl))
    dataUrl=os.path.dirname(dataUrl)

def scanForm():
    global queryUrl,dataUrl
    if len(queryUrl) and len(dataUrl):
        OBJ=Scan(queryUrl,dataUrl)
        OBJ.scanForm()
        queryUrl=''
        dataUrl=''
    else:
        messagebox.showwarning(title='warning',message='first select query image and data image')


file.add_command(label='Open Query File',image=openIcon1,compound=tk.LEFT,accelerator='Ctrl+Q',command=openQueryFile)
file.add_command(label='Open data File',image=openIcon2,compound=tk.LEFT,accelerator='Ctrl+D',command=openDataFile)

buttonScan=tk.Button(mainApplication,text='start scan',padx=30,pady=20,command=lambda:scanForm())
buttonScan.grid(row=1,column=2)

mainMenu.add_cascade(label='File',menu=file)
mainApplication.bind("<Control-q>",openQueryFile)
mainApplication.bind("<Control-d>",openQueryFile)
mainApplication.config(menu=mainMenu)


mainApplication.mainloop()