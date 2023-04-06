import tkinter as tk
import customtkinter as ctk

from PIL import Image
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#Create the app
app = tk.Tk()
app.geometry("555x700")
app.title('PhotoGenerator From Text')
ctk.set_appearance_mode("dark")

#Create space for users to enter text
prompt = ctk.CTkEntry(master= app,height = 45,width = 520)
prompt.place(x = 10,y = 10)

lmain = ctk.CTkLabel(master= app,height = 512,width = 512)
lmain.place(x = 10,y = 110)

modelid = "CompVis/stable-diffusion-v1-4"
device =  "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", use_auth_token=auth_token) 
pipe.to(device) 

def generate():
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(master= app,height=40, width=120,command = generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()