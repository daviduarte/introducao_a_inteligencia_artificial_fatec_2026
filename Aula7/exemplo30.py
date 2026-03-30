"""
Guia para treinar a YOLO em seu próprio conjunto de dados (dataset): 
https://docs.ultralytics.com/tasks/detect/#faq
"""
from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="coco8.yaml", 
                      epochs=100, 
                      imgsz=640,
                      project="/media/davi/ssd-dados1/Davi/Aulas/Disciplinas/FATEC/Introduçao a IA/IA-2026/Aula7/runs", 
                      name="experiment1")

