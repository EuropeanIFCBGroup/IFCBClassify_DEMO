import torch
from torchvision.transforms import v2 as transforms
from torchvision import datasets
from sklearn.model_selection import train_test_split
from utils.CustomTransforms import SquarePad, FullPad, ReflectPad

class DatasetFactory():

    @staticmethod
    def get_dataset(name, data_training_directory, test_dataset_split_size,image_input_width,image_input_height,training=True):
        #run the Dataset Normalisation to find out what these values should be set to for each dataset
        mean = 0.7019
        std = 0.1496  
        
        transform = None
        if name == "dataset":
            
            transform = transforms.Compose([
                #resizes and transforms image data to tensor for CNN input
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((image_input_width,image_input_height),antialias=True), 
            ])

        elif name == "dataset_normalised":    
            #run the Dataset Normalisation to find out what these values should be set to for each dataset
            transform = transforms.Compose([
    
                #resizes and transforms image data to tensor for CNN input
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((image_input_width,image_input_height),antialias=True),
                
                #normalise the image data
                transforms.Normalize(mean=[mean], std=[std]),
            ])

        elif name == "dataset_fullpad":
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),

                #apply full padding to image data
                FullPad(image_input_width, image_input_height),
                
                #normalise the image data
                transforms.Resize((image_input_width,image_input_height),antialias=True),

            ])
            
        elif name == "dataset_squarepad":
            transform = transforms.Compose([
                
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
            
                #apply square padding to image data
                SquarePad(),
                transforms.Resize((image_input_width,image_input_height),antialias=True),
                
                
            ])  
            
        elif name == "dataset_squarepad_normalised":
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),

                #apply square padding to image data
                SquarePad(),
                
                transforms.Resize((image_input_width,image_input_height),antialias=True),

                #normalise the image data
                transforms.Normalize(mean=[mean], std=[std]),
                
            ])        
        
        elif name == "dataset_fullpad_normalised":
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),

                #apply full padding to image data
                FullPad(image_input_width, image_input_height),
                
                transforms.Resize((image_input_width,image_input_height),antialias=True),
                
                #normalise the image data
                transforms.Normalize(mean=[mean], std=[std]),
                
            ])       

        elif name == "dataset_reflectpad":
            transform = transforms.Compose([
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),

                ReflectPad(),
                
                transforms.Resize((image_input_width,image_input_height),antialias=True),

            ])  
    
        dataset = datasets.ImageFolder(data_training_directory, transform=transform)

        
        if training == True:
            #once data has been loaded split into training and validation data
            train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_dataset_split_size,random_state=42)
            #create a dict which contains both train and master datasets
            dataset_master = {}
            dataset_master['train'] = torch.utils.data.Subset(dataset, train_idx)
            dataset_master['test'] =  torch.utils.data.Subset(dataset, test_idx)
            return dataset_master
        else:
            return transform
    
    

        