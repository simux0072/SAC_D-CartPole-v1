import csv
import pandas as pd

class Data():
    def __init__(self):
        self.svg_path = "./database"
        self.svg_file_name = "CartPole-v1"
        self.fields = ["Gen", "Score", "Running Score", "Actor Loss", "Running Actor Loss", "Critic 1 Loss", "Running Critic 1 Loss", "Critic 2 Loss", "Running Critic 2 Loss", "Alpha Loss", "Running Alpha Loss", "Alpha"]
        self.info = {}
        
        self.create_fields()
    
    def create_fields(self):
        with open(f'./{self.svg_path}/{self.svg_file_name}.csv', 'w', newline='') as csv_file:
            csv_writter = csv.DictWriter(csv_file, fieldnames=self.fields)
            csv_writter.writeheader()
            csv_file.close()    
    
    def update_data(self, data):
        with open(f'./{self.svg_path}/{self.svg_file_name}.csv', 'a', newline='') as csv_file:
            csv_writter = csv.DictWriter(csv_file, fieldnames=self.fields)
            
            for index in range(len(self.fields)):
                self.info[self.fields[index]] = data[index]
            
            csv_writter.writerow(self.info)