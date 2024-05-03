
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os






def plot_accuracy_loss(df):
    # Sorting DataFrame by index
    df_sorted = df.sort_index()
    print(df_sorted)

    # Plotting
    plt.figure(figsize=(10, 6))
    for label in df['Label'].unique():
        label_data = df_sorted[df_sorted['Label'] == label]
        plt.plot(label_data.index, label_data['Accuracy'], label=label + ' Accuracy')
        plt.plot(label_data.index, label_data['Loss'], linestyle= "--", label=label + ' Loss')
    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Accuracy and Loss vs epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy_loss_stem(df):
    # Sorting DataFrame by VehicleID
    df_sorted = df.sort_values(by='VehicleID')

    # Plotting
    plt.figure(figsize=(12, 8))
    for label in df['Label'].unique():
        label_data = df_sorted[df_sorted['Label'] == label]
        for i, (_, row) in enumerate(label_data.iterrows()):
            plt.stem([i], [row['Loss']], linefmt='-', markerfmt='o', basefmt=" ", label=row['VehicleID'] + ' ' + label)

    plt.xlabel('VehicleID')
    plt.ylabel('Value')
    plt.title('Accuracy and Loss for Each Vehicle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def get_file_paths():
    folder_path = "./Dataset_Driver_Behavior/Sumo_dataset_V1/Vehicles2/"
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    return file_paths




def choose_random_files(folder_path="./Vehicles2", Slow=6, Normal=10, Dang=10):
    selected_files = []
    if Slow > 0:
        slow_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("Trajectory_Slow")]
        selected_files.extend(random.sample(slow_files, min(Slow, len(slow_files))))


    if Normal > 0:
        normal_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("Trajectory_Normal")]
        selected_files.extend(random.sample(normal_files, min(Normal, len(normal_files))))

    if Dang > 0:
        dang_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("Trajectory_Dangerous")]
        selected_files.extend(random.sample(dang_files, min(Dang, len(dang_files))))


    return selected_files

