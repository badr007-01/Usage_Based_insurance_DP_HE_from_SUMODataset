#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os


from models.Update import Cloud_WithoutDP_HE, Vehicle_withoutDP_HE
import pandas as pd
import time
from utils.functions import plot_accuracy_loss, plot_accuracy_loss_stem, get_file_paths,choose_random_files





if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True)

    train_loss, train_accuracy = [], []
    list_acc, list_loss = [], []
    
    vehicles_dataset=[]
    data = {
        'VehicleID': [],
        'Label': [],
        'Accuracy': [],
        'Loss': [],
        'epoch' : []
    }

    # Création de la DataFrame
    Acc_loss_train_epochs = pd.DataFrame(data)
 

    start_time = time.time()
    ##### List of features
    features = ['Label', 'Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]

    dict_users = {}

    dataset_3000V_Warning = './Dataset_Driver_Behavior/Sumo_dataset_V1/output_SUMO_Simulator/Output_3000V_DS_worning.csv'

    
    ##### the build the cloud
    ##### Creating cloud instance
    cloud_epochs = 10
    Cloud_local_epochs = 6
    cloud = Cloud_WithoutDP_HE( SUMO_data = dataset_3000V_Warning ,Cloud = "Cloud 1", Cloud_local_epochs = Cloud_local_epochs)
    
    

 
    ##### generate dataset of accurancy and loss befre the training methos to plot it 
    file_paths1 = get_file_paths()
    # Acc_loss_test1 = cloud.create_Acc_df(file_paths1)
    # print(Acc_loss_test1)
    # Acc_loss_test1.to_csv("./Dataset_Driver_Behavior/Sumo_dataset_V1/Output_fedDPDH/Test1_Acc_Loss_WithoutDPHE.csv")

    ##### plot
    # average_warnings = Acc_loss_test1.groupby(['Label']).agg({'Accuracy': 'mean','Loss' : 'mean'}).reset_index()
    # plot_accuracy_loss(average_warnings)
    



    
    ##### Beging the training process  
    for epoch in tqdm(range(cloud_epochs)):

        #####  Generate the random MLP for the  participant vehicles   
        vehicles_dataset = choose_random_files(folder_path="./Dataset_Driver_Behavior/Sumo_dataset_V1/Vehicles2/", Slow=1, Normal=1, Dang=1)

        vehicles = []

        ##### Generate participant vehicles
        for i in range(0, len(vehicles_dataset)):
            # h0= time.time()
            vehicle = Vehicle_withoutDP_HE( vehicle_id = i+1,local_data_path=vehicles_dataset[i], vehicle_participate = len(vehicles_dataset))
            vehicles.append(vehicle)
            # h1= time.time()
            # print('- >>>>> Build the vehicle {} in Time        {:f}ms '.format(vehicles_dataset[i], (h1 - h0)*1000))

        
        Ctx_local_weights, local_losses, local_shapes = [], [], []
        for vehicle in vehicles:
            GLobal_model_O  = cloud.get_Global_model_parameters()
            w, loss = vehicle.train_local_model( epochs = 1,  global_round = epoch , GLobal_model_O =  GLobal_model_O )
            Ctx_local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        loss_avg = sum(local_losses) / len(local_losses)
        loss_avg = copy.deepcopy(loss_avg)
        train_loss.append(loss_avg)
        h44 = time.time()


        ##### Updating global model with local model parameters
        
        global_weight = cloud.update_global_model(Ctx_local_weights)
        # global_weight = cloud.update_Ctx_model(Ctx_local_weights, local_shapes)
        h55= time.time()
        print('- ...................... Updating  global model Time with local model parameters      {:f}s '.format(h55-h44))

        ######Collect the accurancy and loss of each epochs from the prediction on using the dataset of participant vehicle.
        Acc_loss_train = cloud.create_Acc_df(vehicles_dataset, epoch)
        print(Acc_loss_train)
        Acc_loss_train_epochs = pd.concat([Acc_loss_train_epochs, Acc_loss_train], ignore_index=True)
        Acc_loss_train_epochs.to_csv("./Dataset_Driver_Behavior/Sumo_dataset_V1/Output_fedDPDH/Train_Acc_Loss_WithoutDPHE.csv")
    
        

        ###### Updating local model with average model parameters
        for batch_idx, (vehicle) in enumerate(vehicles):
            h6 = time.time()
            vehicle.update_global_model(global_weight)
            h7=time.time()
            print('- >>>>> Updating Time local model with average model parameters for the vehicle {} is      {:f}ms '.format(batch_idx,(h7-h6)*1000))
            
        ###### if the average of the accurancy is less then 90 we stop
        if (Acc_loss_train["Accuracy"].mean() > 90):
            print("The Global epoch stop at >>>>>>>>>>>>>>>>>>>>>>> ", epoch)
            break
      

      
    ###then plot the Acc_loss_train
    # print(Acc_loss_train_epochs)
    # plot_accuracy_loss_stem(Acc_loss_train)
    


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


   
    
    
    #####  # Evaluating global model on a test dataset
    file_paths = get_file_paths()
    # print(file_paths)
    folder_path = "./Dataset_Driver_Behavior/Sumo_dataset_V1/Vehicles2/"
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    h8 = time.time()
    Acc_loss_test2 = cloud.create_Acc_df(file_paths )
    print(Acc_loss_test2)
    Acc_loss_test2.to_csv("./Dataset_Driver_Behavior/Sumo_dataset_V1/Output_fedDPDH/Test2_Acc_Loss_WithoutDPHE.csv")
    
    h9=time.time()
    print('-.................. Cloud predictions of {} Time is      {:f}s '.format(len(file_paths), h9-h8))

    dataset4 = './Dataset_Driver_Behavior/Sumo_dataset_V1/output_SUMO_Simulator/Output_3000V_DS_worning.csv'
    test_acc, loss = cloud.inference(test_data = pd.read_csv(dataset4))
    print(f' \n Results after {1} global rounds of training:')
    print("|---- Cloud Accuracy: {:f}%".format(test_acc))

    # print(f' \n Results after {1} global rounds of training:')
    # print("|---- Cloud Accuracy: {:f}%".format(100*test_acc))
    





    




    

   




  

    

