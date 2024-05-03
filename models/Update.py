#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from models.Nets import MLP
from models.Fed import FedAvg, FedAvg_encrypted
import torch.optim as optim
import time

from collections import OrderedDict
from typing import Dict, List, Optional
import tenseal as ts
from tenseal.enc_context import SecretKey
# from tenseal.tensors.bfvvector import BFVVector
from tenseal.tensors.ckksvector import CKKSVector
import re
from opacus import PrivacyEngine








class Vehicle_withoutDP_HE(object):
    def __init__(self, local_data_path=None, vehicle_id ="", vehicle_participate = 2 ):
        self.vehicle_id = vehicle_id
        # print("the vehicle >>>>>>>>>>>>>>>>>>>>>>>>STaaaaaaart>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",vehicle_id )
        
        h0= time.time()
        self.lr = 0.00001
    
        self.local_data = pd.read_csv(local_data_path)
        features = ['Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]
       
        self.model = MLP(dim_in=len(features), dim_out=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        self.criterion = nn.MSELoss() 
        
        h1= time.time()
        print('- >>>>> Build the vehicle {} in Time        {:f}ms '.format(vehicle_id, (h1 - h0)*1000))


        

    


        #labels = ['Slow', 'Normal', 'Dangerous']
        X = self.local_data[features]
        y = self.local_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)

        X_train, X_val, y_train, y_val = sk.train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.to_numpy().astype('float32')
        y_train = y_train.to_numpy().astype('long')
        X_val = X_val.to_numpy().astype('float32')
        y_val = y_val.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        self.train_loader = DataLoader(dataset=TensorDataset(X_train, y_train),
                                  batch_size=64, shuffle=True)
        self.X_test = DataLoader(dataset=TensorDataset(X_val, y_val),
                                batch_size=64, shuffle=True)
       

    def extract_shapes(self, model):
        shapes = {}
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            shapes[key] = value.size()
        return shapes


    def get_local_model_parameters(self):
        return self.model.state_dict()
    


    def train_local_model(self, epochs= 1, global_round=10, GLobal_model_O = None ):
        epoch_loss = []
        self.epochs = epochs
        self.model.load_state_dict(GLobal_model_O)

        self.model.train()
        t1=time.time()

        for epoch in range(self.epochs):

            h22=time.time()
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):# for inputs, labels in train_loader:
                inputs, labels = inputs , labels 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(">>>>>>>>>>>>>>>>>>>>>>input ",inputs.size())
                # print(">>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>>>>>outputs ",outputs.size() )
                loss = self.criterion(outputs.float(), labels.view(-1, 1).float())
                loss.backward()

                self.optimizer.step()

                if (batch_idx % 3000 == 0):
                    print('| Global Round : {} | Local Epoch : {} | Vehicle ID : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( global_round, epoch, self.vehicle_id , batch_idx * len(inputs), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            h33=time.time()
            print('- >>>>> Training time for the vehicle {} in the epochs {} is {:f}s '.format(self.vehicle_id ,epoch ,(h33-h22)))

    

        t4 = time.time()

        print(' ---------------- The Total time taking for train the model of vehicle {} without noise and without encryption : {:f}s | number of eachos = {}'.format(self.vehicle_id, (t4-t1), epochs))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def local_inference(self):
        """ Returns the inference accuracy and loss.
        """
        h4 = time.time()  

        epoch_loss = []
            
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (inputs, labels) in enumerate(self.X_test):
            inputs, labels = inputs , labels 

            # Inference
            outputs = self.model(inputs)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1)).item()
            epoch_loss.append(loss)
           

            # Prediction

            pred_labels = ((outputs > 0.7).float() + (outputs > 1.66).float())
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)
            pred_labels = copy.deepcopy(pred_labels).view(-1)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()  ##correct += (pred_labels == labels.view(-1, 1)).sum().item() #correct += torch.sum(torch.eq(pred_labels, labels)).item()  #
            total += len(labels)
    

        h5=time.time()
        print('\n ---------------- The time taking for the local prediction of the vehicle {} is : {:f}s'.format(self.vehicle_id ,h5-h4))

        accuracy = (correct/total)*100

        
        
        return accuracy, sum(epoch_loss) / len(epoch_loss)
    

    def get_local_model_parameters(self):
        return self.model.state_dict()


     
    def update_global_model(self, cloud_model_parameters):
        t0 = time.time()
        self.model.load_state_dict(cloud_model_parameters)
        self.model 
        t1= time.time()
        # print('\n ---------------- The time taking for the local training of the vehicle {} is : {:f}ms '.format(self.vehicle_id ,(t1-t0)*1000))


    




    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)



class Cloud_WithoutDP_HE(object):
    def __init__(self, SUMO_data = "", Cloud ="", Cloud_local_epochs = 10):
        features = [ 'Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]
        
        h000= time.time()
        self.lr = 0.00001
        self.cloud_id = Cloud
        #smache bien elf.global_model = MLP(input_size=len(features) - 1, hidden_size=64, output_size=1)
        self.global_model = MLP(dim_in=len(features), dim_out=1 )
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vehicle_models = []  # List to store models from each vehicle
        h111= time.time()
        print('- >>>>>++++++++++++++++++++++ The Set up Time is       {:f}ms '.format( (h111 - h000)*1000))


        #Key generation
        # Setup TenSEAL context
        # self.Context_ckks()
        # self.secret_key = ( self.context.secret_key())  # save the secret key before making context public
        # self.public_key = self.context.make_context_public()

        self.SUMO_data = pd.read_csv(SUMO_data)
        X = self.SUMO_data[features]
        y = self.SUMO_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)


        X = X.to_numpy().astype('float32')
        y = y.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)


        self.train_loader = DataLoader(dataset=TensorDataset(X, y), batch_size=200, shuffle=True)
        
        # self.train_Global_model (Cloud_local_epochs)



    def get_context(self):
       return self.context

    def update_global_model(self, local_weights):
        self.global_model.train()
        # copy weights
        global_weights = self.global_model.state_dict()


        # update global weights
        global_weights = FedAvg(local_weights)
        # update global weights
        self.global_model.load_state_dict(global_weights)
        self.global_model 
        return self.global_model.state_dict()



    def update_global_model(self, local_weights):
      self.global_model.train()
      # copy weights
      global_weights = self.global_model.state_dict()


      # update global weights
      global_weights = FedAvg(local_weights)
      # update global weights
      self.global_model.load_state_dict(global_weights)
      self.global_model 
      return self.global_model.state_dict()
    


    
    def train_Global_model(self, Cloud_local_epochs=10):
            epoch_loss = []
            self.global_model.train()
            t1=time.time()

            for epoch in range(Cloud_local_epochs):

                h22=time.time()
                batch_loss = []
                for batch_idx, (inputs, labels) in enumerate(self.train_loader):# for inputs, labels in train_loader:
                    inputs, labels = inputs , labels 
                    self.optimizer.zero_grad()
                    outputs = self.global_model(inputs)
                    # print(">>>>>>>>>>>>>>>>>>>>>>input ",inputs.size())
                    # print(">>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>>>>>outputs ",outputs.size() )
                    loss = self.criterion(outputs.float(), labels.view(-1, 1).float())
                    loss.backward()

                    self.optimizer.step()

                    if (batch_idx % 3000 == 0):
                        print('| Global Round : {} | Cloud ID : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, self.cloud_id , batch_idx * len(inputs), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))

                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                h33=time.time()
                print('- >>>>> Training time for the vehicle {} in the epochs {} is {:f}s '.format(self.cloud_id ,epoch ,(h33-h22)))

            t4 = time.time()
            print('\n ---------------- The time taking to train the local model without noise and without encryption is : {:f}s'.format( (t4-t1)))
            print(' ---------------- The Total time taking for train the model of vehicle {} : {:f}s | number of eachos = {}'.format(self.cloud_id, (t4-t1), Cloud_local_epochs))

            #return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
            # return self.model.state_dict() , shapes , sum(epoch_loss) / len(epoch_loss)

    def get_Global_model_parameters(self):
        return self.global_model.state_dict()


    def inference(self, test_data, epoch = 0):
        features = ['Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]

        X = test_data[features]
        y = test_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)


        X = X.to_numpy().astype('float32')
        y = y.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)


        testloader = DataLoader(dataset=TensorDataset(X, y), batch_size=128, shuffle=True)

        """ Returns the inference accuracy and loss.
        """
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        t0=time.time()
        epoch_loss = []

        


        for batch_idx, (values, labels) in enumerate(testloader):
            
            values, labels = values , labels 

            # Inference
            outputs = self.global_model(values)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1)).item()
            
            epoch_loss.append(loss)
            
            # Prediction

            pred_labels = ((outputs > 0.7).float() + (outputs > 1.66).float())
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)
            pred_labels = copy.deepcopy(pred_labels).view(-1)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()  ##correct += (pred_labels == labels.view(-1, 1)).sum().item() #correct += torch.sum(torch.eq(pred_labels, labels)).item()  #
            total += len(labels)
        


        accuracy = (correct/total)*100
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",accuracy,"<<<<<<<<<<<<<<<<<<<<<<<<<<", epoch)
        t1=time.time()
        # print("||||||||||||||||||||||||||||||||||||||||||",len(epoch_loss))
        print('\n ---------------- The time taking for the prediction of the global model : {:f}s '.format((t1-t0)))
        return accuracy, sum(epoch_loss) / len(epoch_loss)
    

    def create_Acc_df(self,files,epoch = 0):
        data = {
            'VehicleID': [],
            'Label': [],
            'Accuracy': [],
            'Loss': [],
            'epoch' : []
        }
        # Création de la DataFrame
        df = pd.DataFrame(data)
        
        for file in files: 
            test_data = pd.read_csv(file)
            # Expression régulière pour extraire les parties *
            pattern = re.compile(r'Trajectory_(.*).csv')

            # Extraire les parties * des noms de fichiers correspondants
            match = pattern.search(file)

            asterisk_part = match.group(1)
            vehicleID = asterisk_part
            label = vehicleID.split('_')[0]
            
            test_acc1, loss1 = self.inference(test_data, epoch )
            
            
            df = df._append({
                    'VehicleID': vehicleID,
                    'Label': label,
                    'Accuracy': test_acc1,
                    'Loss': loss1,
                    'epoch' : epoch,
                }, ignore_index=True)
        
        return df


class Vehicle_withDP_HE(object):
    def __init__(self, epochs = 1, local_data_path=None, vehicle_id ="", vehicle_participate = 2 ):
        
        
        # self.num_classes = num_classes
        self.local_data = pd.read_csv(local_data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = ['Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]
         #labels = ['Slow', 'Normal', 'Dangerous']
        X = self.local_data[features]
        y = self.local_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)

        X_train, X_val, y_train, y_val = sk.train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.to_numpy().astype('float32')
        y_train = y_train.to_numpy().astype('long')
        X_val = X_val.to_numpy().astype('float32')
        y_val = y_val.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        self.train_loader = DataLoader(dataset=TensorDataset(X_train, y_train),
                                  batch_size=64, shuffle=True)
        self.X_test = DataLoader(dataset=TensorDataset(X_val, y_val),
                                batch_size=64, shuffle=True)
        
        
        
        self.vehicle_id = vehicle_id
        
        print("the vehicle >>>>>>>>>>>>>>>>>>>>>>>>STaaaaaaart>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",vehicle_id )
        h000= time.time()
        self.lr = 0.0001
        
       
       
        self.model = MLP(dim_in=len(features), dim_out=1) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() 

        self.epochs = epochs

        



       
        
        # self.privacy_engine = PrivacyEngine(secure_mode=True) 
        # self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
        #     module= self.model,
        #     optimizer=self.optimizer,
        #     data_loader=self.train_loader,
        #     noise_multiplier=1.3, #sigma=1.3
        #     max_grad_norm=1.0,
        # )

        self.privacy_engine = PrivacyEngine(secure_mode=True) 
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module= self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon= 0.2,
            target_delta= 1e-5,
            epochs= 1,
            # noise_multiplier=1.3, #sigma=1.3
            max_grad_norm=1.0,
        )

        h111= time.time()
        print('- >>>>>++++++++++++++++++++++ The Set up Time is (Init model + init Clipping + init DP)       {:f}ms '.format( (h111 - h000)*1000))



        
       
    def encrypt_vehicle_weights_ckks(self, vehicle_weights: List[OrderedDict], context) -> list:
        # encr = []

        state_dict = vehicle_weights.state_dict() #vehicle_weights = model
        h000 = time.time()
        shapes = self.extract_shapes(vehicle_weights)
        h111=time.time()
        # print("the time tackinh to extract the shapes  in {:f}ms".format((h111-h000)*1000) )
        encr_state_dict = {}
        h00=time.time()
        for key, value in state_dict.items():
            val = value.flatten().cpu()
            # print("?????????????????????", value.shape)
            #print("???????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!??????????", value)
            h0=time.time()
            encr_state_dict[key] = ts.ckks_vector(context, val)
            h1=time.time()
            # print("the time taking encrypt the key number {} is in {:f}ms".format(key ,(h1-h0)*1000) )
            #print("?????????????????????", encr_state_dict[key])
        # encr.append(encr_state_dict)
        h11=time.time()
        # print("the time taking encrypt the all in {:f}ms".format((h11-h00)*1000) )
        return encr_state_dict, shapes 

    def extract_shapes(self, model):
        shapes = {}
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            shapes[key] = value.size()
        return shapes
    
    

    def get_local_model_parameters(self):
        return self.model.state_dict()
    


    def train_local_model(self, global_round=10, context = None, GLobal_model_O = None ):
        epoch_loss = []
        
        self.model.load_state_dict(GLobal_model_O)
       

        self.model.train()
        t1=time.time()

        for epoch in range(self.epochs):

            h22=time.time()
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):# for inputs, labels in train_loader:
                inputs, labels = inputs , labels 
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(">>>>>>>>>>>>>>>>>>>>>>input ",inputs.size())
                # print(">>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>>>>>outputs ",outputs.size() )
                loss = self.criterion(outputs.float(), labels.view(-1, 1).float())
                loss.backward()

                self.optimizer.step()

                if (batch_idx % 3000 == 0):
                    print('| Global Round : {} | Local Epoch : {} | Vehicle ID : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( global_round, epoch, self.vehicle_id , batch_idx * len(inputs), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            h33=time.time()
            print('- >>>>> Training time for the vehicle {} in the epochs {} is {:f}s '.format(self.vehicle_id ,epoch ,(h33-h22)))
        
        t2 = time.time()
        

        
        # # self.add_noise(self.model) # add the differential privacy noise
        # t3 = time.time()

        # print("self model is ???????????????????????????????????????  ", self.vehicle_id  , "  ??????????????????????????",self.model.state_dict())

        enrypted_local , shapes = self.encrypt_vehicle_weights_ckks(self.model,context)

        t4 = time.time()
        print('\n ---------------- The time taking to train the local model without noise and without encryption is : {:f}s'.format( (t2-t1)))
        # print(' ---------------- The time taking to add DP noise is : {:f}ms '.format((t3-t2)*1000))
        print(' ---------------- The time taking for encryption  is : {:f}ms '.format((t4-t2)*1000))
        print(' ---------------- The Total time taking for train the model of vehicle {} : {:f}s | number of eachos = {}'.format(self.vehicle_id, (t4-t1), self.epochs))

        
        # 'eps': 50,                      # privacy budget for each global communication
        #    'delta': 1e-5,                  # approximate differential privacy: (epsilon, delta)-DP
        Delta= 1e-05
        epsilon = self.privacy_engine.accountant.get_epsilon(delta= Delta)
        print(" the epsilone is ????????????????????: {:f}".format( epsilon ))
        print(" the spend is ????????????????????: ",  Delta )

        #return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        return enrypted_local , shapes , sum(epoch_loss) / len(epoch_loss)

    def local_inference(self):
        """ Returns the inference accuracy and loss.
        """
        h4 = time.time()  

        epoch_loss = []
            
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (inputs, labels) in enumerate(self.X_test):
            inputs, labels = inputs , labels 

            # Inference
            outputs = self.model(inputs)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1)).item()
            epoch_loss.append(loss)
           

            # Prediction

            # pred_labels = ((outputs > 0.7).float() + (outputs > 1.66).float())
            pred_labels = ((outputs > -0.33).float() + (outputs > 0.33).float()) - 1
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)
            pred_labels = copy.deepcopy(pred_labels).view(-1)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()  ##correct += (pred_labels == labels.view(-1, 1)).sum().item() #correct += torch.sum(torch.eq(pred_labels, labels)).item()  #
            total += len(labels)
    

        h5=time.time()
        print('\n ---------------- The time taking for the local prediction of the vehicle {} is : {:f}s'.format(self.vehicle_id ,h5-h4))

        accuracy = (correct/total)*100
     
                  
        # print("||||||||||||||||||||||||||||||||||||||||||",len(epoch_loss))
        
        
        return accuracy, sum(epoch_loss) / len(epoch_loss)
    

    def get_local_model_parameters(self):
        return self.model.state_dict()


     
    def update_global_model(self, cloud_model_parameters):
        t0 = time.time()
        self.model.load_state_dict(cloud_model_parameters)
        self.model 
        t1= time.time()
        # print('\n ---------------- The time taking for the local training of the vehicle {} is : {:f}ms '.format(self.vehicle_id ,(t1-t0)*1000))


    


class Cloud_DP_HE:
    def __init__(self, SUMO_data = "", Cloud ="", Cloud_local_epochs = 10):
        features = [ 'Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]
       
        self.SUMO_data = pd.read_csv(SUMO_data)
        X = self.SUMO_data[features]
        y = self.SUMO_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)


        X = X.to_numpy().astype('float32')
        y = y.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)


        self.train_loader = DataLoader(dataset=TensorDataset(X, y), batch_size=200, shuffle=True)
        
        
        h000 = time.time()
        self.cloud_id = Cloud
        #smache bien elf.global_model = MLP(input_size=len(features) - 1, hidden_size=64, output_size=1)
        self.global_model = MLP(dim_in=len(features), dim_out=1 )
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=0.00001) #0.000001
        self.criterion = nn.MSELoss()
        

        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vehicle_models = []  # List to store models from each vehicle
        h222 = time.time()

        #Key generation
        # Setup TenSEAL context
        self.Context_ckks()
        self.secret_key = ( self.context.secret_key())  # save the secret key before making context public
        self.public_key = self.context.make_context_public()
        h111= time.time()
        
        

        

        # self.privacy_engine = PrivacyEngine(secure_mode=True) 
        # self.global_model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
        #     module= self.global_model,
        #     optimizer=self.optimizer,
        #     data_loader=self.train_loader,
        #     noise_multiplier=1.3, #sigma=1.3
        #     max_grad_norm=1.0,
        # )

        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.global_model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module= self.global_model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon= 1,
            epochs= 1,
            target_delta= 1e-5,
            # noise_multiplier=1.3, #sigma=1.3
            max_grad_norm=1.0,
        )

        h333= time.time()
        print('- >>>>>++++++++++++++++++++++ The ML model set up  Time is       {:f}ms '.format( (h222 - h000)*1000))
        print('- >>>>>++++++++++++++++++++++ The  CKKS Set up Time is       {:f}ms '.format( (h111 - h222)*1000))
        print('- >>>>>++++++++++++++++++++++ The  DP Set up Time is       {:f}ms '.format( (h333 - h111)*1000))

        print('- >>>>>++++++++++++++++++++++ The Set up Time is       {:f}ms '.format( (h333 - h000)*1000))
        



        # self.train_Global_model (Cloud_local_epochs)

    # function : return the public HE parameters
    def Context_ckks(self):
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])  # 8192  /[60, 40, 40, 60]
        # 8192, coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]) 128-bit security
        self.context.global_scale = pow(2, 40) # pow(2, 40) / 29
        self.context.generate_galois_keys()

    def get_context(self):
       return self.context

    def update_global_model(self, local_weights):
        self.global_model.train()
        # copy weights
        global_weights = self.global_model.state_dict()


        # update global weights
        global_weights = FedAvg(local_weights)
        # update global weights
        self.global_model.load_state_dict(global_weights)
        self.global_model 
        return self.global_model.state_dict()

    def decrypt_weights_ckks(self, encr_weights: Dict[str, CKKSVector], shapes: Dict[str, torch.Size], secret_key: Optional[SecretKey] = None) -> Dict[str, torch.Tensor]:
        decry_model = {}
        for key, value in encr_weights.items():
            # Ensure the key is in the correct format
            key_str = key[0] if isinstance(key, tuple) else key

            cts = value.decrypt(secret_key)
            # print(cts)
            result2=[]
            for ct in cts:
                rounded_number = round(ct, 9)
                result2.append(rounded_number)

            # print(result2)

            decry_model[key] = torch.reshape(torch.tensor(result2), shapes[key])

            # average weights
            #decry_model[key] = torch.div(decry_model[key], client_weights)


        return decry_model

    def update_global_model(self, local_weights):
      self.global_model.train()
      # copy weights
      global_weights = self.global_model.state_dict()


      # update global weights
      global_weights = FedAvg(local_weights)
      # update global weights
      self.global_model.load_state_dict(global_weights)
      self.global_model 
      return self.global_model.state_dict()
    


    def update_Ctx_model(self, Ctx_local_weights, local_shapes):
      self.global_model.train()
      # copy weights
      #global_weights = self.global_model.state_dict()

      # compute the average
      t2=time.time()
      CTx_global_weight = FedAvg_encrypted(Ctx_local_weights)

      t3=time.time()
      print('\n ---------------- The time taking to compute the average is : {:f}ms '.format((t3-t2)*1000))


      #decryption
      global_weights = self.decrypt_weights_ckks(CTx_global_weight, local_shapes[0] , self.secret_key) ## a voir de quelle shape il faut mettre ((( j'ai mis celui de plaintext)))
      t4 =time.time()
      print('\n ---------------- The time taking to decrypt model is : {:f}ms '.format((t4-t3)*1000))
      
      # update global weights
      self.global_model.load_state_dict(global_weights)
      self.global_model 
      return self.global_model.state_dict()
    
    def train_Global_model(self, Cloud_local_epochs=10):
            epoch_loss = []
            self.global_model.train()
            t1=time.time()

            for epoch in range(Cloud_local_epochs):

                h22=time.time()
                batch_loss = []
                for batch_idx, (inputs, labels) in enumerate(self.train_loader):# for inputs, labels in train_loader:
                    inputs, labels = inputs , labels 
                    self.optimizer.zero_grad()
                    outputs = self.global_model(inputs)
                    # print(">>>>>>>>>>>>>>>>>>>>>>input ",inputs.size())
                    # print(">>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>>>>>outputs ",outputs.size() )
                    loss = self.criterion(outputs.float(), labels.view(-1, 1).float())
                    loss.backward()

                    self.optimizer.step()

                    if (batch_idx % 3000 == 0):
                        print('| Global Round : {} | Cloud ID : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, self.cloud_id , batch_idx * len(inputs), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))

                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                h33=time.time()
                print('- >>>>> Training time for the vehicle {} in the epochs {} is {:f}s '.format(self.cloud_id ,epoch ,(h33-h22)))

            t4 = time.time()
            print('\n ---------------- The time taking to train the local model without noise and without encryption is : {:f}s'.format( (t4-t1)))
            print(' ---------------- The Total time taking for train the model of vehicle {} : {:f}s | number of eachos = {}'.format(self.cloud_id, (t4-t1), Cloud_local_epochs))

            #return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
            # return self.model.state_dict() , shapes , sum(epoch_loss) / len(epoch_loss)

    def get_Global_model_parameters(self):
        return self.global_model.state_dict()


    def inference(self, test_data,epoch=0):
        features = ['Distance_driven', 'Speed_respect', "secure_dist", "TTC_respect", "safe_dist", "Emergency_Brake", "Total"]

        X = test_data[features]
        y = test_data['Label']
        y.replace(['Slow', 'Normal', 'Dangerous'], [-0.99, 0, 0.99], inplace=True)


        X = X.to_numpy().astype('float32')
        y = y.to_numpy().astype('long')

        # Create PyTorch tensors from NumPy arrays
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)


        testloader = DataLoader(dataset=TensorDataset(X, y), batch_size=128, shuffle=True)

        """ Returns the inference accuracy and loss.
        """
        self.global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        t0=time.time()
        epoch_loss = []

        


        for batch_idx, (values, labels) in enumerate(testloader):
            
            values, labels = values , labels 

            # Inference
            outputs = self.global_model(values)
            loss = nn.MSELoss()(outputs, labels.view(-1, 1)).item()
            
            epoch_loss.append(loss)
            
            # Prediction

            # pred_labels = ((outputs > 0.7).float() + (outputs > 1.66).float())
            pred_labels = ((outputs > -0.33).float() + (outputs > 0.33).float()) - 1
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)
            pred_labels = copy.deepcopy(pred_labels).view(-1)
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",pred_labels)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()  ##correct += (pred_labels == labels.view(-1, 1)).sum().item() #correct += torch.sum(torch.eq(pred_labels, labels)).item()  #
            total += len(labels)
        


        accuracy = correct/total*100
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",accuracy,"<<<<<<<<<<<<<<<<<<<<<<<<<<", epoch)
        t1=time.time()
        # print("||||||||||||||||||||||||||||||||||||||||||",len(epoch_loss))
        print('\n ---------------- The time taking for the prediction of the global model : {:f}ms '.format((t1-t0)*1000))
        return accuracy, sum(epoch_loss) / len(epoch_loss)
    

    def create_Acc_df(self,files, epoch = 0):
        data = {
            'VehicleID': [],
            'Label': [],
            'Accuracy': [],
            'Loss': [],
            'epoch' : []
        }
        # Création de la DataFrame
        df = pd.DataFrame(data)
        
        for file in files: 
            test_data = pd.read_csv(file)
            # Expression régulière pour extraire les parties *
            pattern = re.compile(r'Trajectory_(.*).csv')

            # Extraire les parties * des noms de fichiers correspondants
            match = pattern.search(file)
            asterisk_part = match.group(1)
            vehicleID = asterisk_part
            label = vehicleID.split('_')[0]
            
            test_acc1, loss1 = self.inference(test_data,epoch)
            
            
            df = df._append({
                    'VehicleID': vehicleID,
                    'Label': label,
                    'Accuracy': test_acc1,
                    'Loss': loss1,
                    'epoch': epoch,
                }, ignore_index=True)
        
        return df





















