import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from os.path import join
from os import environ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda:1")
#torch.set_num_threads(2)

home=environ['HOME']
folder1='GRIMech_CH4_models_RS3_15_trajectories'
folder2='Datasets'

cantera_species=pd.read_csv(join(hone, folder1,folder2,'State_space_cte_pressure_T1418_st-quarter_phi_4.csv'))
cantera_species=pd.DataFrame(cantera_species)

cantera_sources=pd.read_csv(join(home,folder1,folder2,'Reaction_rates_cte_pressure_T1418_st-quarter_phi_4.csv'))
cantera_sources=pd.DataFrame(cantera_sources)

maximum_values=pd.read_csv(join(home,folder1,folder2,'maximum_values_T1418.csv'))
maximum_values=pd.DataFrame(maximum_values)

def hydrogen_data_clean_shift_grimech_cantera(cantera_species,cantera_sources,maximum_values):
    cantera_sources=cantera_sources.add_suffix('w')
    cantera_sources=cantera_sources.iloc[:,1:] #for taking out the timestep as data 
    
    cantera_time=cantera_species.iloc[:,1]
    cantera_temperature=cantera_species.iloc[:,2]
    cantera_pressure=cantera_species.iloc[:,3]
    
    cantera_species_fractions=cantera_species.iloc[:,4:]
    print(np.shape(cantera_species_fractions))
    
    cantera_species_fractions=cantera_species_fractions.loc[:,(cantera_species!=0).any(axis=0)]
    cantera_species_fractions=cantera_species_fractions.loc[:, (cantera_species != cantera_species.iloc[0]).any()]
    print(np.shape(cantera_species_fractions))
    
    n_columns_mass_fraction=np.shape(cantera_species_fractions)[1]
    print(n_columns_mass_fraction)
    
    cantera_sources=cantera_sources.loc[:,(cantera_sources!=0).any(axis=0)]
    cantera_sources=cantera_sources.loc[:, (cantera_sources != cantera_sources.iloc[0]).any()]
    
    print(np.shape(cantera_sources))
    n_columns_source=np.shape(cantera_sources)[1]
    print(n_columns_source)
    
    cantera_data=pd.concat([cantera_time, cantera_temperature,cantera_species_fractions,cantera_sources],axis=1)

    maximum_values=maximum_values.iloc[:,1:]
    maximum_values=pd.concat([maximum_values.iloc[:,0:2],maximum_values.iloc[:,3:]],axis=1)
    #print(maximum_values)
    
    maximum_values=maximum_values.to_numpy()
    #print(np.shape(maximum_values))
    
    iterations=np.shape(cantera_data)[1]
    
    #cantera_data.divide(maximum_values)
    for j in range(iterations):
        cantera_data.iloc[:,j]=cantera_data.iloc[:,j]/(maximum_values[0,j])
        #print(maximum_values[0,j])
    
    cantera_data_shift=cantera_data.loc[1:,:]
    cantera_data_shift=cantera_data_shift.add_suffix('shift')
    
    cantera_data=cantera_data.reset_index()
    cantera_data_shift=cantera_data_shift.reset_index()
    
    cantera_data=cantera_data.iloc[:,1:]
    cantera_data_shift=cantera_data_shift.iloc[:,1:]

    cantera_data=cantera_data.iloc[0:(np.shape(cantera_data_shift)[0]),:]
    
    data_all=pd.concat([cantera_data, cantera_data_shift], axis=1)

    columns=data_all.columns.to_list()
    
    return data_all, n_columns_source, n_columns_mass_fraction, columns

dataset, n_columns_source, n_columns_mass_fraction, columns=hydrogen_data_clean_shift_grimech_cantera(cantera_species,cantera_sources,maximum_values)

dataset_copy=dataset

seed=42
tf.random.set_seed(seed)
np.random.seed(seed)

alpha=0.001

dropout_p=0 #dropout layer percentage

initializer = tf.keras.initializers.GlorotNormal()

initializer_b='zeros'

input_size=n_columns_mass_fraction+1 #the number of mass fractions plus one more column for the temperature

output_size=input_size #we want the same shape of the input vector 

reduced_size=3 #this is just for starting, we probably will change it later

architecture_e=[130,111,320,178]

architecture_d=architecture_e[::-1]

class NN(Model): #The part inside the parenthesis is a standard syntaxis function for defining the class object
    
    def __init__(self):
        super(NN,self).__init__(input_size,output_size,reduced_size,dropout_p,initializer,initializer_b, architecture_e,architecture_d, alpha)
        
        #inputs=tf.keras.Input(shape=(input_size,None))
        
        self.encoder=Sequential()
        self.encoder.add(tf.keras.Input(shape=(input_size)))
        
        for i in range(len(architecture_e)):
            self.encoder.add(Dense(units=architecture_e[i], kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b))
            self.encoder.add(tf.keras.layers.LeakyReLU(alpha))
            self.encoder.add(Dropout(dropout_p))
        #the for cycle describes the general structure 
        self.encoder.add(Dense(units=reduced_size)) #reduced size output
        
        self.lat_activation=Sequential()
        self.lat_activation.add(tf.keras.layers.Activation('sigmoid')) #lat activation function, just because it appears in the paper
        
        
        self.decoder=Sequential()
        self.decoder.add(tf.keras.Input(shape=(reduced_size)))
        
        for i in range(len(architecture_d)):
            self.decoder.add(Dense(units=architecture_d[i], kernel_initializer=initializer, use_bias=True,bias_initializer=initializer_b))
            self.decoder.add(tf.keras.layers.LeakyReLU(alpha))
            self.decoder.add(Dropout(dropout_p))
       
        self.decoder.add(Dense(units=output_size))
        
    def call(self,x):
        encoded=self.encoder(x)
        mid=self.lat_activation(encoded)
        output=self.decoder(mid)
        return output
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

output_start=3+n_columns_mass_fraction+n_columns_source
output_end=output_start+n_columns_mass_fraction+1

Inputs=dataset.iloc[:,1:2+n_columns_mass_fraction].to_numpy()
Outputs=dataset.iloc[:,output_start:output_end].to_numpy()

monitors=['loss','accuracy', 'val_loss', 'val_accuracy']

repetitions=range(5)

import os
from pathlib import Path
from os.path import join
from os.path import isfile
from pathlib import Path

for i in range(len(monitors)):
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor=monitors[i],
                   min_delta=0.0001,
                   patience=5,
                   restore_best_weights=True)
    print(monitors[i])
    os.mkdir(monitors[i])
    
    from sklearn.model_selection import RepeatedKFold

    for j in range(len(repetitions)):
        n_split=5
        n_repeats=repetitions[j]+1

        for train_index,test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats,random_state=42).split(Inputs):
            x_train,x_test=Inputs[train_index],Inputs[test_index]
            y_train,y_test=Outputs[train_index],Outputs[test_index]
    
            Autoencoder=NN()
            Autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE), run_eagerly=True, metrics = ['accuracy'])

            Autoencoder.fit(x_train, y_train,batch_size=64,
                        epochs=2000,
                        callbacks=[early_stopping],
                        validation_data=(x_test, y_test),
                        verbose=0)
        
            print('Model evaluation ',Autoencoder.evaluate(x_test,y_test))
        
        directory_1=monitors[i]
        directory_2='repeats_'+str(repetitions[j]+1)
        
        address=join(directory_1,directory_2)
        os.mkdir(address)
        name=join(address,f'k_fold_best_model_new_architecture_test{reduced_size}')
        target=Path(address)
    
        results=Autoencoder.decoder(Autoencoder.lat_activation(Autoencoder.encoder(dataset_copy.iloc[:,1:2+n_columns_mass_fraction].to_numpy()))).numpy()
    
        Autoencoder.save(name)
        
        interest_vector=['H2O','O2','H2','T[K]','OH','HO2','H2O2','CH3','CH4']
        
        t_index=columns.index('t[s]shift')
        t_trans=maximum_values.columns.get_loc('t[s]')
        time_plot=(dataset_copy.iloc[:,t_index])*maximum_values.iloc[0,t_trans]
        
        for k in range(len(interest_vector)):
            original_index=columns.index(interest_vector[k]+'shift')
            #print(columns[original_index])
            results_index=columns.index(interest_vector[k]) #minues one due to the time column presence 
            #print(columns[results_index])
            #print(results_index-1)
            transformation_index=maximum_values.columns.get_loc(interest_vector[k])
            plot_name=interest_vector[k]+'.png'
            
            input_label=interest_vector[k]+' Dataset'
            output_label=interest_vector[k]+' Reconstruction'
            
            original=(dataset_copy.iloc[:,original_index]).to_numpy()
            original=original*maximum_values.iloc[0,transformation_index]
    
            output=(results[:,results_index-1])
            output=output*maximum_values.iloc[0,transformation_index]
            
            fig_name=join(address,interest_vector[k])
            
            plt.figure(k)
            plt.scatter(time_plot,original, label=input_label)
            plt.scatter(time_plot,output, label=output_label)
            plt.title(interest_vector[k]+' plot results')
            plt.xlabel('Time [S]')
            plt.ylabel(interest_vector[k])
            plt.legend()
            plt.savefig(fig_name)
            plt.clf()