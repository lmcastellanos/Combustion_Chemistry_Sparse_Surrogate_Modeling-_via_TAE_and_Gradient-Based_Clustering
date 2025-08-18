import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from os import environ
import os as os

home=environ['HOME']
folder1='GRIMech_CH4_models_RS3_15_trajectories'
folder2='Datasets'

phi=np.array([0.45,0.55, 0.65,0.75, 0.85, 0.95,1.05, 1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85])

phis=np.array([1.35,1.45,1.55,1.65,1.75,1.85])

name_1='State_space_cte_pressure_T1418_st-quarter_phi_'
name_2='Reaction_rates_cte_pressure_T1418_st-quarter_phi_'

end='.csv'

cluster_name='cluster_1'
os.mkdir(cluster_name)

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

for i in range(len(phis)):
    
    index,=np.where((phi==phis[i]))
    
    cantera_species=pd.read_csv(join(home, folder1,folder2,name_1+str(index[0])+end))
    cantera_species=pd.DataFrame(cantera_species)
    
    cantera_sources=pd.read_csv(join(home, folder1,folder2,name_2+str(index[0])+end))
    cantera_sources=pd.DataFrame(cantera_sources)
    
    maximum_values=pd.read_csv(join(home, folder1,folder2,'maximum_values_T1418.csv'))
    maximum_values=pd.DataFrame(maximum_values)
    
    dataset, n_columns_source, n_columns_mass_fraction, columns=hydrogen_data_clean_shift_grimech_cantera(cantera_species,cantera_sources,maximum_values)
    
    if i==0:
        n_samples=np.shape(dataset)[0]
        attach=np.ones((n_samples,1))*phis[i]
        attach=pd.DataFrame(attach, columns=['Phi'])
        dataset_copy=pd.concat([dataset, attach],axis=1)
    else:
        n_samples=np.shape(dataset)[0]
        attach=np.ones((n_samples,1))*phis[i]
        attach=pd.DataFrame(attach, columns=['Phi'])
        dataset=pd.concat([dataset, attach],axis=1)
        dataset_copy=pd.concat([dataset_copy,dataset],axis=0)

latent_features=['H', 'CH4', 'H2','CH3O','C2H5', 'Phi']

input_features=np.zeros((np.shape(dataset_copy)[0],len(latent_features)))

for i in range(len(latent_features)):
    label=latent_features[i]
    input_features[:,i]=dataset_copy.loc[:,label]
input_features=pd.DataFrame(input_features, columns=latent_features)

output_start=3+n_columns_mass_fraction+n_columns_source
output_end=output_start+n_columns_mass_fraction+1
outputs=dataset_copy.iloc[:,output_start:output_end]

basis=input_features.iloc[:,:-1]

inputs=input_features
basis=input_features.iloc[:,:-1].to_numpy()
basis_squared=basis**2
for i in range(len(latent_features)-1): 
    #print(i)
    for j in range(np.shape(basis)[1]):
        #print(j)
        new_features=np.multiply(basis[:,j],input_features.iloc[:,i].to_numpy())
        new_features_squared=np.multiply(basis_squared[:,j],input_features.iloc[:,i].to_numpy())
        new_features=pd.DataFrame(new_features)
        new_features_squared=pd.DataFrame(new_features_squared)
        inputs=pd.concat([inputs,new_features, new_features_squared],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.1, random_state=42)

out_columns=columns[1:2+n_columns_mass_fraction]

cols=np.shape(inputs)[1]
rows=np.shape(outputs)[1]
coefficients=np.zeros((rows,cols))
print(np.shape(coefficients))

intercepts=np.zeros((rows,1))
print(np.shape(intercepts))

scores=np.zeros((rows,1))

from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

n_split=5
n_repeats=3

for k in range(rows):

    for train_index,test_index in RepeatedKFold(n_splits=n_split, n_repeats=n_repeats,random_state=42).split(X_train):
    
        x_train,x_test=X_train.iloc[train_index],X_train.iloc[test_index]
        Y_train,Y_test=y_train.iloc[train_index],y_train.iloc[test_index]
    
        reg = Ridge(alpha=0.3).fit(np.asarray(x_train),np.asarray( Y_train.iloc[:,k]))
        
        score=reg.score(np.asarray(x_train), np.asarray(Y_train.iloc[:,k]))
        scores[k]=score
        
        coeff=reg.coef_
        coefficients[k,:]=coeff
        
        intercept=reg.intercept_
        intercepts[k]= intercept

bins1 = np.linspace(0.1, 1.0, 10)  # Replace or adjust as needed
counts, bins = np.histogram(scores, bins=bins1)

plt.figure(figsize=(10, 6), dpi=150)

plt.hist(bins1[:-1], bins1, histtype='step', fill=True,
         weights=counts, color='salmon', edgecolor='black')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('Number\nof Species', fontsize=20, rotation='horizontal', labelpad=50)
plt.xlabel('$R^{2}$-scores', fontsize=20)

for i in range(len(bins1) - 1):
    if counts[i] > 0:
        x = bins1[i] + (bins1[1] - bins1[0]) / 2
        y = counts[i] + 0.1
        plt.text(x, y, str(int(counts[i])), fontsize=14, ha='center')

plt.tight_layout()
plt.savefig(join(cluster_name,'r2_scores_cluster.jpg'),
            dpi=300, bbox_inches='tight')  # High-res output only at save time

scores=pd.DataFrame(scores)
scores.to_csv(join(cluster_name,'cluster1_scores.csv'))

j=11 #equivalence ratio to be checked 

cantera_species=pd.read_csv(join(home, folder1,folder2,name_1+str(j)+end))
cantera_species=pd.DataFrame(cantera_species)
    
cantera_sources=pd.read_csv(join(home, folder1,folder2,name_2+str(j)+end))
cantera_sources=pd.DataFrame(cantera_sources)
    
maximum_values=pd.read_csv(join(home, folder1,folder2,'maximum_values_T1418.csv'))
maximum_values=pd.DataFrame(maximum_values)
    
dataset, n_columns_source, n_columns_mass_fraction, columns=hydrogen_data_clean_shift_grimech_cantera(cantera_species,cantera_sources,maximum_values)

n_samples=np.shape(dataset)[0]
attach=np.ones((n_samples,1))*phi[j]
attach=pd.DataFrame(attach, columns=['Phi'])
dataset=pd.concat([dataset, attach],axis=1)

print('Equivalence ratio')
print(phi[j])

prediction_features=np.zeros((np.shape(dataset)[0],len(latent_features)))

for i in range(len(latent_features)):
    label=latent_features[i]
    prediction_features[:,i]=dataset.loc[:,label]
    
prediction_features=pd.DataFrame(prediction_features, columns=latent_features)

predictions_f=pd.DataFrame(prediction_features)
basis=prediction_features.iloc[:,:-1].to_numpy()
basis_squared=basis**2
for i in range(len(latent_features)-1): 
    #print(i)
    for j in range(np.shape(basis)[1]):
        #print(j)
        new_features=np.multiply(basis[:,j],prediction_features.iloc[:,i].to_numpy())
        new_features_squared=np.multiply(basis_squared[:,j],prediction_features.iloc[:,i].to_numpy())
        new_features=pd.DataFrame(new_features)
        new_features_squared=pd.DataFrame(new_features_squared)
        predictions_f=pd.concat([predictions_f,new_features,new_features_squared],axis=1)

Outputs=dataset.iloc[:,output_start:output_end]

t_index=columns.index('t[s]shift')
t_trans=maximum_values.columns.get_loc('t[s]')
time_plot=(dataset.iloc[:,t_index])*maximum_values.iloc[0,t_trans]


for k in range(31):
    
    prediction=np.matmul(predictions_f,np.transpose(coefficients[k,:]))+intercepts[k]
    
    label=out_columns[k]
    transformation_index=maximum_values.columns.get_loc(label)
    
    original=Outputs.iloc[:,k]*maximum_values.iloc[0,transformation_index]
    reconstruction=prediction*maximum_values.iloc[0,transformation_index]
    
    
    plt.figure(k, figsize=(12, 8), dpi=400)
    plt.plot(time_plot,original, label=label+' Cantera', linewidth=3, color='crimson')
    plt.plot(time_plot,reconstruction, label=label+' Reconstruction', linewidth=3, color='k',linestyle='dotted')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylabel(label, fontsize=22, rotation='horizontal', labelpad=50)
    plt.xlabel('t [s]', fontsize=22)
    plt.legend(loc='best', fontsize=22)
    plt.savefig(join(cluster_name,label + 'phi_'+str(j)+'.jpg'),dpi=400, bbox_inches='tight')

coefficients=pd.DataFrame(coefficients)
intercepts=pd.DataFrame(intercepts)

coefficients.to_csv(join(cluster_name,'cluster1_coefficients.csv'))
intercepts.to_csv(join(cluster_name,'cluster1_intercepts.csv'))
