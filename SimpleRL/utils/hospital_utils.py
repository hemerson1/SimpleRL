#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:30:44 2021

@author: hemerson
"""

import numpy as np

def generate_patient(illness_likelihood, death_probs, recovery_time):
    """
    creates a random patient for the hospital and assigns them an illness and 
    a probability of death
    
    Parameters:
    ----------
        
    Return:
    ------
            
    """
       
    # initialise the patients records
    patient_info = dict()
    
    # select a disease classification
    disease_classification = ["mild", "moderate", "severe"]
    patient_class = np.random.choice(disease_classification, size=1, p=illness_likelihood)[0]
    
    # get their probability of death per timestep
    patient_death_prob = np.random.normal(death_probs[patient_class], 0.1, 1)
    patient_death_prob = np.minimum(np.maximum(patient_death_prob, 0.0), 1.0)[0]
    
    # get the number of timesteps for the patient to recover
    patient_recovery_time = recovery_time[patient_class] + np.random.randint(-2, +3)

    # update the patient records
    patient_info['disease_class'] = patient_class
    patient_info['death_prob'] = patient_death_prob  
    patient_info['recovery_time'] = patient_recovery_time       
    
    # has the patient been diagnosed
    patient_info['diagnosed'] = False 
    
    return patient_info
    
if __name__ == "__main__": 
    
    patient_info = generate_patient()
    
    print(patient_info)
    