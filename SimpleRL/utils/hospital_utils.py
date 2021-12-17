#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:30:44 2021

@author: hemerson
"""

import numpy as np

def generate_patient(illness_likelihood, death_probs, recovery_time, id_number):
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
    
def generate_staff(staff_split, staff_efficiency, staff_roles, staff_number=10):
    """
    creates a random team of staff members for the hospital each with different roles
    
    Parameters:
    ----------
        
    Return:
    ------
            
    """
    
    staff_roster = []
    staff_classifcation = ["doctor", "nurse"]
        
    for stf in range(staff_number):
        
        # initialise the staff record
        staff_member_info = dict()
        
        # get staff title 
        staff_member_title = np.random.choice(staff_classifcation, size=1, p=staff_split)[0]    
        
        # get staff specific parameters
        staff_member_efficiency = staff_efficiency[staff_member_title]
        staff_member_roles = staff_roles[staff_member_title]
        
        # update their info
        staff_member_info['id'] = stf
        staff_member_info["title"] = staff_member_title
        staff_member_info["efficiency"] = staff_member_efficiency
        staff_member_info["roles"] = staff_member_roles 
        
        # add the treatment related info
        staff_member_info["current_role"] = None
        staff_member_info["patient_id"] = None
        
        # add them to the roster
        staff_roster.append(staff_member_info)
       
    return staff_roster


if __name__ == "__main__": 
    
    # define the illness parameters
    id_number = 1
    illness_likelihood = [0.5, 0.3, 0.2]
    death_probs = {"mild": 0.05, "moderate": 0.15, "severe": 0.25}
    recovery_time = {"mild": 2, "moderate": 4, "severe": 6}
    
    # get patient info
    patient_info = generate_patient(illness_likelihood=illness_likelihood, 
                                    death_probs=death_probs, 
                                    recovery_time=recovery_time, 
                                    id_number = id_number
                                    )
    
    print('\nTest Patient Profile:')
    print(patient_info)
    
    # define the staff parameters
    staff_number = 10
    staff_split = [0.5, 0.5]
    staff_efficiency = {"doctor": 2, "nurse": 1}
    staff_roles = {"doctor": ["diagnosis", "treatment"], "nurse": ["treatment"]}
    
    # get the staff information
    staff_roster = generate_staff(staff_split=staff_split,
                                  staff_efficiency=staff_efficiency,
                                  staff_roles=staff_roles,
                                  staff_number=staff_number
                                  )
    
    print('\nTest Staff Profile:')
    print(staff_roster[0])
    
    