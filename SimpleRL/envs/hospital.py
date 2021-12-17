#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 22:23:28 2021

@author: hemerson
"""

""" 
hopsital_env - a simple grid environment for managing a hospital

The player has limited resources and is tasked with treating as many patients
as possible during a set time window. Patients enter the hospital with one of
either a: mild, moderate or severe disease. 

The player must assign staff to the patients to reduce the time they spend 
waiting for treatment and also reduce the likelihood of the patient dying. 
The more staff assigned to a patient the faster the treatment. 

Patients must be first diagnosed and then assigned to the relevant treatment.

"""

class hospital_env:
    def __init__(self):
        
        # TODO: how is the state of the enviornment going to be represented?
        # TODO: what are the actions going to be? (assigning staff to patients)
        
        # define the illness parameters (for mild, moderate & severe)
        self.illness_likelihood = [0.5, 0.3, 0.2] # liklihood of a patient class
        self.death_probs = {"mild": 0.05, "moderate": 0.15, "severe": 0.25} # mean likelihood of death per class
        self.recovery_time = {"mild": 2, "moderate": 4, "severe": 6} # mean recovery time for each class
        
        # define the staff parameters (for doctors and nurses)
        self.staff_number = 10 # number of staff at the hospital
        self.staff_split = [0.5, 0.5] # ratio of doctors to nurses
        self.staff_efficiency = {"doctor": 2, "nurse": 1} # healing bonus for non-specific tasks 
        self.staff_roles = {"doctor": ["diagnosis", "treatment"], "nurse": ["treatment"]}
        
        # define the hospital parameters       
        self.available_diagnosis_rooms = 5
        self.available_treatment_rooms = 5
        
        self.hospital_staff = []
        self.hospital_patients = []        
        
    def reset(self):
        raise NotImplementedError
        
    def step(self, action=None): 
        raise NotImplementedError