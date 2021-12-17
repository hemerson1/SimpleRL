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

from SimpleRL.utils.hospital_utils import generate_patient, generate_staff, generate_rooms

class hospital_env:
    def __init__(self):
        
        # TODO: how is the state of the enviornment going to be represented?
        # TODO: what are the actions going to be? (assigning staff to patients)
        
        # define the environmental parameters
        self.max_timestep = (24 * 60) / 10 # 24 hrs in 10 minute intervals
        
        # define the illness parameters (for mild, moderate & severe)
        self.illness_likelihood = [0.5, 0.3, 0.2] # liklihood of a patient class
        self.death_probs = {"mild": 0.05, "moderate": 0.15, "severe": 0.25} # mean likelihood of death per class
        self.recovery_time = {"mild": 2, "moderate": 4, "severe": 6} # mean recovery time for each class
        
        # define the patient paramaters
        self.patient_spawn_prob = 0.3 # probability of patient per timestep 
        
        # define the staff parameters (for doctors and nurses)
        self.staff_number = 10 # number of staff at the hospital
        self.staff_split = [0.5, 0.5] # ratio of doctors to nurses
        self.staff_efficiency = {"doctor": 2, "nurse": 1} # healing bonus for non-specific tasks 
        self.staff_roles = {"doctor": ["diagnosis", "treatment"], "nurse": ["treatment"]}
        
        # define the hospital parameters    
        self.number_rooms = 10
        
        # Reset the environment parameters
        self.reset()
        
    def reset(self):
        
        # reset the staff roster
        self.hospital_staff = generate_staff(
            staff_split=self.staff_split, 
            staff_efficiency=self.staff_efficiency, 
            staff_roles=self.staff_roles
            )
        
        # reset the current patients in the hospital
        self.hospital_patients = []
        
        self.room_arrangement = {}
        
    def step(self, action=None): 
        """
        state:
        ------
        
        need to show: 
            - which rooms have patients
            - which rooms have doctors and nurses
            - how severe the patient's disease is 
            - how many days they have been waiting
            - whether they require diagnosis or 
            
        2D array:
        number of rooms x patient_severity (0=None, 1=Mild, 2=Moderate, 3=Severe) 
                          + days waiting (0, 1, 2, 3, ...)
                          + diagnosis or treatment (0=None, 1=diagnosis, 2=treatment)
                          
        n x (1 + 1 + 1) = 3n (e.g. 30 states)
        
        action:
        -------
        
        need to choose:
            - which nurses are assigned to which rooms
            - which doctors are assigned to which rooms
            
        could change the arrangement each timestep
        
        1d array:
        number of doctors + number of nurses (give room number 0, 1, ...)        
        n = n (e.g. 10 actions)        
        
        
        """
        pass
        
if __name__ == "__main__": 

    env = hospital_env()
        
