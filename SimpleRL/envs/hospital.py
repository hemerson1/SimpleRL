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

import numpy as np

from SimpleRL.utils.hospital_utils import generate_patient, generate_staff, generate_rooms

class hospital_env:
    def __init__(self, seed=None):
        
        # TODO: how is the state of the enviornment going to be represented?
        # TODO: what are the actions going to be? (assigning staff to patients)
        # TODO: add seeding
        # TODO: add rendering
        
        # define the environmental parameters
        self.seed = seed
        np.random.seed(self.seed)  
        self.max_timestep = 30 * 6 # 6 months
        
        # define the illness parameters (for mild, moderate & severe)
        self.disease_classification = ["mild", "moderate", "severe"]
        self.illness_likelihood = [0.5, 0.3, 0.2] # liklihood of a patient class
        self.death_probs = {"mild": 0.05, "moderate": 0.15, "severe": 0.25} # mean likelihood of death per class
        self.recovery_time = {"mild": 2, "moderate": 4, "severe": 6} # mean recovery time for each class
        
        # define the patient paramaters
        self.patient_spawn_prob = 1.0 # probability of patient per timestep 
        
        # define the staff parameters (for doctors and nurses)
        self.staff_number = 10 # number of staff at the hospital
        self.staff_split = [0.5, 0.5] # ratio of doctors to nurses
        self.staff_efficiency = {"doctor": 2, "nurse": 1} # healing bonus for non-specific tasks 
        self.staff_roles = {"doctor": ["diagnosis", "treatment"], "nurse": ["treatment"]}
        
        # define the reward
        self.death_penalty = -100
        self. waiting_penalty = -1
        
        # define the hospital parameters    
        self.number_rooms = 10
        
        # Reset the environment parameters
        self.reset()
        
    def reset(self):
        
        # reset the staff roster
        self.hospital_staff, self.doctor_ids = generate_staff(
            staff_split=self.staff_split, 
            staff_efficiency=self.staff_efficiency, 
            staff_roles=self.staff_roles
            )
        
        print('Doctor ids: {}'.format(self.doctor_ids))
        
        # reset the current patients in the hospital
        self.hospital_patients = []
        self.current_patient_id = 0
        self.day_counter = 0
        
        # get the score tallys
        self.total_deaths = 0
        self.total_days_waited = 0
        self.total_recoveries = 0
        
        # set up the room arrangement
        self.room_arrangement = generate_rooms(number_rooms=self.number_rooms)
        
        state = np.zeros((self.number_rooms, 3))
        
        return state.flatten()
        
        
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
        
        # initialise the reward tallys
        waiting_today = 0
        deaths_today = 0    
        recovered_today = 0
        
        # update the rooms with the staff choices -------------------
        
        # clear the current staff arrangements
        for idx, _ in enumerate(self.room_arrangement):
            self.room_arrangement[idx]['staff_ids'] = []            
        
        # cycle through the staff assignments
        for staff_idx, room_num in enumerate(action):
            
            # add the ids of the staff to the relevant rooms
            self.room_arrangement[room_num]["staff_ids"].append(staff_idx)                  
            
        # add the changes as a result of staff presence -------------
        
        for room_idx, room in enumerate(self.room_arrangement):
            
            # if there is a patient in the room
            if room["patient_id"] != None:
                
                current_patient = None
                current_patient_idx = None
                                
                # find the patient's medical record
                for idx, patient in enumerate(self.hospital_patients):                  
                    
                    if patient["id"] == room["patient_id"]:
                        current_patient = patient
                        current_patient_idx = idx
                        break
                
                # cycle through staff and allow doctors actions first
                for stf in room['staff_ids']:
                    
                    # if this staff member is a doctor
                    if stf in self.doctor_ids:
                        
                        # diagnose the patient
                        if not current_patient['diagnosed']:
                            current_patient['diagnosed'] = True
                            continue
                    
                        # allow treatment if patient has been diagnosed                    
                        if current_patient['diagnosed']:                        
                            # treat the patient
                            current_patient['recovery_time'] -= self.staff_efficiency['doctor']
                
                # now allow nurses' actions
                for stf in room['staff_ids']:
                    
                    # if this staff member is a doctor
                    if stf not in self.doctor_ids and current_patient['diagnosed']:
                 
                        # treat the patient
                        current_patient['recovery_time'] -= self.staff_efficiency['nurse']
                
                # remove the patient if they have recovered
                recovered = False                
                if current_patient['recovery_time'] <= 0:
                    
                    print('Patient {} has recovered'.format(current_patient['id']))
                    
                    # remove from hospital patients
                    self.hospital_patients.pop(current_patient_idx)
                    
                    # update the room
                    self.room_arrangement[room_idx]["patient_id"] = None       
                    
                    recovered = True
                    
                    # update recovered
                    recovered_today += 1
                
                # update the illness of the patient     
                self.hospital_patients[current_patient_idx]['waiting_time'] += 1
                waiting_today += 1
                
                # update the days deaths
                if np.random.uniform(0, 1, 1) < current_patient['death_prob'] and not recovered:
                    
                    print('Patient {} has died'.format(current_patient['id']))
                    
                    # update deaths
                    deaths_today += 1
                    
                    # remove from hospital patients
                    self.hospital_patients.pop(current_patient_idx)
                    
                    # update the room
                    self.room_arrangement[room_idx]["patient_id"] = None 
                    
                            
        # generate a patient ----------------------------------------
              
        if np.random.uniform(0, 1, 1) < self.patient_spawn_prob and len(self.hospital_patients) < self.number_rooms:
            
            # create new patient
            new_patient = generate_patient(
                disease_classification=self.disease_classification,
                illness_likelihood=self.illness_likelihood, 
                death_probs=self.death_probs, 
                recovery_time=self.recovery_time, 
                id_number = self.current_patient_id
                )
        
            # add patients to the rooms
            for room in self.room_arrangement:
                
                # add patient ID to room and update current_id
                if room['patient_id'] == None:
                    
                    # add patient id to room
                    room['patient_id'] = self.current_patient_id
                    
                    # add room number to patient
                    new_patient['room_number'] = room['room_number']
                    
                    # add patient to the hospital list
                    self.hospital_patients.append(new_patient)
                    
                    # update patient_id
                    self.current_patient_id += 1
                    
                    break
                
        # visualise the next state --------------------------------
        
        state = np.zeros((self.number_rooms, 3))
        
        for idx, room in enumerate(self.room_arrangement):
            
            # if there is a patient in the room
            if room["patient_id"]:
                
                current_patient = None
                
                # find the patient's medical record
                for idx, patient in enumerate(self.hospital_patients):                    
                    if patient["id"] == room["patient_id"]:
                        current_patient = patient
                
                # get the disease class, days waiting and treatment/diagnosis stage
                state[idx, 0] = self.disease_classification.index(current_patient["disease_class"]) + 1
                state[idx, 1] = current_patient["waiting_time"]
                state[idx, 2] = int(current_patient["diagnosed"]) + 1
        
        # calculate the reward -----------------------------------
        
        self.total_days_waited += waiting_today
        self.total_deaths += deaths_today
        self.total_recoveries += recovered_today
        
        reward = self.death_penalty * deaths_today + self.waiting_penalty * waiting_today
        
        # get done -----------------------------------------------
        
        done = False
        self.day_counter += 1
        if self.day_counter == self.max_timestep:
            done = True
        
        info = {}
            
        return state, reward, done, info        
            
        
if __name__ == "__main__": 
    
    test_days = 10
    staff_number = 10
    
    # initialise the environment
    env = hospital_env(seed=0)
    
    for days in range(test_days):
    
        # test action with each element representing a person 
        # and the idx representing their room assignment        
        
        action = np.random.randint(10, size=10)  
        
        print('Chosen_action: {}'.format(action))
        
        print('\nDay {} ----------'.format(days))
        
        # take a step
        env.step(action=action)
        
        print('\nRoom Arrangement:')
        print(env.room_arrangement)
        
        print('\nCurrent Patients:')
        print(env.hospital_patients)
        print('---------------------------')
    
    print('\nSUMMARY ----------------------------------')
    print('Total patients seen: {}'.format(env.current_patient_id + 1))
    print('Total deaths: {}'.format(env.total_deaths))
    print('Total recoveries: {}'.format(env.total_recoveries))
    print('Total days waiting: {}'.format(env.total_days_waited))
    print('-------------------------------------------')
        
    
        
