#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 22:23:28 2021

@author: hemerson
"""

""" 
hopsital_env - a simple grid environment for managing a hospital

The player has limited resources and is tasked with treating as many patients
as possible during a set time window. Patients enter the hospital and following 
diagnosis require one of either: medicine, surgery or monitoring. 


"""

class hospital_env:
    def __init__(self):
        pass
        
    def reset(self):
        raise NotImplementedError
        
    def step(self, action=None): 
        raise NotImplementedError