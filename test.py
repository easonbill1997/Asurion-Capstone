#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:48:15 2019

@author: byc
"""
import os
#add a new working directory (root directory of ticket_recog holder)
os.chdir('/Users/byc/Desktop/Asurion-Capstone')

from ticket_recog import recog
from ticket_recog import clean

#can type your code here
ticket = 'Hey\n I just want to cancel this order. Thank you. Name_hidden'

recog.fit_kmeans(ticket)

