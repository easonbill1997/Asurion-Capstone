#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:46:41 2019

@author: byc
"""

import numpy as np
import pandas as pd
import re
import pickle
import tensorflow as tf
import tensorflow_hub as hub

from ticket_recog import clean

global km_line
global km_senetence
global embed
global sentence_df
global kmdf
global junk_list

#load kmeans pkl
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

with open('mbk_line.pkl', 'rb') as f:
    km_line = pickle.load(f)
with open('mbk_sentence.pkl', 'rb') as f:
    km_sentence = pickle.load(f)


#load cluster csv
sentence_df = pd.read_csv('sentence_df.csv')
kmdf = pd.read_csv('kmdf.csv')

#load new junk list
junk_list = pd.read_csv('new_junk_list.csv')

def add_new_junk(inputname):
    df = pickle.load(open(inputname, 'rb'), encoding='utf-8')
    sen = df
    filtered_df = sen[sen['useless'].isnull()]
    not_identifier = filtered_df[filtered_df['label'].isnull()]
    mean_centr=not_identifier.groupby(not_identifier['clusters'])['distance_center'].mean()
    low_list=list(mean_centr[mean_centr.values<0.7].index)
    low=not_identifier[not_identifier['clusters'].isin(low_list)]
    low_cluster_count = low.groupby("clusters").count()
    low_more_cluster = low_cluster_count[low_cluster_count['ticket'] >= 14]
    junk=list(low_more_cluster.index)
    remove_list=[7,23,56]
    add_list=[197,201,78,280,222,57]
    new_junk=[x for x in junk if x not in remove_list]
    new_junk.extend(add_list)
    junk_list = new_junk
    pd_junk = pd.DataFrame({'junk': junk_list})
    pd_junk.to_csv('new_junk_list.csv')
    return junk_list



def recluster(inputname, outputname):#given a new kmdf/sentence df, it can reoutput a new cluster csv
    df = pickle.load(open(inputname, 'rb'), encoding='utf-8')
    cluster = df.clusters
    cluster_list = list(set(cluster))
    label_list = []
    for item in cluster_list: 
        temp = df[df.clusters == item].label.iloc[0]
        label_list.append(temp)
    cluster_df = pd.DataFrame({'cluster':cluster_list, 'label':label_list})
    cluster_df.to_csv(outputname)
    return cluster_df


def fit_kmeans(ticket):   
    #clean ticket
    ticket=clean.clean(ticket)

    #split lines
    junk_line=[]
    none_line=[]
    lines_splited = re.split('\n',ticket)
    for line in lines_splited:
        text = tf.convert_to_tensor([line])
        text_embed = embed(text)
        text_embed = np.asarray(text_embed["outputs"])
        cluster = km_line.predict(text_embed)[0]
        if kmdf[kmdf.cluster == cluster].label.iloc[0] == 'Not Message':
            junk_line.append(line)
        else: none_line.append(line)
    print('\n')
    
    #split sentences
    junk_sentence = []
    greetings = []
    ident = []
    useful_sentence = []
    for line in none_line: 
        sentences_splited = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',line)
        for sentence in sentences_splited:
            #manually detect some junk
            isjunk = 0
            p=re.compile(r'[-,$()#+&* ]')
            t=sentence
            
            punc_detect=re.findall(p,t)
            if len(punc_detect)/len(t)>=0.8: isjunk = 1
            #by space: filture by no space but more than 10 letter or numbers
            n=re.compile('[^\s-]{10,}')
            space_detect=re.findall(n,t)
            space_detect
            total_len=0
            for i in range(0,len(space_detect)):
                total_len+=len(space_detect[i])
            if total_len/len(t)>=0.8: isjunk = 1
            
            if isjunk == 1: 
                junk_sentence.append(sentence)
                continue
            
            text = tf.convert_to_tensor([sentence])
            text_embed = embed(text)
            text_embed = np.asarray(text_embed["outputs"])
            cluster = km_sentence.predict(text_embed)[0]
            if cluster in junk_list:
                junk_sentence.append(sentence)
            elif sentence_df[sentence_df.cluster == cluster].label.iloc[0] == 'Hello / Bye':
                greetings.append(sentence)
            elif sentence_df[sentence_df.cluster == cluster].label.iloc[0] == 'Identifier':
                ident.append(sentence)
            elif sentence_df[sentence_df.cluster == cluster].label.iloc[0] == 'Useless':
                junk_sentence.append(sentence)
            else: useful_sentence.append(sentence)
    junk_line = '.'.join(junk_line)
    junk_sentence = '.'.join(junk_sentence)
    greetings = '.'.join(greetings)
    ident = '.'.join(ident)
    useful_sentence = '.'.join(useful_sentence)
    return {'junk_line':junk_line, 'junk_sentence':junk_sentence, 'greetings':greetings, 'ident':ident, 'useful_sentence':useful_sentence}



