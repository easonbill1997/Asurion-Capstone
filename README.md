# Asurion-Capstone

This package is to fit the new ticket into some classifiers, including junk lines, junk sentences, greetings, itentifiers and useful sentences. 

There are two python file in the package. 

claen.py: 

clean: clean the ticket(tokenization, remove not english words, lowercase etc.)

recog.py: 

fit_kmeans: fit a newly given ticket to some given clusters (junk, greeting, identifiers...)

recluster: give a new pkl file(sentence_df, kmdf) and load it. Need the pkl file in the holder. 

add_new_junk: add some new junk clusters manually. 

I also attached a test.py outside the file. you can just import the package and run the code here. 
