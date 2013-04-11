'''
Multimedia Retrieval Information
Homework 3, Exercise 3
Text Search Engine

Goal: Developed a program for indexing a small list of webpages

@author: Federico Raue
'''

import os
import fileinput
import pickle
import operator
import logging

from stripogram import html2text
from stemming.porter2 import stem
from math import sqrt

def index(urls):
    """
    Goal:  Download a list of webpage
    
    Parameter:
    urls:  list of strings, which represent the address of each webpage 
    
    """    
    
    if not os.path.isdir('files'):
        os.makedirs('files')
    
    
    for webpage in urls:
        name = webpage.split('/')[-1]
        os.system("wget "+webpage+ " -q -O files/"+name)
        logging.info("Downloaded: "+ name )
        
    b_o_w = {}
    
    for web_file in os.listdir('files'):
        
        try:
            text_html = open('files/'+web_file,'r').read();
            text = [stem(word.lower()) for word in html2text(text_html).split()]
            b_o_w[web_file] = text
            logging.info("Tokenized: "+web_file)
        except :
            #Something strange happened with the webpage of New_York_City
            print ("There is a problem with "+web_file)
    
    index_file = open("index_file.pck", "w") 
    pickle.dump(b_o_w, index_file)
    index_file.close()



def search (query):
    """
    Goal:  Given a query in order to rank a list of webpages
    
    
    Parameter:
    query:  keyword for searching
    
    """
    index_file = open("index_file.pck", "r") 
    b_o_w = pickle.load(index_file)
    index_file.close()
    
    print "\nResults of Query: "+ query
    print "=================="+len(query)*"="
    ranking = {}
    for key in b_o_w.keys():
        if query in b_o_w[key]:
            ranking[key] = float(1.0/(sqrt(len(b_o_w[key])) * sqrt(len(query)))) * 100
            #print 'Cosine Measure ('+ key +'):' + str(float(1.0/(len(b_o_w[key])*len(query))))
        else:
            #print 'Cosine Measure ('+key+'): 0'
            ranking[key] = 0.0
            
    sorted_ranking = sorted(ranking.iteritems(), key=operator.itemgetter(1),  reverse=True)
    
    for index in range(len(sorted_ranking)):
        print str(sorted_ranking[index][0])+" "+str(sorted_ranking[index][1])
        
    
            
#===============================================================================
#             MAIN PROGRAM                                                     #
#===============================================================================
logging.basicConfig(level=logging.INFO)

list_files = []
for line in fileinput.input('list-wikipedia.txt'):
    list_files.append(line.rstrip('\n'))       
    
index(list_files)  

search('colosseum')
search('hungary')
search('tower')
