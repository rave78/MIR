#!/usr/bin/env python

import os, glob, string, sys, time, random, numpy
from numpy import zeros,array

def read_dataset(f):
  input=open(f)
  lines=input.readlines()
  data=zeros( (len(lines),2) )
  labels=zeros(len(lines)) 
  for i,l in enumerate(lines):
    v=[float(t) for t in l.split()]
    data[i,0]=v[0]
    data[i,1]=v[1]
    labels[i]=v[2]
  input.close()
  return data,labels
