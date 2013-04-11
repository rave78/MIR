'''
Multimedia Retrieval Information
Homework 6, part 2
Visual Words

Goal: Givan an image and a codebook, produce a probability map of P(concept|visual word)

@author: Federico Raue
'''

import pylab
import Image
import os
import numpy

from scipy.cluster.vq import kmeans, vq

try:
   import cPickle as pickle
except:
   import pickle



def test(image, label, codebook, hashMap_classes):
    dim_x, dim_y, dim_z = image.shape
    patch_size = 8

    number_patch_x = (int(dim_x) / patch_size) - 1
    number_patch_y = (int(dim_y) / patch_size) - 1
    image_patch = numpy.zeros((number_patch_x * number_patch_y * 3, 64), dtype = numpy.uint8)

    
   
    index = 0
    
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            
            image_patch[index,:] = image[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1), 0 ].ravel()
            
            image_patch[index+1,:] = image[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1),1 ].ravel()
                                               
            image_patch[index+2,:] = image[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1),2 ].ravel()
                    
            index = index + 3    
    
    
    index_cluster, distortion = vq(image_patch[:, :], codebook)
    
    index = 0
    quantize_patch_image = numpy.zeros((number_patch_x*patch_size, number_patch_y*patch_size), dtype = numpy.uint8)

    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            p1 = 0.0
            p2 = 0.0
            
            for i in hashMap_classes[index_cluster[index]].keys():
                if(i == label):
                    p1 += hashMap_classes [index_cluster[index]][label] 
                elif (i != label):
                    p2 += hashMap_classes [index_cluster[index]][i]    
           
            print str(p1) + ' ' + str(p2)
            if (p1 + p2) != 0:        
                quantize_patch_image[patch_size*index_x : patch_size*(index_x+1), \
                                     patch_size*index_y : patch_size*(index_y+1)] = \
                                     255 * 0.5*p1/(0.5*p1+0.5*p2)
                       
            elif(p1 + p2) == 0:
                quantize_patch_image[patch_size*index_x : patch_size*(index_x+1), \
                                     patch_size*index_y : patch_size*(index_y+1)] = 0
            
                
            index += 1 
    
    return quantize_patch_image    




file_hashmap = open('hashmap.pic')
hashMap_classes = pickle.load(file_hashmap)

file_codebook = open('codebook.pic')
codebook = pickle.load(file_codebook)


image_name = 'soccer/54Z92Qur1fM_000020_003664.jpg'

label = image_name.split('/')[0]
print label
try:
    input_image = Image.open(image_name)
    input_image.show()
    input_image.save('image_map/' +image_name.split('/')[1].split('.')[0]+'_orig.jpg')
except :
    print "There is a problem with " + image_name.split('/')[1]
    

width, height = input_image.size
scale_down_image = input_image.resize((width/2, height/2), Image.BICUBIC)
scale_down_image.save('image_map/' +image_name.split('/')[1].split('.')[0]+'_scale.jpg')
scale_down_image.show()
scale_down_array = pylab.asarray(scale_down_image)

image_result = test(scale_down_array, label, codebook, hashMap_classes)

Image.fromarray(image_result).show()
Image.fromarray(image_result).save('image_map/' +image_name.split('/')[1].split('.')[0]+'_map.jpg')