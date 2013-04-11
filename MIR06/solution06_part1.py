'''
Multimedia Retrieval Information
Homework 6, part 1
Visual Words

Goal: Quantized version of the image using patches with different configurations such as
varying values of number of clusters, the size of a patch and applying brightness normalization

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




hashMap_classes = []
for i in range(500):
    hashMap_classes.append( {'tree':0, 'none':0, 'birds':0, 'soccer':0 })

############################################################################
# Goal:  Given an image and the patch size, obtained a vector of the image based
# on sampling the patches.  In addition, each path is sub-sampling by a patch of 8x8.
# The result is a 64-feature vector
#
# Parameters:
# image_array: (numpy.array) a color image, which is represented by an array
# patch_size: (int) the size of the patch, which could be 16 
# patch_brightness: (boolean, default: false) flag for applying brightness normalization
##############################################################################



def patch_sampling (image_array, patch_size, patch_brightness = False):
    
    [dim_x, dim_y, dim_z] = image_array.shape
    index = 0
    
    number_patch_x = (int(dim_x) / patch_size) - 1
    number_patch_y = (int(dim_y) / patch_size) - 1
    
    
    
    if patch_size == 16:
        local_feature_vector = numpy.zeros((number_patch_x * number_patch_y*3*4, 64), dtype = numpy.uint8 )
    
    
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            patch_values = image_array[patch_size*index_x :  patch_size*(index_x+1), patch_size*index_y:patch_size*(index_y+1), : ]
                              
                    
            if(patch_values.shape[0] != patch_size or patch_values.shape[1] != patch_size ):
                print "Error en "+str(index_x) + " " + str(index_y)
            elif (patch_size == 16):
                local_feature_vector[index + 0, :] = numpy.array(patch_values[0:8, 0:8, 0].ravel())
                local_feature_vector[index + 1, :] = numpy.array(patch_values[0:8, 0:8, 1].ravel())
                local_feature_vector[index + 2, :] = numpy.array(patch_values[0:8, 0:8, 2].ravel())
                
                local_feature_vector[index + 3, :] = numpy.array(patch_values[0:8, 8:16, 0].ravel())
                local_feature_vector[index + 4, :] = numpy.array(patch_values[0:8, 8:16, 1].ravel())
                local_feature_vector[index + 5, :] = numpy.array(patch_values[0:8, 8:16, 2].ravel())
                
                local_feature_vector[index + 6, :] = numpy.array(patch_values[8:16, 0:8, 0].ravel())
                local_feature_vector[index + 7, :] = numpy.array(patch_values[8:16, 0:8, 1].ravel())
                local_feature_vector[index + 8, :] = numpy.array(patch_values[8:16, 0:8, 2].ravel())
                
                local_feature_vector[index + 9, :] = numpy.array(patch_values[8:16, 8:16, 0].ravel())  
                local_feature_vector[index + 10, :] = numpy.array(patch_values[8:16, 8:16, 1].ravel())
                local_feature_vector[index + 11, :] = numpy.array(patch_values[8:16, 8:16, 2].ravel())
                
                index = index + 12
            
    return local_feature_vector


############################################################################
# Goal:  Given a codebook and an image, obtained a quantized version of the image
# The result is array, which represents an image
#
# Parameters:
# codebook: (numberClusters x 64  array) the list of centroids of the clusters, which were obtained by numpy.kmeans
# scale_down_array: (numpy.array) a greyscale image, which is represented by an array.  In this case, the image is 
#             scale down for quantizying it.  There is onlye one scenario, which is scaling down by a factor of 2   
# patch_brightness: (boolean, default: false) flag for applying brightness normalization
##############################################################################    


def patching_quantize(codebook, scale_down_array, label, patch_brightness = False):
    global hashMap_classes
    dim_x, dim_y, dim_z = scale_down_array.shape
    patch_size = 8

    number_patch_x = (int(dim_x) / patch_size) - 1
    number_patch_y = (int(dim_y) / patch_size) - 1
    image_patch = numpy.zeros((number_patch_x * number_patch_y * 3, 64), dtype = numpy.uint8)

    
   
    index = 0
    
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            
            image_patch[index,:] = scale_down_array[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1), 0 ].ravel()
            
            image_patch[index+1,:] = scale_down_array[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1),1 ].ravel()
                                               
            image_patch[index+2,:] = scale_down_array[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1),2 ].ravel()
                    
            index = index + 3    
    
    
    index_cluster, distortion = vq(image_patch[:, :], codebook)
    
    
    index = 0
    quantize_patch_image = numpy.zeros((number_patch_x*patch_size, number_patch_y*patch_size, 3), dtype = numpy.uint8)

    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            
            quantize_patch_image[patch_size*index_x : patch_size*(index_x+1), \
                                 patch_size*index_y : patch_size*(index_y+1), 0 ] = \
                                 numpy.mean(codebook[index_cluster[index],:])
            
            hashMap_classes[index_cluster[index]][label]+=1
            
            
            quantize_patch_image[patch_size*index_x : patch_size*(index_x+1),\
                        patch_size*index_y : patch_size*(index_y+1), 1 ] = numpy.mean(codebook[index_cluster[index+1],:])
            
            
            hashMap_classes[index_cluster[index+1]][label]+=1
            
            
            quantize_patch_image[patch_size*index_x : patch_size*(index_x+1),\
                        patch_size*index_y : patch_size*(index_y+1), 2 ] = numpy.mean(codebook[index_cluster[index+2],:])
            hashMap_classes[index_cluster[index+2]][label]+=1
            index = index + 3
    
    return quantize_patch_image    


#===============================================================================
#              MAIN PROGRAM
#=================================#==============================================

if __name__ == '__main__':
    pass

#image_name = os.listdir('input_images')[0]
#image_name = '100333061_12e0d98a8c_b.jpg'
file_codebook = open('codebook.pic','w')
file_hashmap = open('hashmap.pic','w')
patch_size = 16
number_cluster = 500
normalization = False

if normalization == True:    
    folder = 'output_images_'+str(patch_size)+'_'+str(number_cluster)+'_norm/'
else:
    folder = 'output_images_'+str(patch_size)+'_'+str(number_cluster)+'/'
    
    
if not os.path.isdir(folder):
    os.makedirs(folder)


if patch_size == 16:
    factor = 2
elif patch_size == 32:
    factor = 4

list_image = []
i=0


train_set = open('imgs.txt.train','r')

for line in train_set:
    
    if len (line.split()) == 2:
        image_name = line.split()[0]
        label = line.split()[1]
    elif len (line.split()) == 1:
        image_name = line.split()[0]
        
        
    try:
        input_image = Image.open(image_name)
    except :
        print "There is a problem with " + image_name.split('/')[1]
        continue
    
    image_array = (pylab.asarray(input_image))  
    
    try:
        temp = patch_sampling(image_array, patch_size, normalization)
    except:
        print "There is a problem in Patch_sampling with " + image_name.split('/')[1]
        continue
        
    for j in range(temp.shape[0]):
        list_image.append(temp[j,:])
    
    i = i + 1
    print i
    
    if (i == 100): break


print "Image Patching"

image_sampling = numpy.array(list_image)

codebook, distortion = kmeans(image_sampling[:,:], number_cluster, iter=10, thresh=1e-03)
print codebook.shape
pickle.dump(codebook, file_codebook)
print "Clustering"



i = 0
train_set = open('imgs.txt.train','r')

for line in train_set:
    
    if len (line.split()) == 2:
        image_name = line.split()[0]
        label = line.split()[1]
    elif len (line.split()) == 1:
        image_name = line.split()[0]
        label = line.split()[0].split('/')[0]
        
    try:
        input_image = Image.open(image_name)
        input_image.save("output_images/" + line.split(' ')[0].split('/')[1])
    except :
        print "There is a problem with " + line.split(' ')[0].split('/')[1]
        continue
     

    
    scale_down_image = input_image.copy() 
    width, height = scale_down_image.size


    scale_down_image = scale_down_image.resize((width/factor, height/factor), Image.BICUBIC)
    scale_down_array = pylab.asarray(scale_down_image)


    image_quantized_sampling = patching_quantize(codebook, scale_down_array, label, normalization)
    Image.fromarray(image_quantized_sampling).save(folder+line.split(' ')[0].split('/')[1])
    
    i = i + 1
    print i
    if (i == 100): break

for i in range(len(hashMap_classes)):
    print hashMap_classes[i]
pickle.dump(hashMap_classes, file_hashmap)