'''
Created on Jan 5, 2012

@author: federico raue
'''
import pylab
import Image
import os
import numpy

from scipy.cluster.vq import kmeans, vq

def patch_sampling (image_array, patch_size, patch_brightness = False):
    f_var = 50
    dim_x, dim_y = image_array.shape
    index = 0
    number_patch_x = (int(dim_x) / patch_size) - 1
    number_patch_y = (int(dim_y) / patch_size) - 1
    if patch_size == 16:
        local_feature_vector = numpy.zeros((number_patch_x * number_patch_y*4,64), dtype = numpy.uint8 )
    else:
        local_feature_vector = numpy.zeros((number_patch_x * number_patch_y*16,64), dtype = numpy.uint8 )

    
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            patch_values = image_array[patch_size*index_x :  patch_size*(index_x+1), patch_size*index_y:patch_size*(index_y+1) ]
            
            if (patch_brightness == True):
                
                if numpy.std(patch_values) >= 1:
                    patch_values = (f_var * (patch_values - numpy.mean(patch_values) + 128))/numpy.std(patch_values)
                else:
                    patch_values = (f_var * (patch_values - numpy.mean(patch_values) + 128))
                    
                    
            if(patch_values.shape[0] != patch_size or patch_values.shape[1] != patch_size ):
                print "Error en "+str(index_x) + " " + str(index_y)
            elif (patch_size == 16):
                local_feature_vector[index + 0, :] = numpy.array(patch_values[0:8, 0:8].ravel())
                local_feature_vector[index + 1, :] = numpy.array(patch_values[0:8, 8:16].ravel())
                local_feature_vector[index + 2, :] = numpy.array(patch_values[8:16, 0:8].ravel())
                local_feature_vector[index + 3, :] = numpy.array(patch_values[8:16, 8:16].ravel())  
                index = index + 4
            elif (patch_size == 32):
                local_feature_vector[index + 0, :] = numpy.array(patch_values[0:8, 0:8].ravel())
                local_feature_vector[index + 1, :] = numpy.array(patch_values[0:8, 8:16].ravel())
                local_feature_vector[index + 2, :] = numpy.array(patch_values[0:8, 16:24].ravel())
                local_feature_vector[index + 3, :] = numpy.array(patch_values[0:8, 24:32].ravel())  
                
                local_feature_vector[index + 4, :] = numpy.array(patch_values[8:16, 0:8].ravel())
                local_feature_vector[index + 5, :] = numpy.array(patch_values[8:16, 8:16].ravel())
                local_feature_vector[index + 6, :] = numpy.array(patch_values[8:16, 16:24].ravel())
                local_feature_vector[index + 7, :] = numpy.array(patch_values[8:16, 24:32].ravel())  
                
                local_feature_vector[index + 8, :] = numpy.array(patch_values[16:24, 0:8].ravel())
                local_feature_vector[index + 9, :] = numpy.array(patch_values[16:24, 8:16].ravel())
                local_feature_vector[index + 10, :] = numpy.array(patch_values[16:24, 16:24].ravel())
                local_feature_vector[index + 11, :] = numpy.array(patch_values[16:24, 24:32].ravel())  
                
                local_feature_vector[index + 12, :] = numpy.array(patch_values[24:32, 0:8].ravel())
                local_feature_vector[index + 13, :] = numpy.array(patch_values[24:32, 8:16].ravel())
                local_feature_vector[index + 14, :] = numpy.array(patch_values[24:32, 16:24].ravel())
                local_feature_vector[index + 15, :] = numpy.array(patch_values[24:32, 24:32].ravel())  
                
                index = index + 16
                
                
                
            
    return local_feature_vector

def patching_quantize(codebook, scale_down_array, patch_brightness = False):
    dim_x, dim_y = scale_down_array.shape
    patch_size = 8
    f_var = 50
    number_patch_x = (int(dim_x) / patch_size) - 1
    number_patch_y = (int(dim_y) / patch_size) - 1
    image_patch = numpy.zeros((number_patch_x * number_patch_y, 64), dtype = numpy.uint8)
    

    
    quantize_patch_image = numpy.zeros((number_patch_x*patch_size, number_patch_y*patch_size), dtype = numpy.uint8)
    index = 0
    
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            
            image_patch[index,:] = scale_down_array[patch_size*index_x : patch_size*(index_x+1),\
                                               patch_size*index_y : patch_size*(index_y+1) ].ravel()
            
            if (patch_brightness == True):
                if numpy.std(image_patch[index,:]) >= 1:
                    image_patch[index,:] = (f_var * (image_patch[index,:] - numpy.mean(image_patch[index,:]) + 128))/numpy.std(image_patch[index,:])
                else:
                   
                    image_patch[index,:] = (f_var * (image_patch[index,:] - numpy.mean(image_patch[index,:]) + 128))
                    
            index = index + 1    
    
    index_cluster, distortion = vq(image_patch, codebook)
 
    
    index = 0
    for index_x in range(number_patch_x):
        for index_y in range(number_patch_y):
            
            quantize_patch_image[patch_size*index_x : patch_size*(index_x+1),\
                        patch_size*index_y : patch_size*(index_y+1) ] = numpy.mean(codebook[index_cluster[index],:])
            index = index + 1
    
    return quantize_patch_image    
#===============================================================================
#              MAIN PROGRAM
#=================================#==============================================

if __name__ == '__main__':
    pass


patch_size = 16
number_cluster = 100
normalization = True

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
image_to_qv = os.listdir('input_images')[:10]

for image_name in os.listdir('input_images')[:10]:
        
    try:
        input_image = Image.open("input_images/" + image_name).convert('L')
    except :
        print "There is a problem with " + image_name
        continue
    
    image_array = (pylab.asarray(input_image))  
    
    temp = patch_sampling(image_array, patch_size, normalization)
    
    for j in range(temp.shape[0]):
        list_image.append(temp[j,:])
    
    i = i + 1
    print i


print "Image Patching"

image_sampling = numpy.array(list_image)
print image_sampling.shape
codebook, distortion = kmeans(image_sampling, number_cluster)

print "Clustering"

i = 0
for image_name in image_to_qv:
        
    try:
        input_image = Image.open("input_images/" + image_name).convert('L')
        input_image.save("output_images/" + image_name)
    except :
        print "There is a problem with " + image_name
        continue
     
    scale_down_image = input_image.copy() 
    width, height = scale_down_image.size

    scale_down_image = scale_down_image.resize((width/factor, height/factor), Image.BICUBIC)
    scale_down_array = pylab.asarray(scale_down_image)

    image_quantized_sampling = patching_quantize(codebook, scale_down_array, normalization)
    Image.fromarray(image_quantized_sampling).save(folder+image_name)
    