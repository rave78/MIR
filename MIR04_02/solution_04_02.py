#!\bin\python

'''
Multimedia Retrieval Information
Homework 4, Exercise 2
Blobs

Goal: Developed a program for finding blobs using Difference-of-Gaussian

@author: Federico Raue
'''


from __future__ import division
from scipy.ndimage import filters
from numpy import asarray
from pylab import where,float64, sqrt
from sets import Set

import Image
import ImageDraw



def preprocessing(input_image, scale, factor):
    """Given an image object, return the difference of gaussian matrix
    
    """
    var1 = scale / sqrt(2)
    var2 = scale * sqrt(2)
    image_array = (asarray(input_image)).astype(float64)/255. * scale
    threshold = image_array.max() * factor
    image_array[where(image_array < threshold)] = 0
    image_array[where(image_array >= threshold)] = 255
    
    image_gaussian = filters.gaussian_filter(image_array, sigma=(var1,var1)) - \
                     filters.gaussian_filter(image_array, sigma=(var2,var2))
    
    return image_gaussian    
    

    
def local_maximum(image_gaussian,i,j):
    """Given a point in the image, return 1 if is local maxima in its 8-neighborhood 
    
    """
    
    return (image_gaussian[i,j-1] < x  and   
            image_gaussian[i,j+1] < x  and 
            image_gaussian[i-1,j] < x  and 
            image_gaussian[i+1,j] < x  and 
            image_gaussian[i-1,j-1] < x  and  
            image_gaussian[i-1,j+1] < x  and 
            image_gaussian[i+1,j-1] < x  and 
            image_gaussian[i+1,j-1] < x)                         

            

def radio_maximum_north(image_gaussian, i, j):      
    """Given a point in the image, return the radio in the north
    
    """
    radio = 0
    if(i - 1 > 0):
        while( i > 0 and image_gaussian[i,j] > image_gaussian[i-1,j] ):
            radio = radio + 1
            i = i - 1
            
    return radio
            


def radio_maximum_south(image_gaussian, i, j):
    """Given a point in the image, return the radio in the south
    
    """
    dimX = image_gaussian.shape[0]
    radio = 0
    if(i+1 < dimX-1): 
        
        while(i < dimX and image_gaussian[i,j] > image_gaussian[i+1,j] ):
            radio = radio + 1
            if(i+1 < dimX - 1): i = i + 1
            else: break

    return radio



def radio_maximum_east(image_gaussian, i, j):
    """Given a point in the image, return the radio in the east
    
    """
    dimY = image_gaussian.shape[1]
    radio = 0
    if(j+1 < dimY):
        
        while(j < dimY and image_gaussian[i,j] > image_gaussian[i,j+1] ):
            radio = radio + 1
            if(j+1 < dimY-1): j = j + 1
            else: break
        
    return radio



def radio_maximum_west(image_gaussian, i, j):
    """Given a point in the image, return the radio in the west
    
    """
    radio = 0
    if(j -1 > 0):                
        
        while(j > 0 and image_gaussian[i,j] > image_gaussian[i,j-1] ):
            radio = radio + 1
            j = j - 1

    return radio



def radio_maximum(image_gaussian, i, j):
    """Given a point in the image, return the maximum radio among the N-S-E-W directions
    
    """
    radio_max = []
    
    radio_max.append(radio_maximum_north(image_gaussian, i ,j))
    radio_max.append(radio_maximum_south(image_gaussian, i ,j))
    radio_max.append(radio_maximum_east(image_gaussian, i ,j))
    radio_max.append(radio_maximum_west(image_gaussian, i ,j))    
    
    return max(radio_max)    



def cluster(local_max):
    """  Given all local maximum in the image, group all its points based on proximity 
    
    """
    cluster = [[local_max.pop()]]

    for x,y,radio in local_max:    
        for j in range(len(cluster)):
            x_average = 0
            y_average = 0
            radio_max = cluster[j][0][2]
            add = False
        
            for z in range(len(cluster[j])):
                x_average += cluster[j][z][0]
                y_average += cluster[j][z][1]
            
                if(cluster[j][z][2] > radio_max):
                    radio_max = cluster[j][z][2] 
        
                distance = sqrt( (x - x_average/len(cluster[j]))**2 + (y - y_average/len(cluster[j]))**2 )     
        
            if( distance <= radio or distance <= radio_max ):
                cluster[j].append((x, y, radio))
                add = True
                break
    
        if(add == False):
            cluster.append([((x, y, radio))])  
    
    return cluster        
            


def finding_center(cluster_points):
    """Given all the points in the blob, return their centroids
    
    """
    
    blob_centers = []
    
    for i in range(len(cluster_points)):
        x_average = 0
        y_average = 0
        radio_average = 0
        
        for j in range(len(cluster_points[i])):
            x_average += cluster_points[i][j][0] 
            y_average += cluster_points[i][j][1]
            radio_average += cluster_points[i][j][2]
        
        normalize = len(cluster_points[i])
        
        blob_centers.append((x_average/normalize,y_average/normalize,radio_average/normalize))
    
    return blob_centers             



def blob_drawing(name_file):
    """Given the name of image file, drawing the image and return the average of the radios among all blobs
    
    """
    input_image = Image.open("input_images/" + name_file + ".jpg").convert("RGB")
    draw = ImageDraw.Draw(input_image)
    r_average = 0
    for y,x,r in blob_centers:
        draw.ellipse((x-r,y-r, x+r, y+r),outline="#FF0000")
        r_average = r + r_average
    
    return input_image, r_average

                        
            
#===================================================================#
#                                                                   #
#                        MAIN PROGRAM                               # 
#                                                                   #     
#===================================================================#


#---Parameters for the program-------------------------------- 
name_file = "cells1"
scale = 3.9
factor = 0.90

try:
    input_image = Image.open("input_images/" + name_file + ".jpg").convert("L")
except :
    print "There is a problem with " + name_file + ".jpg"


image_gaussian = preprocessing(input_image, scale, factor)

local_max = Set([])

dimX = image_gaussian.shape[0]
dimY = image_gaussian.shape[1]

for i in range(1, dimX-1):
    for j in range(1, dimY-1):
        x = image_gaussian[i,j]
        if (x > 0 and local_maximum(image_gaussian, i, j)):            
            local_max.add((i,j,radio_maximum(image_gaussian, i, j)))
                 
                    
cluster_points = cluster(local_max)
        
blob_centers = finding_center(cluster_points)
    
input_image, r_average = blob_drawing(name_file)

input_image.show()
input_image.save("output_images/"+name_file+"_blobs.jpg")


print "Number of BLOBS in "+ name_file + " is: " + str( len(blob_centers) )  
print "Radio average: " + str(r_average/len(blob_centers))