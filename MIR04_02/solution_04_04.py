'''
Multimedia Retrieval Information
Homework 4, Exercise 4
Blobs

Goal: Developed a program for quantization an image

@author: Federico Raue
'''
import Image
from pylab import asarray
from scipy.cluster.vq import kmeans, vq


file_name = 'scarlett'
im = Image.open ('input_images/'+file_name+'.jpg')
image_array = (asarray(im))
image_histogram = image_array.copy()

#Histogram 4 x 4 x 4
image_histogram[(0 <= image_histogram) & (image_histogram <= 64)] = 32
image_histogram[(64 <= image_histogram) & (image_histogram < 128)] = 96
image_histogram[(128 <= image_histogram) & (image_histogram < 192)] = 168
image_histogram[(192 <= image_histogram) & (image_histogram <= 255)] = 223



Image.fromarray(image_histogram).show() 
Image.fromarray(image_histogram).save('output_images/'+file_name+'_hist.jpg')


#Vector quantized
image_vq = image_array.copy()

for cluster in [8, 16, 32, 64]:

    centroids_RGB, distortion_RGB = kmeans(image_array[:,:,:].ravel(), cluster);

    vq_RGB, vqdistortion_RGB = vq (image_array[:,:,:].ravel(), centroids_RGB)
  

    for i in range(len(image_array[:,:,:].ravel())):
        image_vq[:,:,:].ravel()[i] = centroids_RGB[vq_RGB[i]]

    Image.fromarray(image_vq).show()
    Image.fromarray(image_vq).save('output_images/'+file_name+'_vq_'+str(i)+'.jpg')

