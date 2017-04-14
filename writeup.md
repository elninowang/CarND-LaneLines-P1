# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

**my english is not good, but i will do my best**

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

![The original image][./examples/image.jpg]

First, I converted the images to grayscale. 
![The grayscale image][./examples/gray.jpg]

Then I use gaussian blur to the images. 
![The gaussian grayscale image][./examples/blur_gray.jpg]

Then I use Canny edge detector on this image.
![The edges image][./examples/edges.jpg]

Then I use the mask to find the region of interest.
![The edges image with interest][./examples/masked_edges.jpg]

Then i use Hough Transform to find the lines on the images. 
![The lines image][./examples/line_image.jpg]

At last, i draw lines. <br />
![The final image][./examples/final_image.jpg]

that's pipeline all 


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by below:

First, i caculate the slope of each lines, in this step i filter out some lines do not need to consider(less than 30 degrees or greater than 60 degrees), and then spilte lines to 2 class, one is left line, and the other is right line by whether the slope is greater than zero. Then i calculate the left or right lane of the weighted average slope .(Weight is the square of the distance, Because the longer the line means that the more important) the i use the weighted average slope to filter out the line which slope is much difference. the rest lines is all my interest. Then I calculate the average center position x and y of all line in lefe lane class. the use the average center position and the slope to drawline form bottom to y bottom-210 use the former y = kx + b. that's all. 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][./examples/grayscale.jpg]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen:

- The lane line is not clear, on and off, for example at night or when it rains
- The median line also same length as the lane line
- The narrator cars may interfere with the calculation
- Steep uphill and downhill


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...
Interested in a more precise area, such as analysis of commonly used possible position, left and right lane to find out the possible area, so we can rule out other distractions

Another potential improvement could be to ...
Find possible line, in the color of the reference line, the lane line is usually yellow and white, other colors can be regarded as interference, the need to rule out
