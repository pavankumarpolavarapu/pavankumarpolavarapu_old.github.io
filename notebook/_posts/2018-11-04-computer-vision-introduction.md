---
layout: post
title: "Computer Vision Introduction"
date:   2018-11-04
category: notebook
comments: true
math: true
author: "Pavan Kumar Polavarapu"
---

>Recently, I have started following CS231n MIT Course and the first week class is about introduction to computer vision and here I am taking my notes.

543 Million Years ago, earth is filled with water with animals floating on top of water with no vision but around 540 million years ago within a very short period of time (10 million years), the number of animal species exploded from few to hundreds of thousands.  

Evolutionary Biologists call this s Evolution's Big Bang and it is theoretically it's because of first animal developing eyes. Now vision developed into biggest sensory system. 50% of neurons in our cortex involved in visual processing.

#### Humans & Mechanical Vision
<ul>
<li>First Camera - "Camera Obscura" developed by Gemma Frisius in 1545</li>
<li>Encyclopedie, 18th Century</li>
<li>Leonardo da vinci, 16th Century</li>
</ul>

#### Hubel & Wiesel, 1959
Cat Brain Analysis
<ul>
<li>Simple Cells - Response to light orientation</li>
<li>Complex Cells - Response to light orientation and movement </li>
<li>Hyper Complex Cells - Response to movement with an end point</li>
</ul>

Basically visualization in brain starts with the edges a.k.a simple structures and builds on complex information.

#### MIT Summer Vision Project, 1966
Aimed at developing significant part of our visual system

#### David Marr - Vision, 1970's
Input Image -> Primal Sketch -> 2-1/2D Sketch -> 3D Model Sketch

**Primal Sketch** - Zero Crossings, blobs, edges, bars, ends, virtual lines, groups, curves, boundaries.

**2-1/2D Sketch** - Local surface orientation and discontinuities in depth and in surface orientation.

**3D Model** - Hierarchically organized in terms of surface and volumetric primitives.


#### Generalized Cylinder - Brooks & Binford, 1979
Wanted to move from block world to real world in recognizing things
<img src="/assets/notebook-images/vision/generalizedcylinder.png" alt="Generalized Cylinder">

#### Pictorial Structure - Fischler & Elschlager, 1973
Tried reducing complex information into simple structures
<img src="/assets/notebook-images/vision/pictorialstructure.png" alt="Pictorial Structure">


All these works are insignificant compared to the task at hand. If object recognition is too hard, object segmentation could be worked.

#### Normalized Cut - Shi & Malik, 1997
Graph theory algorithm for segmentation.

### Face Detection - Viola & Jones, 2001
1999 - 2000 Machine Learning Techniques
<ul>
<li> Statistic Machine Learning</li>
<li>Support Vector Machines</li>
<li>Boosting</li>
<li>Graphical Models</li>
</ul>

After 5 years, in 2006, Fuji Film released first camera that can detect face.

#### "SIFT" & Object detection - David Lowe, 1999 
Feature based detection

#### Spatial Pyramid Matching - Lazebnik, Schimd & Ponce - 2006
To recognize landscape holistically

#### Histogram of Gradients - Dalal & Triggs, 2005

#### Deformable Part Model - Felzenswalb, Mc Allester, Ramanan, 2009

#### Pascal Visual Object Challenge (2006 - 2012)

Mean Precision improved from around 25% to 500% between 2006 & 2012

#### Imagenet Large Scale Visual Recognition Challenge

Machine learning models tend to overfit if we do not give ample data as input to train the models and hence image-net and word-net formed, a crowd sourcing approach to gather data.

In 2012, Convolutional Neural Network Model (Deep Learning).