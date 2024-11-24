---
layout: post
title:  "3D Computer Vision Study Notes - 01"
date:   2024-11-18 16:18:12 -0700
categories: 3D Vision, Computer Vision
---

## Useful Links:
- [Visual SLAM Roadmap](https://github.com/changh95/visual-slam-roadmap)
- [First Principles of Computer Vision](https://fpcv.cs.columbia.edu/)
- CMU Course: [16-825: Learning for 3D Vision](https://learning3d.github.io/schedule.html)
- [An Invitation to 3D Vision: A Tutorial for Everyone](https://github.com/mint-lab/3dv_tutorial)
- 3D Vision (UIUC), [3D Vision (CS 598) – Fall 2021](https://courses.grainger.illinois.edu/cs598dwh/fa2021/)
- Multiview 3D Geometry in Computer Vision (UMN), [Spring 2018 CSCI 5980 Multiview 3D Geometry in Computer Vision](https://www-users.cse.umn.edu/~hspark/CSci5980/csci5980_3dvision.html)

## Topics in 3D Vision:
- 3D Reconstruction: SfM/SLAM, Multi-view Stereo, RGB-D Fusion.
- 3D Scene Understanding: plane, normal, depth
- Depth Estimation, Room Layout Estimation, NeRFs, Inverse Rendering, 
- Computer Graphics and Computational Photography

Also includes:
- Representation Learning
- Image and Video Synthesis
- Vision for Robotics and Autonomous Vehicles


## Introduction

### 3D Scene Understanding

- Task: Holistic reasoning of everything that is in the scene
- Involves many semantic tasks: Semantic Segmentation, Instance-level segmentation, Segmentation of the scene into semantic labels.
- Tracking

For example, from a short (≈10s) video sequence to infer:
- Geometric properties, e.g., street orientation
- Topological properties, e.g., number of intersecting streets
- Semantic activities, e.g., traffic situations at an intersection
- 3D objects, e.g., cars
