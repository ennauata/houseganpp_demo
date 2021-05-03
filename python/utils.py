#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
import networkx as nx
# import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor
import cv2
from torchvision.utils import save_image
import copy
import random
import webcolors
cv2.setNumThreads(0)
EXP_ID = random.randint(0, 1000000)

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}

def fix_nodes(real_mks, ind_fixed_nodes):
    given_masks = torch.tensor(real_mks)
    ind_not_fixed_nodes = torch.tensor([k for k in range(given_masks.shape[0]) if k not in ind_fixed_nodes])
    
    ## Set non fixed masks to -1.0
    given_masks[ind_not_fixed_nodes.long()] = -1.0
    given_masks = given_masks.unsqueeze(1)
    
    ## Add channel to indicate given nodes 
    inds_masks = torch.zeros_like(given_masks)
    inds_masks[ind_not_fixed_nodes.long()] = 0.0
    inds_masks[ind_fixed_nodes.long()] = 1.0
    given_masks = torch.cat([given_masks, inds_masks], 1)
    return given_masks

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGBA', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def remove_multiple_components(masks):
    new_masks = []
    drops = 0.0
    for mk in masks:
        m_cv = np.array(mk)
        m_cv[m_cv>0] = 255.0
        m_cv[m_cv<0] = 0.0
        m_cv = m_cv.astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255 , 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:  
            cnt_m = np.zeros_like(m_cv)
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(cnt_m, [c], 0, (255, 255, 255), -1)
            cnt_m[cnt_m>0] = 1.0
            cnt_m[cnt_m<0] = -1.0
            new_masks.append(cnt_m)
            drops += 1.0
        else:
            new_masks.append(mk)
    return new_masks, drops

def check_validity(masks):
    is_broken = False
    for mk in masks:
        m_cv = np.array(mk)
        m_cv[m_cv>0] = 255.0
        m_cv[m_cv<0] = 0.0
        m_cv = m_cv.astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255 , 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:  
            is_broken = True
            break
    return is_broken

def combine_images(samples_batch, nodes_batch, edges_batch, nd_to_sample, ed_to_sample):
    samples_batch = samples_batch.detach().cpu().numpy()
    nodes_batch = nodes_batch.detach().cpu().numpy()
    edges_batch = edges_batch.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    all_imgs = []
    shift = 0
    for b in range(batch_size):

        # split batch
        inds_nd = np.where(nd_to_sample==b)
        inds_ed = np.where(ed_to_sample==b)
        sps = samples_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]

        # draw
        _image = draw_masks(sps, nds)
        
        # store
        all_imgs.append(torch.FloatTensor(np.array(_image.convert('RGBA')).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
        shift += len(nds)
    return torch.stack(all_imgs)

def get_nxgraph(g_true):

    # build true graph
    G_true = nx.Graph()

    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label+1
        if _type >= 0 and _type not in [15, 17]:
            G_true.add_nodes_from([(k, {'label':k})])
           
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
   
    # add edges
    for k, m, l in g_true[1]:
        _type_k = g_true[0][k]  
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [15, 17] and _type_l not in [15, 17]):
            G_true.add_edges_from([(k, l)])
        elif m > 0 and (_type_k==15 or _type_l==15):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])  
            else:
                G_true.add_edges_from([(k, -1)])
    return G_true


def get_mistakes(masks, nodes, G_gt):

    masks, penalty = remove_multiple_components(masks.copy())

    # create graph
    masks = masks.copy()
    G_estimated = nx.Graph()
    DOOR_IN, DOOR_EX = 15, 17

    # add nodes
    for k, label in enumerate(nodes):
        _type = label
        if _type >= 0 and _type not in [DOOR_IN, DOOR_EX]:
            G_estimated.add_nodes_from([(k, {'label':k})])
   
    # add outside node
    G_estimated.add_nodes_from([(-1, {'label':-1})])
   
    # add node-to-door connections
    doors_inds = np.where((nodes == DOOR_IN) | (nodes == DOOR_EX))[0]
    rooms_inds = np.where((nodes != DOOR_IN) & (nodes != DOOR_EX))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:  
                m1, m2 = masks[k], masks[l]
                m1[m1>0] = 1.0
                m1[m1<=0] = 0.0
                m2[m2>0] = 1.0
                m2[m2<=0] = 0.0
                iou = np.logical_and(m1, m2).sum()/float(np.logical_or(m1, m2).sum())
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))    

    # draw connections            
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] != DOOR_IN:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                G_estimated.add_edges_from([(-1, l1)])

    # add missed edges
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])

    # add edges colors
    mistakes = 0
    per_node_mistakes = defaultdict(int)
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            mistakes += 1
            per_node_mistakes[k] += 1
            per_node_mistakes[l] += 1
           
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            mistakes += 1
            per_node_mistakes[k] += 1
            per_node_mistakes[l] += 1
    return mistakes+penalty
