import argparse
import os
import numpy as np
import math
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
# import matplotlib.pyplot as plt
# import networkx as nx
import glob
import cv2
import webcolors
import time
import svgwrite

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label
        if _type >= 0 and _type not in [15, 17]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
            
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    
    # add edges
    for k, m, l in g_true[1]:
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [15, 17] and _type_l not in [15, 17]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==15 or _type_l==15) and (_type_l!=17 and _type_k!=17):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])   
            elif _type_l==15:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
    nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
            font_weight='bold', edgecolors=edgecolors, edge_color=edge_color, width=4.0, with_labels=False)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    plt.close('all')
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return G_true, rgb_im

def _snap(polygons, ths=[2, 4]):
    polygons = list(polygons)
    cs = np.concatenate([np.stack(p) for ps in polygons for p in ps], 0).reshape(-1, 2)
    new_cs = np.array(cs)
    for th in ths:
        for i in range(len(new_cs)):
            x0, y0 = new_cs[i]
            x0_avg, y0_avg = [], []
            tracker = []
            for j in range(len(new_cs)):
                x1, y1 = new_cs[j]

                # horizontals
                if abs(x1-x0) <= th:
                    x0_avg.append(x1)
                    tracker.append((j, 0))
                # verticals
                if abs(y1-y0) <= th:
                    y0_avg.append(y1)
                    tracker.append((j, 1))
            avg_vec = [np.mean(x0_avg), np.mean(y0_avg)]

            # set others
            for k, m in tracker:
                new_cs[k, m] = avg_vec[m]

    # create map to new corners
    c_map = {}
    for c, new_c in zip(cs, new_cs):
        c_map[tuple(c)] = tuple(new_c)

    # update polygons
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            for k in range(len(polygons[i][j])):
                xy = polygons[i][j][k][0]
                polygons[i][j][k][0] = np.array(c_map[tuple(xy)])
    return polygons

def _fix(contours):

    # fill old contours
    m_cv = np.full((256, 256, 1), 0).astype('uint8')
    cv2.fillPoly(m_cv, pts=contours, color=(255, 255, 255))

    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_cv = cv2.erode(m_cv, kernel)
    
    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_cv = cv2.dilate(m_cv, kernel)

    # get new contours
    ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
    new_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return new_contours

def _assign_door(contours, rooms):

    # define doors
    horizontal_door = np.array([[[-8, -2]], [[8, -2]], [[8, 2]], [[-8, 2]]]) # (width, height)
    vertical_door = np.array([[[-2, -8]], [[2, -8]], [[2, 8]], [[-2, 8]]]) # (width, height)
    unit = np.array([[[-1, -1]], [[1, -1]], [[1, 1]], [[-1, 1]]]) # debug

    # assign door to room
    door = np.concatenate(contours, 0)[:, 0, :]
    door_mean = np.mean(door, 0)
    rooms = np.concatenate([r for rs in rooms for r in rs], 0)[:, 0, :]
    dist = np.sum((rooms - door_mean)**2, axis=1)
    idx = np.argmin(dist)
    pt = rooms[idx]

    # determine vertical/horizontal
    wh = np.max(door, 0)-np.min(door, 0)
    if wh[0] > wh[1]: # horizontal
        new_door = horizontal_door + np.array([door_mean[0], pt[1]]).astype('int')
    else: # vertical
        new_door = vertical_door + np.array([pt[0], door_mean[1]]).astype('int')
    return new_door

def _draw_polygon(dwg, contours, color, with_stroke=True):
    for c in contours:
        pts = [(float(c[0]), float(c[1])) for c in c[:, 0, :]]
        if with_stroke:
            dwg.add(dwg.polygon(pts, stroke='black', stroke_width=4, fill=color))
        else:
            dwg.add(dwg.polygon(pts, stroke='none', fill=color))
    return

def draw_masks(masks, real_nodes, im_size=256):
    
    # process polygons
    polygons = []
    for m, nd in zip(masks, real_nodes):
        # resize map
        m[m>0] = 255
        m[m<0] = 0
        m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_NEAREST) 

        # extract contour
        m_cv = m_lg[:, :, np.newaxis].astype('uint8')
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) > 0]
        polygons.append(contours)
    polygons = _snap(polygons)

    # draw rooms polygons
    dwg = svgwrite.Drawing('./floorplan.svg', (256, 256))
    rooms = []
    for nd, contours in zip(real_nodes, polygons):
        # pick color
        color = ID_COLOR[nd]
        r, g, b = webcolors.hex_to_rgb(color)
        if nd not in [15, 17]:
            new_contours = _fix(contours) 
            new_contours = [c for c in new_contours if cv2.contourArea(c) >= 4] # filter out small contours
            _draw_polygon(dwg, new_contours, color)
            rooms.append(new_contours)

    # draw doors
    for nd, contours in zip(real_nodes, polygons):
        # pick color
        color = ID_COLOR[nd]
        if nd in [15, 17] and len(contours) > 0:
            contour = _assign_door([contours[0]], rooms)
            _draw_polygon(dwg, [contour], color, with_stroke=False)
    return dwg.tostring()

## OLD CODE -- BACKUP
# def draw_masks(masks, real_nodes, im_size=256):
    
#     # process polygons
#     polygons = []
#     for m, nd in zip(masks, real_nodes):
#         # resize map
#         m[m>0] = 255
#         m[m<0] = 0
#         m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_NEAREST) 

#         # extract contour
#         m_cv = m_lg[:, :, np.newaxis].astype('uint8')
#         ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if len(c) > 0]
#         polygons.append(contours)
#     polygons = _snap(polygons)

#     # draw rooms polygons
#     dwg = svgwrite.Drawing('./floorplan.svg', (256, 256))
#     bg_img = np.full((256, 256, 3), 255).astype('uint8')
#     rooms = []
#     for nd, contours in zip(real_nodes, polygons):
#         # pick color
#         color = ID_COLOR[nd]
#         r, g, b = webcolors.hex_to_rgb(color)
#         if nd not in [15, 17]:
#             new_contours = _fix(contours) 
#             new_contours = [c for c in new_contours if cv2.contourArea(c) >= 4] # filter out small contours
#             cv2.fillPoly(bg_img, pts=new_contours, color=(r, g, b, 255))
#             cv2.drawContours(bg_img, new_contours, -1, (0, 0, 0, 255), 2)
#             _draw_polygon(dwg, new_contours, color)
#             rooms.append(new_contours)

#     # draw doors
#     for nd, contours in zip(real_nodes, polygons):
#         if nd in [15, 17] and len(contours) > 0:
#             # cv2.fillPoly(bg_img, pts=[contours[0]], color=(0, g, 0, 255))
#             contour = _assign_door([contours[0]], rooms)
#             cv2.fillPoly(bg_img, pts=[contour], color=(r, g, b, 255))
#             _draw_polygon(dwg, [contour], color, with_stroke=False)
#     bg_img = Image.fromarray(bg_img)
#     return dwg.tostring()