import numpy as np
from python.viz import draw_graph, draw_masks
import matplotlib.pyplot as plt
import torch
from python.models_new import Generator
from PIL import Image
import base64
from io import BytesIO
import json
import sys
import time
from python.utils import fix_nodes, check_validity, get_nxgraph, get_mistakes, remove_multiple_components

# enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# Rooms
ROOM_CLASS = {"living": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining": 7, "study": 8,
			  "storage": 10 , "front": 15, "unknown": 16, "interior": 17}

def parse_json(json):
	nds, eds = [], []
	rooms, doors = dict(json["nodes"]), list(json["edges"])
	# convert to tuples
	doors = [tuple(d) for d in doors]
	front_door_ind = -1
	# handle nodes
	for k, n in enumerate(rooms):
		if "outside" in rooms[n]:
			nds.append(ROOM_CLASS['front']) 
			front_door_ind = k
		else:
			nds.append(ROOM_CLASS[rooms[n]])

	to_add = []
	for k, l in doors:
		if k != front_door_ind and l != front_door_ind:
			nds.append(ROOM_CLASS['interior'])
			to_add.append((k, len(nds)-1))
			to_add.append((l, len(nds)-1))
	doors += to_add

	# handle edges        
	for k in range(len(nds)):
		for l in range(len(nds)):
			if k < l:
				if ((k, l) in doors) or ((l, k) in doors):
					eds.append([k, 1, l])
				else:
					eds.append([k, -1, l])

	return np.array(nds), np.array(eds)

def load_model(checkpoint='python/pretrained_new.pth'):
	model = Generator()
	model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=True)
	return model.eval()

def one_hot_embedding(labels, num_classes=19):
	y = torch.eye(num_classes) 
	return y[labels] 

def _init_input(graph, prev_state=None, mask_size=64):

	# initialize graph
	nds, eds = graph
	given_nds = one_hot_embedding(nds)[:, 1:].float()
	given_eds = torch.tensor(eds).long()
	z = torch.randn(len(nds), 128).float()

	# unpack
	fixed_nodes = prev_state['fixed_nodes']
	prev_mks = torch.zeros((given_nds.shape[0], mask_size, mask_size))-1.0 if (prev_state['masks'] is None) else prev_state['masks']

	# # threshold masks -- not a game changer
	# prev_mks[prev_mks<0] = -1.0
	# prev_mks[prev_mks>=0] = 1.0

	# initialize masks
	given_masks_in = fix_nodes(prev_mks, torch.tensor(fixed_nodes))

	return z, given_masks_in, given_nds, given_eds

def _infer(graph, model, prev_state=None, device='cpu'):

	# Init
	z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)

	# Run pytorch
	with torch.no_grad():
		masks = model(z.to(device).float(), given_masks_in.to(device).float(), given_nds.to(device).float(), given_eds.to(device))
		masks = masks.detach().float().cpu().numpy()
	return masks

def run_model(graph_data):

	# parse json
	fp_graph = parse_json(graph_data)
	G_gt = get_nxgraph(fp_graph)

	# block malicious requests
	if len(fp_graph[0]) > 400 or len(fp_graph[1]) > 1200: # set max of 20 fully connected rooms 
		return "Err"

	# create image
	all_images = []

	# run inference
	start_time = time.time()
	device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
	model = load_model().to(device)
	model.zero_grad(set_to_none=True)
	# model.half()

	print("load model: --- %s seconds ---" % (time.time() - start_time))
	# add room types incrementally
	real_nodes = fp_graph[0]
	all_types = sorted(list(set(real_nodes)))
	selected_types = [all_types[:k+1] for k in range(50)]
	# selected_types = [[9, 15], [9, 15], [9, 15], [3, 9, 15], [1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 5, 9, 15], [0, 1, 2, 3, 5, 6, 9, 15], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # best for exp_random_types_attempt_3_A_500000_G - FID
	# selected_types += [_types for k in range(90)]
	for k in range(6):
		state = {'masks': None, 'fixed_nodes': []}
		masks = _infer(fp_graph, model, state, device)
		_tracker = (get_mistakes(masks.copy(), real_nodes, G_gt), masks)
		for l, _types in enumerate(selected_types):

			# refinement step
			start_time = time.time()
			_fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) if len(_types) > 0 else np.array([])
			state = {'masks': masks, 'fixed_nodes': _fixed_nds}
			masks = _infer(fp_graph, model, state, device)

			# track score
			score = get_mistakes(masks.copy(), real_nodes, G_gt)
			if score <= _tracker[0]:
				_tracker = (score, masks.copy())
			
			# reset layout
			if l%5 == 0 and l>0:
				state = {'masks': None, 'fixed_nodes': []}
				masks = _infer(fp_graph, model, state, device)

			# greed search -- found best solution
			if _tracker[0] == 0:
				break

		# if last round send the best one
		masks = _tracker[1]
		masks, _ = remove_multiple_components(masks)

		print("runtime: --- %s seconds ---" % (time.time() - start_time))
		print("Using GPU:", torch.cuda.is_available(), "CUDNN Version:",  torch.backends.cudnn.version())
		print("Search score {}".format(_tracker[0]))

		# send masks
		im_svg = draw_masks(masks.copy(), real_nodes, im_size=256)
		yield '<stop>' + str(im_svg)
