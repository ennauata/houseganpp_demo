var network = null;
var seed = 2;

// label to color map
var room_to_color = {"living": '#EE4D4D', "kitchen": '#C67C7B', "bedroom": '#FFD274',
                     "bathroom": '#BEBEBE', "balcony": '#BFE3E8', "entrance": '#7BA779',
                     "dining": '#E87A90', "study": '#FF8C69', "storage": '#1F849B'};

// Template - Studio
var template_nodes_1 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bathroom", color: '#BEBEBE'},
  { id: 2, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
  { id: 4, label: "balcony", color: '#BFE3E8'},
];

var template_edges_1 = [
  { from: 0, to: 1 , color: '#D3A2C7', width: 3},
  { from: 2, to: 0 , color: '#D3A2C7', width: 3},
  { from: 2, to: 1 , color: '#D3A2C7', width: 3},
  { from: 2, to: 'outside' , color: '#D3A2C7', width: 3},
  { from: 2, to: 4 , color: '#D3A2C7', width: 3},
  { from: 0, to: 4 , color: '#D3A2C7', width: 3},
];

// Template - Two-bedroom suite
var template_nodes_2 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bedroom", color: '#FFD274'},
  { id: 2, label: "bathroom", color: '#BEBEBE'},
  { id: 3, label: "bathroom", color: '#BEBEBE'},
  { id: 4, label: "balcony", color: '#BFE3E8'},
  { id: 5, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
];

var template_edges_2 = [
  { from: 5, to: 1 , color: '#D3A2C7', width: 3},
  { from: 5, to: 2 , color: '#D3A2C7', width: 3},
  { from: 5, to: 3 , color: '#D3A2C7', width: 3},
  { from: 5, to: 4 , color: '#D3A2C7', width: 3},
  { from: 'outside', to: 5 , color: '#D3A2C7', width: 3},
  { from: 0, to: 2 , color: '#D3A2C7', width: 3},
  { from: 1, to: 3 , color: '#D3A2C7', width: 3},
];


// Template - Three-bedroom suite
var template_nodes_3 = [
  { id: 0, label: "bedroom", color: '#FFD274'},
  { id: 1, label: "bedroom", color: '#FFD274'},
  { id: 2, label: "bedroom", color: '#FFD274'},
  { id: 3, label: "bathroom", color: '#BEBEBE'},
  { id: 4, label: "bathroom", color: '#BEBEBE'},
  { id: 5, label: "kitchen", color: '#C67C7B'},
  { id: 6, label: "living", color: '#EE4D4D'},
  { id: 'outside', label: "outside", color: '#727171'},
  { id: 8, label: "balcony", color: '#BFE3E8'},
];

var template_edges_3 = [
  { from: 4, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 0 , color: '#D3A2C7', width: 3},
  { from: 6, to: 1 , color: '#D3A2C7', width: 3},
  { from: 6, to: 2 , color: '#D3A2C7', width: 3},
  { from: 6, to: 3 , color: '#D3A2C7', width: 3},
  { from: 0, to: 3 , color: '#D3A2C7', width: 3},
  { from: 1, to: 3 , color: '#D3A2C7', width: 3},
  { from: 6, to: 5 , color: '#D3A2C7', width: 3},
  { from: 'outside', to: 6 , color: '#D3A2C7', width: 3},
  { from: 8, to: 2 , color: '#D3A2C7', width: 3},
];
// create a network
var container = document.getElementById("mynetwork");

// legend
function add_legend(data) {
  var mynetwork = document.getElementById("mynetwork");
  var x = -mynetwork.clientWidth / 2 + 50;
  var y = -mynetwork.clientHeight / 2 - 50;
  var step = 60;
  var i = 0;
  for (var key in room_to_color) {
    // console.log("legend_" + key);
      data.nodes.add({
              x: x + i*step,
              y: y,
              id: "legend_" + key,
              label: key,
              color: room_to_color[key],
              group: key,
              shape: "square",
              value: 1,
              fixed: true,
              physics: false,
            });
      i++;
    }
  }

function destroy() {
  if (network !== null) {
    network.destroy();
    network = null;
  }
}

function draw(data) {
  destroy();

  // create a network
  var container = document.getElementById("mynetwork");
  var options = {
    nodes: { borderWidth: 2 },
    layout: { randomSeed: seed },
    locales: {
      en: {
        edit: 'Edit',
        del: 'Delete selected',
        back: 'Back',
        addNode: 'Add Node',
        addEdge: 'Add Edge',
        editNode: 'Edit Node',
        editEdge: 'Edit Edge',
        addDescription: 'Select a room type from the legend.',
        edgeDescription: 'Click on a node and drag the edge to another node to connect them.',
        editEdgeDescription: 'Click on the control points and drag them to a node to connect to it.',
        createEdgeError: 'Cannot link edges to a cluster.',
        deleteClusterError: 'Clusters cannot be deleted.',
        editClusterError: 'Clusters cannot be edited.'
      }
    },
    manipulation: {
      addNode: function (nodeData, callback) {
        // filling in the popup DOM elements
        if (network.getSelectedNodes().length > 0){
          selectedNode = network.getSelectedNodes()[0]
          var nodesLength = data.nodes.getIds().length;
          if (String(selectedNode).includes('legend') && nodesLength <= 22){
            editNode(nodeData, selectedNode, clearNodePopUp, callback);
          }
          else if(nodesLength > 22){
            alert("Err: trying to add too many nodes.");
            callback(null);
          }
          else {
            alert("Err: node must be selected from the legend.");
            callback(null);
          }
        }
      },
      deleteNode: function(data, callback) {
        if (String(data.nodes[0]).includes('legend_') == true || String(data.nodes[0]).includes('outside') == true){
          alert("Err: this node can not be deleted.");
          callback(null);
        }
        else{
          callback(data);
          generate(); 
        }
      },
      addEdge: function (data, callback) {
        if (data.from == data.to) {
          alert("Err: trying to add self connections.");
          callback(null);
          return;
        }
        document.getElementById("edge-operation").innerText = "Add Edge";
        saveEdgeData(data, callback);
      },
      deleteEdge: function(data, callback) {
        callback(data);
        generate();
      },
    },
  };
  network = new vis.Network(container, data, options);
  network.fit()
}

function editNode(data, label, cancelAction, callback) {
  var mynetwork = document.getElementById("mynetwork");
  var x = -mynetwork.clientWidth / 2 + 200;
  var y = -mynetwork.clientHeight / 2 + 200;
  data.x = x;
  data.y = y;
  data.label = label.split("_")[1];
  data.color = room_to_color[data.label]
  clearNodePopUp();
  callback(data);
  generate();
}

// Callback passed as parameter is ignored
function clearNodePopUp() {
  document.getElementById("node-saveButton").onclick = null;
  document.getElementById("node-cancelButton").onclick = null;
  document.getElementById("node-popUp").style.display = "none";
}

function cancelNodeEdit(callback) {
  clearNodePopUp();
  callback(null);
}

function saveEdgeData(data, callback) {
  if (typeof data.to === "object") data.to = data.to.id;
  if (typeof data.from === "object") data.from = data.from.id;
  data.color = '#D3A2C7';
  data.width = 3;
  callback(data);
  generate();
}

function checker_reset(k){
  $(".check"+k.toString()).attr("class", "check"+k.toString());
  $(".fill"+k.toString()).attr("class", "fill"+k.toString());
  $(".path"+k.toString()).attr("class", "path"+k.toString());
}

function checker_complete(k){
  $(".check"+k.toString()).attr("class", "check"+k.toString()+" check-complete"+k.toString()+" success");
  $(".fill"+k.toString()).attr("class", "fill"+k.toString()+" fill-complete"+k.toString()+" success");
  $(".path"+k.toString()).attr("class", "path"+k.toString()+" path-complete"+k.toString());
}

function generate() {

  // start checker
  for (var i = 0; i < 6; i++){
    checker_reset(i);
  }
  // get current graph
  var nodeIndices = data.nodes.getIds();
  var edgesIndices = data.edges.getIds();
  var nodes = new Array();
  var types = new Object();
  var edges = new Array();
  var edgesObj = new Array();
  for (var i = 0; i < nodeIndices.length; i++) {
      if (nodeIndices[i].toString().includes("legend_") == false){
        nodes.push(nodeIndices[i].toString());
        types[i.toString()] = data.nodes.get(nodeIndices[i]).label;
      }
  }
  for (var i = 0; i < edgesIndices.length; i++) {
    edgesObj.push(data.edges.get(edgesIndices[i]));
  }
  for (var i = 0; i < nodes.length; i++) {
    for (var j = 0; j < nodes.length; j++) {
      if (j < i){
        for (var k = 0; k < edgesObj.length; k++) {
          if ((edgesObj[k].from == nodes[i] && edgesObj[k].to == nodes[j])||(edgesObj[k].from == nodes[j] && edgesObj[k].to == nodes[i])){
            edges.push([i, j]);
          }
        }
      }
    }
  }
  graph_info = ({"nodes":types, "edges":edges});

  // generate layout
  var xhr = new XMLHttpRequest();
  var _ptr = 0;
  var num_iters = 1;
  var n_samples = 6;
  var _tracker = 0;
  xhr.onreadystatechange = function () {
      if (this.status === 200) {
          var data_stream = this.responseText;
          data_stream = data_stream.split('<stop>');
          for (var i = _tracker; i < data_stream.length; i++){
            image_data = data_stream[i];

            // populate
            if(Math.floor(_ptr/num_iters) == 0 && image_data != "" && _ptr < n_samples*num_iters){
              var svg_element = document.getElementById("lgContainer");
              lgContainer.innerHTML = image_data;
              var svg = lgContainer.firstChild;
              scaleSVG(svg, 2.2);
              var progress = document.getElementById("progress");
              progress.innerHTML = "Sample Generated House Layout"
              _ptr++;
              if(Math.floor(_ptr/num_iters) > 0){
                checker_complete(0);
              }
            }
            else if (image_data != "" && _ptr < n_samples*num_iters){
              var smContainer = document.getElementById("sm_img_"+(Math.floor(_ptr/num_iters)-1));
              smContainer.innerHTML = image_data;
              smContainer.onclick = function() {
                miniDisplay = document.getElementById($(this).attr('id'));
                miniDisplaySVG = miniDisplay.firstChild;
                largeDisplay = document.getElementById("lgContainer");
                largeDisplaySVG = largeDisplay.firstChild;
                var s = new XMLSerializer();
                miniDisplay.innerHTML = s.serializeToString(largeDisplaySVG);
                scaleSVG(miniDisplay.firstChild, 1.0/2.2);
                scaleSVG(miniDisplay.firstChild, 0.49);
                largeDisplay.innerHTML = s.serializeToString(miniDisplaySVG);
                scaleSVG(largeDisplay.firstChild, 1.0/0.49);
                scaleSVG(largeDisplay.firstChild, 2.2);
              }
              var svg = smContainer.firstChild;
              scaleSVG(svg, 0.49);
              checker_complete(Math.floor(_ptr/num_iters)-1);
              _ptr++;
            }
          }
          _tracker += (data_stream.length-_tracker)
          if (_tracker == data_stream.length){
            checker_complete(Math.floor(_ptr/num_iters)-1);
          }
      }
  }

  // xhr.open("POST", 'http://localhost:5000/generate', true);
  xhr.open("POST", 'http://houseganpp.com/generate', true);
  xhr.setRequestHeader('Content-Type', 'text/plain');
  xhr.send(JSON.stringify(graph_info));
}

function scaleSVG(svg, factor){
  var svgWidth = parseFloat(svg.getAttributeNS(null, "width"));
  var svgHeight = parseFloat(svg.getAttributeNS(null, "height"));
  // console.log(svgWidth, svgHeight);
  svg.setAttributeNS(null, "width", svgWidth*factor);
  svg.setAttributeNS(null, "height", svgHeight*factor);
  svg.setAttributeNS(null, "viewBox", "0 0 " + svgWidth + " " + svgHeight); 
}

function setTemplate(id){
  data = get_data_object(id);
  add_legend(data);
  draw(data);
  generate();
}

function get_data_object(template_id) {
  var data = new Object();
  if (template_id==0){
    data.nodes = new vis.DataSet(template_nodes_1);
    data.edges = new vis.DataSet(template_edges_1);
  } 
  else if (template_id==1){
    data.nodes = new vis.DataSet(template_nodes_2);
    data.edges = new vis.DataSet(template_edges_2);
  }
  else if (template_id==2){
    data.nodes = new vis.DataSet(template_nodes_3);
    data.edges = new vis.DataSet(template_edges_3);
  }

  return data;
}

window.addEventListener("load", () => {
  var defaultTemplate = 1;
  setTemplate(defaultTemplate);
});
