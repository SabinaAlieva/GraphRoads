from lxml import etree, objectify
import matplotlib.pyplot as plt
import numpy as np
import csv

FILENAME = "map.xml"

with open(FILENAME, "rb") as f:
    xml = f.read()
    
root = objectify.fromstring(xml)

road = []           
roads = []          

way = root.way
for row in way:     
    tag = row.find("tag")
    if not (tag is None) and tag.attrib['k'] == 'highway':
        nd = row.nd          
        for ref in nd:       
            road.append(int(ref.attrib['ref']))
        roads.append(road)
        road = []
        
node = root.node             

nodes = {int(row.attrib['id']): [float(row.attrib['lat']), float(row.attrib['lon'])] for row in node}

count_node_in_roads = {int(row.attrib['id']): 0 for row in node}

for road in roads:
    for nd in road:
        count_node_in_roads[int(nd)] = count_node_in_roads[int(nd)] + 1
        
count_edge_in_roads = {int(row.attrib['id']): 0 for row in node}
for road in roads:
    for i in range(len(road) - 1):
        count_edge_in_roads[road[i]] = count_edge_in_roads[road[i]] + 1
        count_edge_in_roads[road[i + 1]] = count_edge_in_roads[road[i + 1]] + 1
        
nodes_to_delete = []
for nd in nodes:
    if count_node_in_roads[nd] == 0 or (count_node_in_roads[int(nd)] == 1 and count_edge_in_roads[int(nd)] == 2):
        nodes_to_delete.append(nd)
    
for nd in nodes_to_delete:
    nodes.pop(nd)
    
list_adj = {nd:[] for nd in nodes}

row = []

for road in roads:
    for nd in road:
        if nd in nodes:        
            row.append(nd)
    for i in range(len(row) - 1):   
        list_adj[row[i]].append(int(row[i+1]))
        list_adj[row[i+1]].append(int(row[i]))
    row = []
    
for key in list_adj:
    list_adj[key] = list(set(list_adj[key]))
    
line_csv = []
with open("list_taganrog.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for key in list_adj:
        line_csv.append(key)
        for r in list_adj[key]:
            line_csv.append(r)
        writer.writerow(line_csv)
        line_csv = []
        
hospitals = {}

for nd in node:     
    tag = nd.find("tag")
    if not (tag is None) and tag.attrib['v'] == 'hospital':
        hospitals[int(nd.attrib['id'])] = [float(nd.attrib['lat']), float(nd.attrib['lon'])]
        
import math

target = []
for h in hospitals:
    min_w = 10000
    nd_min = 0
    for nd in nodes:
        w = math.sqrt((nodes.get(nd)[0] - hospitals.get(h)[0])**2 + (nodes.get(nd)[1] - hospitals.get(h)[1])**2)
        if w < min_w:
            min_w = w
            nd_mai = nd
    target.append(nd_mai)

lat = 47.261885 
lon = 38.908703

min_w = 1000000
for nd in nodes:
    w = math.sqrt((nodes.get(nd)[0] - lat)**2 + (nodes.get(nd)[1] - lon)**2)
    if w < min_w:
        min_w = w
        start = nd
        
from heapq import heappush
from heapq import heappop 

def Dejicstra(a, b):
    
    len_sh_way = {nd: 1000000 for nd in nodes}  
    len_sh_way.update({a: 0})
    
    shortest_way = {nd: [] for nd in nodes} 
    shortest_way.update({a: [a]})
    
    current_node = a
    
    nd_to_visit = []                 
    deleted = []                    
    nd_to_visit.append(a)
    
    sort_len = []  
    
    while(1):
        if current_node != b:   
            
            sort_len = [len_sh_way.get(nd) for nd in nd_to_visit]  
            sort_len.sort()            
                     
            for key in nd_to_visit: 
                if len_sh_way.get(key) == sort_len[0]: 
                    current_node = key              
                    deleted.append(key)            
                    break
            
            nd_to_vis = list(tuple(list_adj.get(current_node))) 
            
            for nd in nd_to_vis:           
                nd_to_visit.append(nd)
            nd_to_visit = list(set(nd_to_visit)) 
                
            for nd in deleted:      
                if nd_to_visit.count(nd) > 0:
                    nd_to_visit.remove(nd)    
            
            for u in nd_to_visit:         
                weight = math.sqrt((nodes.get(u)[1] + nodes.get(current_node)[1])**2 + (nodes.get(u)[0] + nodes.get(current_node)[0])**2)
                
                if len_sh_way.get(u) > len_sh_way.get(current_node) + weight:         
                
                    w = len_sh_way.get(current_node) + weight
                    len_sh_way.update({u: w}) 
                                   
                    mas = list(tuple(shortest_way.get(current_node)))
                    mas.append(u)
                    m = list(tuple(mas))
                    shortest_way.update({u: m}) 
                    
            if len(nd_to_visit) == 0:
                return shortest_way.get(b)
                break
        
        else: 
            break
        
    return (shortest_way.get(b))



from collections import deque

def Levita(a, b):
    len_sh_way = {nd: 1000000 for nd in nodes} 
    len_sh_way.update({a: 0})
    
    shortest_way = {nd: [] for nd in nodes} 
    shortest_way.update({a: [a]})
    
    nodes_already_calc = []
    common_queue = deque((a,))       
    rush_queue = deque()         
    
    nodes_not_calc = [nd for nd in nodes] 
    nodes_not_calc.remove(a)
    
    cur_node = a
    
    while(len(common_queue) > 0 or len(rush_queue) > 0):
        if len(rush_queue) > 0:
            current_node = rush_queue.popleft()
        else:
            cur_node = common_queue.popleft()
             
        adj = list_adj.get(cur_node)
        
        for nd in adj:
            w = math.sqrt((nodes.get(nd)[0] - nodes.get(cur_node)[0])**2 + (nodes.get(nd)[1] - nodes.get(cur_node)[1])**2)
           
            if nodes_not_calc.count(nd) > 0:
                common_queue.append(nd)
                nodes_not_calc.remove(nd)
                
                if len_sh_way.get(nd) > len_sh_way.get(cur_node) + w:
                    
                    len_sh_way.update({nd: len_sh_way.get(cur_node) + w})
                    way = shortest_way.get(cur_node)[:]
                    way.append(nd)
                    shortest_way.update({nd: way})
                          
                
            elif common_queue.count(nd) > 0 or rush_queue.count(nd) > 0:
                if len_sh_way.get(nd) > len_sh_way.get(cur_node) + w:
                    
                    len_sh_way.update({nd: len_sh_way.get(cur_node) + w})
                    way = shortest_way.get(cur_node)[:]
                    way.append(nd)
                    shortest_way.update({nd: way})
                    
            elif nodes_already_calc.count(nd) > 0 and len_sh_way.get(nd) > len_sh_way.get(cur_node) + w:
                
                rush_queue.append(nd)
                nodes_already_calc.remove(nd)
                
                len_sh_way.update({nd: len_sh_way.get(cur_node) + w})
                way = shortest_way.get(cur_node)[:]
                way.append(nd)
                shortest_way.update({nd: way})        
                          
        nodes_already_calc.append(cur_node)     
                          
    return(shortest_way.get(b))


import networkx as nx

GG = nx.Graph()

for key in nodes:
    GG.add_node(str(key),pos=tuple(nodes.get(key)))

for nd in list_adj:
    for nod in list_adj.get(nd):
        w = math.sqrt((nodes.get(nd)[1] + nodes.get(nod)[1])**2 + (nodes.get(nd)[0] + nodes.get(nod)[0])**2)
        GG.add_edge(str(nd),str(nod),weight=w)
        
nodesi = {str(key): [nodes.get(key)[1], nodes.get(key)[0]] for key in nodes}

import math

def ManhattanDist(start, finish):
    return abs(nodes.get(float(start))[0]-nodes.get(float(finish))[0])/2 + abs(nodes.get(float(start))[1]-nodes.get(float(finish))[1])/2

def ChebDist(start, finish):
    return max(abs(nodes.get(float(start))[0]-nodes.get(float(finish))[0]), abs(nodes.get(float(start))[1]-nodes.get(float(finish))[1]))

def EuklidDist(start, finish):
    return math.sqrt((nodes.get(float(start))[0]-nodes.get(float(finish))[0])**2 + (nodes.get(float(start))[1]-nodes.get(float(finish))[1])**2)

import time

def timeDejicstra(start, target):
    start_time = time.time()
    Dejicstra = nx.dijkstra_path(GG, str(start), str(target))
    return time.time() - start_time
    #print("--- %s seconds ---" % (time.time() - start_time))
    #drawShortestWay(GG, Dejicstra,start, target[1])

def timeAstarManh(start, target):
    start_time = time.time()
    Astar_Manh = nx.astar_path(GG, str(start), str(target), ManhattanDist)
    return time.time() - start_time
    #print("--- %s seconds ---" % (time.time() - start_time))
    #drawShortestWay(GG, Astar_Manh) 

def timeAstarCheb(start, target):
    start_time = time.time()
    Astar_Cheb = nx.astar_path(GG, str(start), str(target), ChebDist)
    return time.time() - start_time
    #print("--- %s seconds ---" % (time.time() - start_time))
    #drawShortestWay(GG, Astar_Cheb) 

def timeAstarEukl(start, target):
    start_time = time.time()
    Astar_Eukl = nx.astar_path(GG, str(start), str(target), EuklidDist) 
    return time.time() - start_time
    #print("--- %s seconds ---" % (time.time() - start_time))
    #drawShortestWay(GG, Astar_Eukl)
    
rando = list(nodes.items())[1:1501:15]
    
filename =  "Statistics__100__points" + ".csv"

fieldnames = ['idNode', 'Dejicstra', 'Levit', 'AstarEuklid','AstarManh', 'AstarCheb']

with open(filename, "w") as file:
    writer = csv.DictWriter(file, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()
    
def csv_writer(path, fieldnames, data):
    with open(path, "a") as out_file:
        writer = csv.DictWriter(out_file, delimiter=',', fieldnames=fieldnames)
        for row in data:
            writer.writerow(row)

for nd in rando:
    cur_node = nd[0]
    line = []
    line_dict = []
    dejicstr = timeDejicstra(start, cur_node)
    aStEukl = timeAstarEukl(start, cur_node)
    aStManh = timeAstarManh(start, cur_node)
    aStCheb = timeAstarCheb(start, cur_node)
    
    start_time = time.time()
    way = Levita(start, cur_node)
    levit = time.time() - start_time
    
    line = [cur_node, dejicstr, levit, aStEukl, aStManh, aStCheb]
    inner_dict = dict(zip(fieldnames, line))
    line_dict.append(inner_dict)
    csv_writer(filename, fieldnames, line_dict)
    
"""    
filename =  "Statistics__100__points" + ".csv"

node100 = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        node100.append(row[0])
        
    
filename =  "TimeWay__100__points" + ".csv"

fieldnames = ['idNode', 'time']

with open(filename, "w") as file:
    writer = csv.DictWriter(file, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()
    
def csv_writer(path, fieldnames, data):
    with open(path, "a") as out_file:
        writer = csv.DictWriter(out_file, delimiter=',', fieldnames=fieldnames)
        for row in data:
            writer.writerow(row)

for nd in node100[1:]:
    way = Levita(start, int(nd));
    len_way_km = 0
    for i in range(len(way)-1):
        nd_1 = nodes.get(way[i])
        nd_2 = nodes.get(way[i + 1])
        len_way_km = len_way_km +  math.sqrt(((nd_1[0] - nd_2[0])**2)*76057 + ((nd_1[1] - nd_2[1])**2)*111000)
    time = len_way_km/40
    line = []
    line_dict = []
    line = [nd, time]
    inner_dict = dict(zip(fieldnames, line))
    line_dict.append(inner_dict)
    csv_writer(filename, fieldnames, line_dict)
    
""" 
hamilton = [start]   
way_between = {nd:[] for nd in target}

way_to = []     
to = [nd for nd in target]

cur_way = 0
len_cur_way = 0

for useless in range(len(hospitals)):
    
    for nd in to:
        cur_way = nx.dijkstra_path(GG, str(start), str(nd))
        len_cur_way = 0
        for i in range(len(cur_way)-1):
            nd_1 = nodes.get(int(cur_way[i]))
            nd_2 = nodes.get(int(cur_way[i + 1]))
            len_cur_way = len_cur_way +  math.sqrt((nd_1[0] - nd_2[0])**2 + (nd_1[1] - nd_2[1])**2)

        heappush(way_to, (len_cur_way, nd, cur_way))   
    
    next_node = heappop(way_to)[1]        
    to.remove(next_node)                  
    hamilton.append(next_node)            
    start = next_node                    
    way_to = []
    
    
print (hamilton)

node_to = [nd for nd in target]
node_to.append(start)
hamilton = list(tuple(node_to))
size = len(node_to)  

def lenght_edge(i, j):
    l = math.sqrt((nodes.get(int(i))[0] - nodes.get(int(j))[0])**2 + (nodes.get(int(i))[1] - nodes.get(int(j))[1])**2)
    return l

def weight_way(way):
    w = 0
    for i in range(len(way) - 1):
        w = w + lenght_edge(way[i], way[i+1])
    return w

edges = {}
for i in node_to:
    for j in node_to:
        t = tuple([i,j])
        way = nx.dijkstra_path(GG, str(i), str(j))   
        edges[t] = weight_way(way)  

min_weight = 1000000;
min_edge = () 

for key in edges:
    if edges.get(key) < min_weight and edges.get(key)!=0 :
        min_weight = edges.get(key)
        min_edge = key
        
if min_edge[0] > min_edge[1]:
    min_edge = ((min_edge[1],min_edge[0]))
    
node_to.remove(min_edge[0])    
node_to.remove(min_edge[1])    


cicle_edge = []    
cicle_node = []    
cicle_edge.append(min_edge)  
cicle_node.append(min_edge[0])
cicle_node.append(min_edge[1])

    
def lenght_way(cicle_ed, nd, ed):
    w = 0
    w = w + edges.get((ed[0], nd)) 
    w = w + edges.get((ed[1], nd)) 
    for nod in cicle_ed:
        w = w + edges.get(nod)     
    w = w - edges.get(ed)          
    return w;
    
for useless in range(size - 2):
    min_w_cicle = 1000000
    ed_to_remove = ()
    nd_to_add = 0
    
    for nd in node_to:
        for ed in cicle_edge: 
            w = lenght_way(tuple(cicle_edge), nd, ed)
            if w < min_w_cicle:
                min_w_cicle = w
                ed_to_remove = ed
                nd_to_add = nd
    
    
    if ed_to_remove[0] < nd_to_add:
        cicle_edge.append((ed_to_remove[0], nd_to_add))
    else:
        cicle_edge.append((nd_to_add, ed_to_remove[0]))
        
    if ed_to_remove[1] < nd_to_add:
        cicle_edge.append((ed_to_remove[1], nd_to_add))
    else:
        cicle_edge.append((nd_to_add, ed_to_remove[1]))
       
    
    if useless != 0:
        cicle_edge.remove(ed_to_remove)
    
    cicle_node.append(nd_to_add)
    
    node_to.remove(nd_to_add)

print (cicle_edge)
