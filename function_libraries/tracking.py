import os
import math
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import torch
from function_libraries.generation_graph import correlation_function, intensity_function, distance_function, sigmoid_distance_function, time_function, sigmoid_time_function, sigmoid_distance_function_traj, time_function_minus,parallel_gen_graph




def somehow_connected(first_index, second_index, num_jump, thres,edges_condition):
    list_branches = []
    for i in np.arange(num_jump):
        next_index = np.copy(edges_condition[np.isin(edges_condition[:,0], first_index)])
        next_index = next_index[next_index[:,2]> thres]
        next_index= next_index[:,1]
        first_index = np.copy(next_index).tolist()
        list_branches.append(first_index)
    list_branches = [m for n in list_branches for m in n]

    if np.isin(second_index, list_branches).sum() > 0:
        connected = 1
    else:
        connected = 0
    return connected


def back_search(connectivity,test, threshold, number_of_color, i):
    current_merged = connectivity[i]
    merged = current_merged[1]
    low_particle_index = current_merged[0]
    arg_the_same = np.argwhere(connectivity[:i,4] == merged)
    if len(arg_the_same) > 0:
        recent_merged = connectivity[arg_the_same[0]]
        recent_merged = recent_merged[-1]
        candidate_low_particle_index = recent_merged[0]

        low_particle_FI = test[test[:,4+number_of_color] == low_particle_index].mean(axis=0)[np.newaxis]

        candidate_particle_FI = test[test[:,4+number_of_color] == candidate_low_particle_index].mean(axis=0)[np.newaxis]

        corr = correlation_function(low_particle_FI, candidate_particle_FI , 3,  number_of_color + 3, 1, one_to_one = True)

        int_corr = intensity_function(low_particle_FI, candidate_particle_FI , 3,  number_of_color + 3, 1, one_to_one = True)
        total_corr = corr * int_corr

        if total_corr > threshold:
            is_connected = True
        else:
            is_connected = False
    else:
        is_connected = False
    return is_connected


def max_blinkling_interval(temp, interval):
    time_min = temp[:,2].min().astype(int)
    time_max = temp[:,2].max().astype(int)
    time_binary = np.zeros(time_max - time_min + 1)
    time_binary[temp[:,2].astype(int) - time_min] = 1

    max_blincking = np.convolve(time_binary,np.repeat(1, interval), 'valid') / interval
    max_blincking = min(max_blincking)

    return max_blincking


def test_initial(test_traj, width, time_thres):   
    if len(test_traj) > 1:
        temp = test_traj[:time_thres]
        temp_xy = temp[:,0:2]
        temp_dxy = temp_xy[1:,:] - temp_xy[:-1,:]
        temp_dis= np.sqrt((temp_dxy ** 2).sum(axis=1))
        dis_convol = np.convolve(temp_dis, np.repeat(1,width), "valid")
        dis_convol = min(dis_convol)

    else:
        dis_convol = 30
    return dis_convol


def sensitivity_corr(max_intensity, intensity_threshold):
    sensitivity_max = 3
    sensitivity_min = 1
    saturated_multiplier = 3
    if max_intensity <= intensity_threshold:
        value_sensitivty = sensitivity_max
    if intensity_threshold < max_intensity:
        value_sensitivty = (max_intensity - intensity_threshold) * (sensitivity_min - sensitivity_max)/(saturated_multiplier * intensity_threshold - intensity_threshold) + sensitivity_max
    if value_sensitivty < sensitivity_min:
        value_sensitivty = sensitivity_min
    return value_sensitivty


def check_split_merge(merger1, merger2, merged, actual_min_intensity):
    merge_matrix = np.vstack((merger1, merger2, merged))
    max_FI_channel = merge_matrix.max(axis = 0)
    merge_matrix_norm = merge_matrix/max_FI_channel
    sum_1_2_norm = merge_matrix_norm[0] +  merge_matrix_norm[1] 
    merge3_norm = merge_matrix_norm[2]
    channel_interest = ~(max_FI_channel < actual_min_intensity/2)
    ratio = merge3_norm/(sum_1_2_norm + 0.00001)
    ratio[ratio>1] = 1/ratio[ratio>1]
    ratio = ratio[channel_interest]

    if len(ratio) == 1:
        ratio = np.append(ratio,ratio)
    if len(ratio) == 3:
        ratio = ratio[np.argsort(ratio)][:2]
    corr = np.prod(ratio)
    return corr


def blinking_percentage(temp):
    length_temp = len(temp)
    time_min = temp[:,2].min()
    time_max = temp[:,2].max()
    when_does_it_exist = np.arange(time_min,time_max)
    arg_exist =np.isin(when_does_it_exist, temp[:,2])
    when_does_it_exist[:] = 0
    when_does_it_exist[arg_exist] = 1
    when_does_it_exist_subst = when_does_it_exist[1:] - when_does_it_exist[:-1]
    when_does_it_exist_subst = (when_does_it_exist_subst == -1) * -1
    when_does_it_exist_subst = np.append(0, when_does_it_exist_subst)
    when_does_it_exist = when_does_it_exist + when_does_it_exist_subst
    when_does_it_exist = when_does_it_exist[when_does_it_exist != 0]
    when_does_it_exist[when_does_it_exist == -1] = 0

    return when_does_it_exist.sum()/len(when_does_it_exist)


def tracking_main(vertex, edges_original, number_of_color,  intensity_threshold):
    Range_a = 3
    Range_b = Range_a + number_of_color

    initial_speed_tres = 20
    merging_tres = -1
       
    merging_tres = -1
    vertex = np.column_stack((vertex, torch.zeros(len(vertex),13)))
    vertex[:,15+number_of_color] = np.arange(len(vertex))
    vertex[:,3+number_of_color]    = np.arange(len(vertex))
    vertex[:,4+number_of_color]    = np.arange(len(vertex))


    vertex[:,4+number_of_color:8+number_of_color] = -1
    vertex[:,9+number_of_color:15+number_of_color] = -1
    actual_min_intensity = vertex[:,3:3 + number_of_color].min()
    tail = vertex[edges_original[:,0].astype(int)]
    head = vertex[edges_original[:,1].astype(int)]

    distance = distance_function(tail,head, one_to_one = True)
    time_corr_matrix = time_function(tail, head, one_to_one = True)   

    correlation = correlation_function(tail,head,Range_a, Range_b, 4, one_to_one = True) * \
        intensity_function(tail,head,Range_a, Range_b, 1, one_to_one = True) * \
            sigmoid_distance_function(distance, 10)  * \
                sigmoid_time_function(time_corr_matrix, 7,one_to_one = True)

    correlation_color = correlation_function(tail,head,Range_a, Range_b, 7, one_to_one = True) * \
        intensity_function(tail,head,Range_a, Range_b, 3, one_to_one = True) * \
            sigmoid_distance_function(distance, 1)  * \
                sigmoid_time_function(time_corr_matrix, 7,one_to_one = True)

    max_min_intensity = np.column_stack((tail[:,3:3+number_of_color].max(axis = 1), head[:,3:3+number_of_color].max(axis = 1))).min(axis = 1)

    edges_original = np.column_stack((edges_original,correlation ,time_corr_matrix, distance,correlation_color,max_min_intensity,head[:,2] ))
    vertex[:,4+number_of_color] = vertex[:,3+number_of_color]
    edges_condition = np.copy(edges_original[edges_original[:,2] > 0])
    connectivity_condition = 1
    time_0 = time.time()

    while connectivity_condition == 1:
        no_of_nodes = len(np.unique(edges_condition[:,0:2]))
        graph_to = edges_condition[:,1]
        graph_from = edges_condition[:,0]
        g = nx.DiGraph(zip(graph_from, graph_to))
        g.add_nodes_from(range(0, no_of_nodes))
        longest_path_index = list(map(int, nx.dag_longest_path(g)))

        if len(longest_path_index) < 10:
            connectivity_condition = -1    
        else:
            vertex[longest_path_index,4+number_of_color] = vertex[longest_path_index[-1],4+number_of_color] 
            vertex[longest_path_index,5+number_of_color] = 0
            vertex[longest_path_index,6+number_of_color] = 0
            vertex[longest_path_index[0],6+number_of_color] = -1
            vertex[longest_path_index[-1],5+number_of_color] = -1

            edges_condition = edges_condition[np.isin(edges_condition[:,0], longest_path_index, invert = True) ]

            edges_condition = edges_condition[np.isin(edges_condition[:,1], longest_path_index, invert = True) ]

    time_0 = time.time()

    temp_condition = edges_original[(edges_original[:,4] < 2) |(edges_original[:,5] < 20) ]
    temp_condition = temp_condition[temp_condition[:,2] > 0]
    temp_condition = temp_condition[temp_condition[:,5] > 20]

    temp_condition = temp_condition[(temp_condition[:,3] > 0.2) | (temp_condition[:,5] < 3)| (temp_condition[:,2]  > 6)  ]

    for time_interval in [6]:
        edges_condition_temp = np.copy(temp_condition[temp_condition[:,4] < time_interval + 1])
        connection = 0
        for i in np.argsort(edges_condition_temp[:,2])[::-1]:
            temp_edge = edges_condition_temp[i]
            tail = temp_edge[0].astype(int)
            head = temp_edge[1].astype(int)
            if ((vertex[:,5+number_of_color][tail] == -1) & (vertex[:,6+number_of_color][head] == -1)) == True:
                vertex[:,6+number_of_color][head] = 0
                vertex[:,5+number_of_color][tail] = 0
                vertex[:,4+number_of_color][vertex[:,4+number_of_color] == vertex[head,4+number_of_color]] = vertex[:,4+number_of_color][tail]
                connection = connection + 1
    time_0 = time.time()

    temp_condition = edges_original[((edges_original[:,3] > 0.5)|(edges_original[:,2] > 0)) &(edges_original[:,5] < 5) ]

    for time_interval in [6]:
        edges_condition_temp = np.copy(temp_condition[temp_condition[:,4] < time_interval + 1])
        connection = 0
        for i in np.argsort(edges_condition_temp[:,3])[::-1]:
            temp_edge = edges_condition_temp[i]
            tail = temp_edge[0].astype(int)
            head = temp_edge[1].astype(int)
            if ((vertex[:,5+number_of_color][tail] == -1) & (vertex[:,6+number_of_color][head] == -1)) == True:
                vertex[:,6+number_of_color][head] = 0
                vertex[:,5+number_of_color][tail] = 0
                vertex[:,4+number_of_color][vertex[:,4+number_of_color] == vertex[head,4+number_of_color]] = vertex[:,4+number_of_color][tail]
                connection = connection + 1
    vertex = vertex[vertex[:,4+number_of_color] != -1] 
    vertex_sorted = np.copy(vertex)
    time_0 = time.time()
    longer_traj = []

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        if temp.shape[0] > 2 :
            if (blinking_percentage(temp[:50,:]) > 0.8)| ( sum(temp[:5,3:3+number_of_color].mean(axis = 0)) > actual_min_intensity * 0.4):
                longer_traj.append(temp)
            else:
                temp[:,5+number_of_color:7+number_of_color] = -1
                longer_traj.append(temp)
        else:
            temp[:,5+number_of_color:7+number_of_color] = -1
            longer_traj.append(temp)
    vertex_sorted = np.vstack(longer_traj)
    vertex_sorted = vertex_sorted[np.argsort(vertex_sorted[:,2])]
    time_0 = time.time()

    edges_condition = edges_original[np.isin(edges_original[:,0], vertex_sorted[:,3+number_of_color]),:]

    edges_condition = edges_condition[np.isin(edges_condition[:,1], vertex_sorted[:,3+number_of_color]),:]

    vertex_sorted = vertex_sorted[np.isin(vertex_sorted[:,3+number_of_color], edges_condition[:,0:2].flatten() )]
    vertex_sequence_index = np.zeros((int(edges_condition[:,0:2].max())+1,2))
    vertex_sequence_index[:,0] = np.arange(len(vertex_sequence_index))
    vertex_sequence_index[:,1] = -1

    vertex_sequence_index[vertex_sorted[:,3+number_of_color].astype(int),1] = np.arange(len(vertex_sorted[:,3+number_of_color]))

    edges_condition[:,0] = vertex_sequence_index[edges_condition[:,0].astype(int),1]

    edges_condition[:,1] = vertex_sequence_index[edges_condition[:,1].astype(int),1]
    vertex_sorted[:,3+number_of_color] = np.arange(len(vertex_sorted))      
    time_0 = time.time()
    connected = np.copy(edges_condition[edges_condition[:,2]>0])
    connected = connected[connected[:,4]<3]
    connected = connected[connected[:,5]<8]
    connected = connected[np.argsort(connected[:,8])]

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp_traj = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        index_traj = temp_traj[:,3+number_of_color]
        same_traj = np.isin(connected[:,0:2], index_traj)
        different_traj = ~((same_traj[:,0] == True) & (same_traj[:,1] == True))
        connected = connected[different_traj]
    connected = np.unique(connected, axis = 0)
    time_0 = time.time()
    max_unique = vertex_sorted[:,4+number_of_color].max()

    for i in connected:
        index_connection = i[0:2].astype(int)
        unique_tail = vertex_sorted[index_connection[0]][4+number_of_color] 
        arg_traj_tail = np.argwhere(vertex_sorted[:,4+number_of_color] == unique_tail)[:,0]

        arg_traj_head = np.argwhere(vertex_sorted[:,4+number_of_color] == unique_tail)[:,0]
        if len(arg_traj_tail[arg_traj_tail>index_connection[0]]) > 0:
            vertex_sorted[index_connection[0]][5+number_of_color] = -1
            max_unique = max_unique + 1     
            vertex_sorted[arg_traj_tail[arg_traj_tail>index_connection[0]], 4+number_of_color] = max_unique

            vertex_sorted[arg_traj_tail[arg_traj_tail>index_connection[0]][0], 6+number_of_color] = -1
        unique_head = vertex_sorted[index_connection[1]][4+number_of_color] 

        arg_traj_head = np.argwhere(vertex_sorted[:,4+number_of_color] == unique_head)[:,0]

        index_head_new = np.argwhere(arg_traj_head == index_connection[1])[:,0][0]

        if index_head_new > 0:
            vertex_sorted[arg_traj_head[index_head_new]][5+number_of_color] = -1
            max_unique = max_unique + 1
            vertex_sorted[arg_traj_head[arg_traj_head>=index_connection[1]], 4+number_of_color] = max_unique

            vertex_sorted[arg_traj_head[arg_traj_head>=index_connection[1]][0], 6+number_of_color] = -1
    time_0 = time.time()

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        temp[:,5+number_of_color] = 0
        temp[:,6+number_of_color] = 0
        temp[0,6+number_of_color] = -1
        temp[-1,5+number_of_color] = -1
        vertex_sorted[vertex_sorted[:,4+number_of_color] == i] = temp
    time_0 = time.time()
    connection = -1

    edges_condition_one_to_one = np.copy(edges_condition[edges_condition[:,2]>-2])

    edges_condition_one_to_one = edges_condition_one_to_one[(edges_condition_one_to_one[:,4] < 5) | (edges_condition_one_to_one[:,5] < 10)]

    while connection != 0:  
        ending = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == -1) & (vertex_sorted[:,6+number_of_color] == 0)& (vertex_sorted[:,4+number_of_color] != -1)])
        starting = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == 0) & (vertex_sorted[:,6+number_of_color] == -1) & (vertex_sorted[:,4+number_of_color] != -1)])
        tail = np.copy(ending)
        head = np.copy(starting)
        for i in range(0, len(tail)):
            unique = tail[i][4+number_of_color]
            temp_unique = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique]
            FI_temp = temp_unique[-20:,3:3+number_of_color].mean(axis=0)
            tail[i,3:3+number_of_color] =  FI_temp
        for i in range(0, len(head)):
            unique = head[i][4+number_of_color]
            temp_unique = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique]
            FI_temp = temp_unique[:20,3:3+number_of_color].mean(axis=0)
            head[i,3:3+number_of_color] =  FI_temp        
        distance = distance_function(tail,head, one_to_one = False)
        time_corr_matrix = time_function(tail, head,one_to_one = False)      
        time_corr_matrix[time_corr_matrix==0] = 2

        correlation = correlation_function(tail,head,Range_a, Range_b, 4, one_to_one = False) *\
                        intensity_function(tail,head,Range_a, Range_b, 3, one_to_one = False) * \
                            sigmoid_distance_function(distance * np.sqrt(time_corr_matrix), 1) * sigmoid_time_function(time_corr_matrix**2, 4, one_to_one = False)
        arg_xy= np.argwhere(correlation>0.4)
        arg_x = arg_xy[:,0]
        arg_y = arg_xy[:,1]
        arg_value = correlation[arg_x, arg_y]
        sparse_correlation = np.column_stack((arg_x, arg_y, arg_value))
        connection = 0

        if len(sparse_correlation) > 0:
            while sparse_correlation[:,2].max() > 0.4:
                max_index = sparse_correlation[np.argmax(sparse_correlation[:,2]),0:2].astype(int)
                GNN_inferred = somehow_connected(tail[max_index[0], 3+number_of_color], head[max_index[1], 3+number_of_color], 5, -4, edges_condition_one_to_one)
                if (tail[max_index[0],4+number_of_color] != head[max_index[1],4+number_of_color]) & (GNN_inferred > 0):
                    sparse_correlation[sparse_correlation[:,0] == max_index[0],2] = 0
                    sparse_correlation[sparse_correlation[:,1] == max_index[1],2] = 0
                    vertex_sorted[tail[max_index[0],3+number_of_color].astype(int)][5+number_of_color] = 0

                    vertex_sorted[head[max_index[1],3+number_of_color].astype(int)][6+number_of_color] = 0

                    vertex_sorted[((vertex_sorted[:,4+number_of_color] == tail[max_index[0],4+number_of_color])&(vertex_sorted[:,2] >= head[max_index[1],2])), 4+number_of_color] = -1

                    vertex_sorted[vertex_sorted[:,4+number_of_color] == head[max_index[1],4+number_of_color], 4+number_of_color] = tail[max_index[0],4+number_of_color]

                    tail[tail[:,4+number_of_color] == head[max_index[1],4+number_of_color],4+number_of_color] = tail[max_index[0],4+number_of_color]

                    head[max_index[1],4+number_of_color] = tail[max_index[0],4+number_of_color]
                    connection = connection + 1

                elif tail[max_index[0],4+number_of_color] == head[max_index[1],4+number_of_color] :
                    sparse_correlation[(sparse_correlation[:,0] ==max_index[0])& (sparse_correlation[:,1] ==max_index[1]) ,2] = 0
                else:
                    sparse_correlation[sparse_correlation[:,0] == max_index[0],2] = 0
                    sparse_correlation[sparse_correlation[:,1] == max_index[1],2] = 0
    vertex_sorted = vertex_sorted[np.argsort(vertex_sorted[:,2])]
    vertex_sorted = vertex_sorted[vertex_sorted[:,4+number_of_color] != -1,:]

    vertex_sorted = vertex_sorted[np.isin(vertex_sorted[:,3+number_of_color], edges_condition[:,0:2].flatten() )]

    edges_condition = edges_condition[np.isin(edges_condition[:,0], vertex_sorted[:,3+number_of_color]),:]

    edges_condition = edges_condition[np.isin(edges_condition[:,1], vertex_sorted[:,3+number_of_color]),:]
    vertex_sequence_index = np.zeros((int(edges_condition[:,0:2].max())+1,2)) 
    vertex_sequence_index[:,0] = np.arange(len(vertex_sequence_index))
    vertex_sequence_index[:,1] = -1

    vertex_sequence_index[vertex_sorted[:,3+number_of_color].astype(int),1] = np.arange(len(vertex_sorted[:,3+number_of_color]))

    edges_condition[:,0] = vertex_sequence_index[edges_condition[:,0].astype(int),1]

    edges_condition[:,1] = vertex_sequence_index[edges_condition[:,1].astype(int),1]
    vertex_sorted[:,3+number_of_color] = np.arange(len(vertex_sorted))  
    vertex_sorted[:,5+number_of_color:7+number_of_color] = -1

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        if len(temp) > 4:
            temp[:,5+number_of_color] = 0
            temp[:,6+number_of_color] = 0
            temp[0,6+number_of_color] = -1
            temp[-1,5+number_of_color] = -1
            vertex_sorted[vertex_sorted[:,4+number_of_color] == i] = temp    
    time_0 = time.time()
    stranged_exist = 1
    stranded_edges = np.copy(edges_condition[edges_condition[:,2] > -4])
    stranded_edges = stranded_edges[stranded_edges[:,5]<5]

    while stranged_exist == 1:
        stranded_vertex = np.argwhere((vertex_sorted[:,5+number_of_color:7+number_of_color] == -1).sum(axis=1) ==2)[:,0]
        stranged_exist = 0
        stranded_vertex = stranded_vertex[np.isin(stranded_vertex, stranded_edges[:,0:2].flatten())]
        fixed_unique_index = []
        for i in stranded_vertex:
            temp_vertex = vertex_sorted[i]
            connection = stranded_edges[(stranded_edges[:,0] == i) | (stranded_edges[:,1] == i) ]
            stranded_excluding_i = np.delete(stranded_vertex, np.argwhere(stranded_vertex==i)[0])

            connection = connection[~np.isin(connection[:,0], stranded_excluding_i)]

            connection = connection[~np.isin(connection[:,1], stranded_excluding_i)]
            connection = connection[np.argsort(connection[:,2])[::-1]]

            for j in connection:
                counter_vertex = j[0:2][j[0:2] !=i][0].astype(int)
                candidate_traj = vertex_sorted[vertex_sorted[:,4+number_of_color] == vertex_sorted[counter_vertex,4+number_of_color]]
                checking_sandwiched = (candidate_traj[:,2].max() > temp_vertex[2] ) & (candidate_traj[:,2].min() < temp_vertex[2]  )             
                if (np.isin([temp_vertex[2]]  ,candidate_traj[:,2])[0] == False) & checking_sandwiched:    
                    vertex_sorted[i,4+number_of_color] = np.copy(vertex_sorted[counter_vertex,4+number_of_color])
                    vertex_sorted[i,5+number_of_color:7+number_of_color] = 0 
                    stranged_exist = 1    
                    fixed_unique_index.append(vertex_sorted[i,4+number_of_color])
                    break

        for i in fixed_unique_index:
            temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
            if len(temp) > 4:
                temp[:,5+number_of_color] = 0
                temp[:,6+number_of_color] = 0
                temp[0,6+number_of_color] = -1
                temp[-1,5+number_of_color] = -1
                vertex_sorted[vertex_sorted[:,4+number_of_color] == i] = temp    
    vertex_sorted = vertex_sorted[vertex_sorted[:,4+number_of_color] != -1]

    edges_condition = edges_condition[np.isin(edges_condition[:,0], vertex_sorted[:,3+number_of_color]),:]

    edges_condition = edges_condition[np.isin(edges_condition[:,1], vertex_sorted[:,3+number_of_color]),:]
    vertex_sequence_index = np.zeros((int(edges_condition[:,0:2].max())+1,2))
    vertex_sequence_index[:,0] = np.arange(len(vertex_sequence_index))
    vertex_sequence_index[:,1] = -1

    vertex_sequence_index[vertex_sorted[:,3+number_of_color].astype(int),1] = np.arange(len(vertex_sorted[:,3+number_of_color]))

    edges_condition[:,0] = vertex_sequence_index[edges_condition[:,0].astype(int),1]

    edges_condition[:,1] = vertex_sequence_index[edges_condition[:,1].astype(int),1]
    vertex_sorted[:,3+number_of_color] = np.arange(len(vertex_sorted))  

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] ==i]
        temp[:,5+number_of_color] = 0
        temp[:,6+number_of_color] = 0
        temp[0,6+number_of_color] = -1
        temp[-1,5+number_of_color] = -1
        vertex_sorted[vertex_sorted[:,4+number_of_color] ==i] = temp

    temp_condition = edges_condition[(edges_condition[:,2] > 0)|(edges_condition[:,3] > 0.6)]

    for time_interval in [6]:
        edges_condition_temp = np.copy(temp_condition[temp_condition[:,4] < time_interval + 1])
        connection = 0
        for i in np.argsort(edges_condition_temp[:,3])[::-1]:
            temp_edge = edges_condition_temp[i]
            tail = temp_edge[0].astype(int)
            head = temp_edge[1].astype(int)
            if ((vertex_sorted[:,5+number_of_color][tail] == -1) & (vertex_sorted[:,6+number_of_color][head] == -1)) == True:
                vertex_sorted[:,6+number_of_color][head] = 0
                vertex_sorted[:,5+number_of_color][tail] = 0
                vertex_sorted[:,4+number_of_color][vertex_sorted[:,4+number_of_color] == vertex_sorted[head,4+number_of_color]] = vertex_sorted[:,4+number_of_color][tail]
                connection = connection + 1

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] ==i]
        temp[:,5+number_of_color] = 0
        temp[:,6+number_of_color] = 0
        temp[0,6+number_of_color] = -1
        temp[-1,5+number_of_color] = -1
        vertex_sorted[vertex_sorted[:,4+number_of_color] ==i] = temp

    edges_condition_id_change = np.copy(edges_condition[edges_condition[:,2] > -3])

    edges_condition_id_change = edges_condition_id_change[edges_condition_id_change[:,5] < 10]

    edges_condition_id_change = edges_condition_id_change[edges_condition_id_change[:,4] < 3]

    starting = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == 0) & (vertex_sorted[:,6+number_of_color] == -1)])

    ending = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == -1) & (vertex_sorted[:,6+number_of_color] == 0)])
    tail = np.copy(ending)
    head = np.copy(starting)

    for i in range(0, len(tail)):
        unique = tail[i][4+number_of_color]
        temp_unique = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique]
        FI_temp = temp_unique[-20:,3:3+number_of_color].mean(axis=0)
        tail[i,3:3+number_of_color] =  FI_temp
    for i in range(0, len(head)):
        unique = head[i][4+number_of_color]
        temp_unique = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique]
        FI_temp = temp_unique[:20,3:3+number_of_color].mean(axis=0)
        head[i,3:3+number_of_color] =  FI_temp        
    distance = distance_function(tail,head, one_to_one = False)
    time_corr_matrix = time_function_minus(tail, head,one_to_one = False)      
    time_corr_matrix[time_corr_matrix<0] = 100

    ID_check = ((tail[:,4+number_of_color][:,np.newaxis] == head[:,4+number_of_color]) ==False) * 1

    correlation = correlation_function(tail,head,Range_a, Range_b, 4, one_to_one = False) *\
        intensity_function(tail,head,Range_a, Range_b, 3, one_to_one = False) * \
            sigmoid_distance_function(distance, 10) * sigmoid_time_function(time_corr_matrix, 6, one_to_one = False) * ID_check
    connection = 0

    while correlation.max() > 0.4:

        max_index = np.unravel_index(np.argmax(correlation), correlation.shape)
        tail_list = vertex_sorted[vertex_sorted[:,4+number_of_color] == tail[max_index[0],4+number_of_color] ]

        head_list = vertex_sorted[vertex_sorted[:,4+number_of_color] == head[max_index[1],4+number_of_color] ] 

        GNN_inferred = edges_condition_id_change[(np.isin(edges_condition_id_change[:,0], tail_list[:,3+number_of_color])) & (np.isin(edges_condition_id_change[:,1], head_list[:,3+number_of_color]))]

        if len(GNN_inferred) >0:
            if tail[max_index[0],4+number_of_color] != head[max_index[1],4+number_of_color]:
                correlation[max_index[0],:] = 0
                correlation[:,max_index[1]] = 0
                vertex_sorted[tail[max_index[0],3+number_of_color].astype(int)][5+number_of_color] = 0

                vertex_sorted[head[max_index[1],3+number_of_color].astype(int)][6+number_of_color] = 0

                vertex_sorted[((vertex_sorted[:,4+number_of_color] == tail[max_index[0],4+number_of_color])&(vertex_sorted[:,2] >= head[max_index[1],2])), 4+number_of_color] = -1

                vertex_sorted[vertex_sorted[:,4+number_of_color] == head[max_index[1],4+number_of_color], 4+number_of_color] = tail[max_index[0],4+number_of_color]

                tail[tail[:,4+number_of_color] == head[max_index[1],4+number_of_color],4+number_of_color] = tail[max_index[0],4+number_of_color]

                head[max_index[1],4+number_of_color] = tail[max_index[0],4+number_of_color]
                connection = connection + 1

            else:
                correlation[max_index[0],:] = 0
                correlation[:,max_index[1]] = 0
        else:
            correlation[max_index[0],:] = 0
            correlation[:,max_index[1]] = 0
    vertex_sorted = vertex_sorted[vertex_sorted[:,4+number_of_color] != -1,:]

    edges_condition = edges_condition[np.isin(edges_condition[:,0], vertex_sorted[:,3+number_of_color]),:]

    edges_condition = edges_condition[np.isin(edges_condition[:,1], vertex_sorted[:,3+number_of_color]),:]
    vertex_sequence_index = np.zeros((int(edges_condition[:,0:2].max())+1,2))
    vertex_sequence_index[:,0] = np.arange(len(vertex_sequence_index))
    vertex_sequence_index[:,1] = -1

    vertex_sequence_index[vertex_sorted[:,3+number_of_color].astype(int),1] = np.arange(len(vertex_sorted[:,3+number_of_color]))

    edges_condition[:,0] = vertex_sequence_index[edges_condition[:,0].astype(int),1]

    edges_condition[:,1] = vertex_sequence_index[edges_condition[:,1].astype(int),1]
    vertex_sorted[:,3+number_of_color] = np.arange(len(vertex_sorted))     
    longer_traj = []

    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        if len(temp) > 3:
            if (temp[:,3:3+ number_of_color].mean(axis = 0).max() > actual_min_intensity * 1.7)&(len(temp) >20) :
                longer_traj.append(temp)
            elif (temp[:,3:3+ number_of_color].mean(axis = 0).max() > actual_min_intensity * 1.2)& (max_blinkling_interval(temp[:], 10) > 0.7) :
                longer_traj.append(temp)                
            elif (temp[:,3:3+ number_of_color].mean(axis = 0).max() > actual_min_intensity * 1.2)& (len(temp) >100) :
                longer_traj.append(temp)     
    vertex_sorted = np.vstack(longer_traj)
    vertex_sorted = np.copy(vertex_sorted[np.argsort(vertex_sorted[:,2])])
    for i in np.unique(vertex_sorted[:,4+number_of_color]):
        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == i]
        dis_temp = (temp[1:,0:2] - temp[:-1,0:2])**2
        dis_temp = dis_temp.sum(axis = 1)
        if  (np.mean(dis_temp[:20]) >20):
            temp[:,5+number_of_color] = -1
            temp[:,6+number_of_color] = -1
            temp[:,4+number_of_color] = -1
        vertex_sorted[vertex_sorted[:,4+number_of_color] == i] = temp
    vertex_sorted = vertex_sorted[vertex_sorted[:,4+number_of_color] != -1]
    vertex_sorted = vertex_sorted[vertex_sorted[:,4+number_of_color] != -1]

    edges_condition = edges_condition[np.isin(edges_condition[:,0], vertex_sorted[:,3+number_of_color]),:]

    edges_condition = edges_condition[np.isin(edges_condition[:,1], vertex_sorted[:,3+number_of_color]),:]
    vertex_sequence_index = np.zeros((int(edges_condition[:,0:2].max())+1,2))
    vertex_sequence_index[:,0] = np.arange(len(vertex_sequence_index))
    vertex_sequence_index[:,1] = -1

    vertex_sequence_index[vertex_sorted[:,3+number_of_color].astype(int),1] = np.arange(len(vertex_sorted[:,3+number_of_color]))

    edges_condition[:,0] = vertex_sequence_index[edges_condition[:,0].astype(int),1]

    edges_condition[:,1] = vertex_sequence_index[edges_condition[:,1].astype(int),1]
    vertex_sorted[:,3+number_of_color] = np.arange(len(vertex_sorted))  

    starting = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == 0) & (vertex_sorted[:,6+number_of_color] == -1)][:,3+number_of_color]).astype(int)

    ending = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == -1) & (vertex_sorted[:,6+number_of_color] == 0) & (vertex_sorted[:,2] < vertex_sorted[:,2].max() -2)][:,3+number_of_color]).astype(int)

    arg_edges_condition_a = np.argwhere((edges_condition[:,2] > -3) & (edges_condition[:,4] <3)   & (edges_condition[:,5] <40) )[:,0]

    arg_edges_condition_b = np.argwhere((edges_condition[:,3] > 0.1) & (edges_condition[:,4] <4) & (edges_condition[:,2] > -3) & (edges_condition[:,5] <40) )[:,0]

    arg_edge_condition = np.hstack(( arg_edges_condition_a,arg_edges_condition_b )).ravel()
    arg_edge_condition = np.unique(arg_edge_condition).astype(int)
    edges_condition_split = np.copy(edges_condition[arg_edge_condition,:])
    vertex_sorted[:,9+number_of_color] = vertex_sorted[:,4+number_of_color]
    vertex_sorted[:,10+number_of_color] = -1

    a,b,c = np.unique(edges_condition_split[:,0],  return_index=True, return_counts=True)
    unique_count_index = np.column_stack((a,b,c))
    diverged_index_a = unique_count_index[:,0][unique_count_index[:,2]>1]

    diverged_index_a = np.argwhere(np.isin(edges_condition_split[:,0], diverged_index_a))[:,0]

    a,b,c = np.unique(edges_condition_split[:,1], return_index=True, return_counts=True)
    unique_count_index = np.column_stack((a,b,c))   
    diverged_index_b = unique_count_index[:,0][unique_count_index[:,2]>1]

    diverged_index_b = np.argwhere(np.isin(edges_condition_split[:,1], diverged_index_b))[:,0]

    diverged = np.unique(np.append(diverged_index_a,diverged_index_b)).astype(int)
    diverged = edges_condition_split[diverged]

    for i in starting:
        bunch_index = vertex_sorted[i][9+number_of_color]
        diverted_origin = diverged[diverged[:,1] == i]
        diverted_origin = diverted_origin[diverted_origin[:,5] < 40]
        if len(diverted_origin) > 0:
            diverted_origin_sorted = diverted_origin[~np.argsort(diverted_origin[:,2])]
            diverted_index = diverted_origin_sorted[:,0].astype(int)
            diverted_index = diverted_index[:3]
            unique_index = vertex_sorted[diverted_index,4+number_of_color]
            inferred_corr = diverted_origin_sorted[:3,2]

            unique_index_inferred_corr = np.column_stack((unique_index,diverted_index, inferred_corr))
            list_diverted_index = []

            for unique_index_i in np.unique(unique_index_inferred_corr[:,0]):
                temp = unique_index_inferred_corr[unique_index_inferred_corr[:,0] == unique_index_i]
                index_max_corr = np.argmax(temp[:,2])
                temp_max = temp[index_max_corr]
                list_diverted_index.append(temp_max[1].astype(int))
            for j,k in enumerate(list_diverted_index):
                diverted_unique = vertex_sorted[k][9+number_of_color]
                vertex_sorted[i,10+number_of_color+j] = vertex_sorted[k][4+number_of_color]
                vertex_sorted[:,9+number_of_color][vertex_sorted[:,9+number_of_color] == diverted_unique] = bunch_index
            vertex_sorted[i,7+number_of_color] = 1
    for i in ending:
        bunch_index = vertex_sorted[i][9+number_of_color]
        diverted_origin = diverged[diverged[:,0] == i]
        diverted_origin = diverted_origin[diverted_origin[:,5] < 40]
        if len(diverted_origin) > 0:
            diverted_origin_sorted = diverted_origin[~np.argsort(diverted_origin[:,2])]
            diverted_index = diverted_origin_sorted[:,1].astype(int)
            diverted_index = diverted_index[:3]
            unique_index = vertex_sorted[diverted_index,4+number_of_color]
            inferred_corr = diverted_origin_sorted[:3,2]

            unique_index_inferred_corr = np.column_stack((unique_index,diverted_index, inferred_corr))
            list_diverted_index = []

            for unique_index_i in np.unique(unique_index_inferred_corr[:,0]):
                temp = unique_index_inferred_corr[unique_index_inferred_corr[:,0] == unique_index_i]
                index_max_corr = np.argmax(temp[:,2])
                temp_max = temp[index_max_corr]
                list_diverted_index.append(temp_max[1].astype(int))
            for j,k in enumerate(list_diverted_index):
                diverted_unique = vertex_sorted[k][9+number_of_color]
                vertex_sorted[i,10+number_of_color+j] = vertex_sorted[k][4+number_of_color]
                vertex_sorted[:,9+number_of_color][vertex_sorted[:,9+number_of_color] == diverted_unique] = bunch_index
            vertex_sorted[i,7+number_of_color] = 0

    starting = np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] == 0) & (vertex_sorted[:,6+number_of_color] == -1) & (vertex_sorted[:,7+number_of_color] == -1)][:,3+number_of_color]).astype(int)

    ending =   np.copy(vertex_sorted[(vertex_sorted[:,5+number_of_color] ==-1) & (vertex_sorted[:,6+number_of_color] ==  0) & (vertex_sorted[:,7+number_of_color] == -1)& (vertex_sorted[:,2] < vertex_sorted[:,2].max() -2)][:,3+number_of_color]).astype(int)

    for i in starting:
        cutting_unique = vertex_sorted[i][4+number_of_color]
        temp_i = vertex_sorted[vertex_sorted[:,4+number_of_color] == cutting_unique]
        if vertex_sorted[int(i)][10+number_of_color] == -1:
            if test_initial(temp_i, 10,150) > initial_speed_tres:
                cutting_global_unique = vertex_sorted[i][9+number_of_color]
                cutting_time = vertex_sorted[i][2]
                candidate_cutted = edges_condition[(edges_condition[:,1] == i) & (edges_condition[:,4] <4) & (edges_condition[:,5] < 35)]
                candidate_cutted = candidate_cutted[:,0]
                candidate_cutted = candidate_cutted.astype(int)
                list_split = np.zeros((0,2))

                if len(candidate_cutted) > 0:
                    for j in candidate_cutted:
                        unique_cutted = vertex_sorted[j][4+number_of_color]
                        temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique_cutted]
                        if len(temp)>0:
                            Range_a = 3
                            Range_b = 3+number_of_color
                            mean_FI_before_cutting = np.copy(temp[(temp[:,2] < cutting_time) & (cutting_time - 10 < temp[:,2] )]).mean(axis = 0)[np.newaxis]

                            mean_FI_after_cutting  =np.copy(temp[(temp[:,2] > cutting_time) & (cutting_time + 10 > temp[:,2] )]).mean(axis = 0)[np.newaxis]

                            mean_FI_cutting = np.copy(vertex_sorted[vertex_sorted[:,4+number_of_color] ==cutting_unique])[:10,:].mean(axis = 0)[np.newaxis]            
                            merge_sum = mean_FI_after_cutting + mean_FI_cutting

                            min_FI =min(mean_FI_before_cutting[:,Range_a:Range_b].max(), mean_FI_cutting[:,Range_a:Range_b].max(), mean_FI_after_cutting[:,Range_a:Range_b].max())

                            if math.isnan(min_FI) == True:
                                sensitivity_value = 2
                            else:
                                sensitivity_value = sensitivity_corr(min_FI,intensity_threshold )
                            dot_corr_a = correlation_function(merge_sum, mean_FI_before_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)
                            intensity_corr_a = intensity_function(merge_sum,mean_FI_before_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True) 
                            total_corr_a = dot_corr_a * intensity_corr_a

                            substraction_b = mean_FI_before_cutting - mean_FI_after_cutting

                            dot_corr_b = correlation_function(mean_FI_cutting, substraction_b, Range_a, Range_b, sensitivity_value, one_to_one = True)

                            intensity_corr_b = intensity_function(mean_FI_cutting,substraction_b,Range_a , Range_b, sensitivity_value, one_to_one = True)                 
                            total_corr_b = dot_corr_b * intensity_corr_b

                            FI_temp_0_guess =  mean_FI_before_cutting - mean_FI_after_cutting
                            FI_temp_0_guess[FI_temp_0_guess<0] = 0

                            dot_corr_a_capcity = correlation_function(FI_temp_0_guess, mean_FI_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)

                            intensity_corr_a_capcity = intensity_function(FI_temp_0_guess,mean_FI_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True)                        

                            total_corr_a_capacity = dot_corr_a_capcity * intensity_corr_a_capcity

                            FI_temp_4_guess = mean_FI_before_cutting - mean_FI_cutting
                            FI_temp_4_guess[FI_temp_4_guess<0] = 0

                            dot_corr_b_capcity = correlation_function(FI_temp_4_guess, mean_FI_after_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)

                            intensity_corr_b_capcity = intensity_function(FI_temp_4_guess,mean_FI_after_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True)                        

                            total_corr_b_capcity = dot_corr_b_capcity * intensity_corr_b_capcity    

                            total_corr_capacity = (total_corr_a_capacity + total_corr_b_capcity)/2

                            total_corr = (total_corr_a + total_corr_b)/2 * total_corr_capacity
                            index_corr = np.array([j, total_corr[0]])
                            list_split = np.vstack((list_split, index_corr))

                    if list_split[:,1].max() > 0.6:


                        vertex_sorted[i.astype(int)][10+number_of_color] = unique_cutted
                        vertex_sorted[i.astype(int)][7+number_of_color] = 1

                        unique_global_cutted =   vertex_sorted[j.astype(int)][9+number_of_color]

                        vertex_sorted[:,9+number_of_color][vertex_sorted[:,9+number_of_color] == cutting_global_unique] = unique_global_cutted

    for i in ending:
        cutting_unique = vertex_sorted[i][4+number_of_color]
        temp_i = vertex_sorted[vertex_sorted[:,4+number_of_color] == cutting_unique]
        if vertex_sorted[int(i)][10+number_of_color] == -1:
            cutting_global_unique = vertex_sorted[i][9+number_of_color]
            cutting_time = vertex_sorted[i][2]
            candidate_cutted = edges_condition[(edges_condition[:,0] == i) & (edges_condition[:,4] <4) & (edges_condition[:,5] < 35)]
            candidate_cutted = candidate_cutted[:,1]
            candidate_cutted = candidate_cutted.astype(int)
            list_split = np.zeros((0,2))

            if len(candidate_cutted) > 0:
                for j in candidate_cutted:
                    unique_cutted = vertex_sorted[j][4+number_of_color]
                    temp = vertex_sorted[vertex_sorted[:,4+number_of_color] == unique_cutted]
                    if len(temp)>0:
                        Range_a = 3
                        Range_b = 3+number_of_color
                        mean_FI_before_cutting = np.copy(temp[(temp[:,2] < cutting_time) & (cutting_time - 10 < temp[:,2] )]).mean(axis = 0)[np.newaxis]

                        mean_FI_after_cutting  =np.copy(temp[(temp[:,2] > cutting_time) & (cutting_time + 10 > temp[:,2] )]).mean(axis = 0)[np.newaxis]

                        mean_FI_cutting = np.copy(vertex_sorted[vertex_sorted[:,4+number_of_color] ==cutting_unique])[-10:,:].mean(axis = 0)[np.newaxis]            
                        merge_sum = mean_FI_before_cutting + mean_FI_cutting

                        min_FI =min(mean_FI_before_cutting[:,Range_a:Range_b].max(), mean_FI_cutting[:,Range_a:Range_b].max(), mean_FI_after_cutting[:,Range_a:Range_b].max())

                        if math.isnan(min_FI) == True:
                            sensitivity_value = 2
                        else:
                            sensitivity_value = sensitivity_corr(min_FI,intensity_threshold )
                        dot_corr_a = correlation_function(merge_sum, mean_FI_after_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)
                        intensity_corr_a = intensity_function(merge_sum,mean_FI_after_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True) 
                        total_corr_a = dot_corr_a * intensity_corr_a
                        substraction_b = mean_FI_after_cutting - mean_FI_cutting

                        dot_corr_b = correlation_function(mean_FI_before_cutting, substraction_b, Range_a, Range_b, sensitivity_value, one_to_one = True)

                        intensity_corr_b = intensity_function(mean_FI_before_cutting,substraction_b,Range_a , Range_b, sensitivity_value, one_to_one = True)                 
                        total_corr_b = dot_corr_b * intensity_corr_b

                        FI_temp_0_guess =  mean_FI_after_cutting - mean_FI_cutting
                        FI_temp_0_guess[FI_temp_0_guess<0] = 0

                        dot_corr_a_capcity = correlation_function(FI_temp_0_guess, mean_FI_before_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)

                        intensity_corr_a_capcity = intensity_function(FI_temp_0_guess,mean_FI_before_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True)                        

                        total_corr_a_capacity = dot_corr_a_capcity * intensity_corr_a_capcity

                        FI_temp_1_guess = mean_FI_after_cutting - mean_FI_before_cutting
                        FI_temp_1_guess[FI_temp_1_guess<0] = 0

                        dot_corr_b_capcity = correlation_function(FI_temp_1_guess, mean_FI_cutting, Range_a, Range_b, sensitivity_value , one_to_one = True)

                        intensity_corr_b_capcity = intensity_function(FI_temp_1_guess,mean_FI_cutting,Range_a , Range_b, sensitivity_value, one_to_one = True)                        

                        total_corr_b_capcity = dot_corr_b_capcity * intensity_corr_b_capcity    

                        total_corr_capacity = (total_corr_a_capacity + total_corr_b_capcity)/2

                        total_corr = (total_corr_a + total_corr_b)/2 * total_corr_capacity
                        index_corr = np.array([j, total_corr[0]])
                        list_split = np.vstack((list_split, index_corr))

                if list_split[:,1].max() > 0.6:


                    vertex_sorted[i.astype(int)][10+number_of_color] = unique_cutted
                    vertex_sorted[i.astype(int)][7+number_of_color] = 1

                    unique_global_cutted =   vertex_sorted[j.astype(int)][9+number_of_color]

                    vertex_sorted[:,9+number_of_color][vertex_sorted[:,9+number_of_color] == cutting_global_unique] = unique_global_cutted
    final_liposome = np.copy(vertex_sorted)


    def traj_optimization(sub_traj):
        Range_a = 3
        Range_b = 3+number_of_color
        test = np.copy(sub_traj)
        connectivity = np.zeros((0,8))
        for i in test[test[:,10+number_of_color] !=-1]:
            for j in range(0, (i[10+number_of_color:15+number_of_color] != -1).sum()):
                temp = [i[4+number_of_color], i[10+number_of_color + j], i[7+number_of_color], i[2], -1,1,1,0]
                connectivity = np.vstack((connectivity, temp))
        connectivity[connectivity[:,2]==1, 3] = connectivity[connectivity[:,2]==1, 3] - 1
        connectivity = connectivity[np.argsort(connectivity[:,3]),:]
        max_index = connectivity[:,[0,1,4]].max()

        for j, i in enumerate(connectivity):
            new_index = np.copy(max_index + 1)
            target = i[1]
            time_cut = i[3]
            if (len(test[(test[:,4+number_of_color] == target) & (test[:,2] >time_cut )]) > 0) & (len(test[(test[:,4+number_of_color] == target) & (test[:,2] <=time_cut )]) > 0):
                test[:,4+number_of_color][(test[:,4+number_of_color] == target) & (test[:,2] >time_cut )]= new_index
                i[4] = new_index
                connectivity[j+1:,0][(connectivity[j+1:,0] == target) & (connectivity[j+1:,3]>time_cut)] = new_index

                connectivity[j+1:,1][(connectivity[j+1:,1] == target) & (connectivity[j+1:,3]>time_cut)] = new_index
                connectivity[j] = i
                max_index = max_index + 1

            else:
                arg_previous = np.argwhere(connectivity[:j,1]==i[1])[:,0]
                if len(arg_previous)> 0:
                    index_previous_cutted = np.argwhere(connectivity[:j,1]==i[1])[:,0][-1]
                    i[4] = connectivity[index_previous_cutted, 4]
                else:
                    if i[2] == 0:
                        i[4] = i[1]
                        i[1] = i[0]
                        i[0] = i[4]
                        i[4] = -1
                    elif i[2] == 1:
                        i[4] = -1                            
        connectivity_unique = np.unique(connectivity[:,[0,1,4]]).astype(int)
        traj_unique = np.unique(test[:,4+number_of_color])
        exist_in_traj = np.isin(connectivity_unique, traj_unique)
        null_connectivity = connectivity_unique[~exist_in_traj]
        connectivity_unique = connectivity_unique[exist_in_traj]

        for i in null_connectivity:
            connectivity[connectivity[:,0] == i,0] = -1
            connectivity[connectivity[:,1] == i,1] = -1
            connectivity[connectivity[:,4] == i,4] = -1
        connectivity_unique = np.unique(connectivity[:,[0,1,4]]).astype(int)
        connectivity_unique = connectivity_unique[connectivity_unique !=-1]
        re_indexing = np.arange(len(connectivity_unique))
        maps = np.arange(connectivity_unique.max() + 2)
        maps[[connectivity_unique]] = re_indexing
        maps[-1] = -1
        connectivity = connectivity.astype(int)
        connectivity[:,0] = maps[connectivity[:,0]]
        connectivity[:,1] = maps[connectivity[:,1]]
        connectivity[:,4] = maps[connectivity[:,4]]
        test[:,4+number_of_color] = maps[test[:,4+number_of_color].astype(int)]
        connectivity = connectivity[np.argsort(connectivity[:,3])]
        in_connectivity = np.unique(connectivity[:,[0,1,4]])
        not_in_connectivity = np.unique(test[:,4+number_of_color]).astype(int)

        not_in_connectivity = not_in_connectivity[~np.isin(not_in_connectivity, in_connectivity)]
        max_index = in_connectivity.max()
        independent_source = []

        for n_i in not_in_connectivity:
            max_index = max_index + 1
            test[test[:,4+number_of_color]==n_i, 4+number_of_color] = max_index 
            independent_source.append(max_index)
        for i in np.arange(connectivity.shape[0]):
            temp_connectivity = np.copy(connectivity[i])
            if (temp_connectivity[:5] == -1).sum() ==0:
                if temp_connectivity[2] == 0:
                    FI_temp_0 = test[test[:,4+number_of_color] == temp_connectivity[0]][-10:,Range_a:Range_b].mean(axis = 0)
                    FI_temp_1 = test[test[:,4+number_of_color] == temp_connectivity[1]][-10:,Range_a:Range_b].mean(axis = 0)
                    FI_temp_4 = test[test[:,4+number_of_color] == temp_connectivity[4]][:10,Range_a:Range_b].mean(axis = 0) 

                    corr = check_split_merge(FI_temp_0, FI_temp_1, FI_temp_4, actual_min_intensity)

                    position_0 = test[test[:,4+number_of_color] == temp_connectivity[0]][-1][np.newaxis]

                    position_4 = test[test[:,4+number_of_color] == temp_connectivity[4]][0][np.newaxis]

                    distance = distance_function(position_0,position_4, one_to_one = True)
                    dist_corr = sigmoid_distance_function_traj(distance, 2) 
                    total_corr =corr* dist_corr
                    connectivity[i][7] = (total_corr[0] * 100).astype(int)

                    if total_corr[0] < 0.20:
                        connectivity[i][5] = -1 
                    if  len(test[test[:,4+number_of_color] == temp_connectivity[4]])  < 3:
                        connectivity[i][5] = -1 
                if temp_connectivity[2] == 1:
                    temp_4 = test[test[:,4+number_of_color] == temp_connectivity[4]]
                    temp_1 = test[test[:,4+number_of_color] == temp_connectivity[1]]
                    temp_0 = test[test[:,4+number_of_color] == temp_connectivity[0]]

                    FI_temp_0 = test[test[:,4+number_of_color] == temp_connectivity[0]][:10,Range_a:Range_b].mean(axis = 0)

                    FI_temp_1 = test[test[:,4+number_of_color] == temp_connectivity[1]][-10:,Range_a:Range_b].mean(axis = 0)

                    FI_temp_4 = test[test[:,4+number_of_color] == temp_connectivity[4]][:10,Range_a:Range_b].mean(axis = 0)                    

                    corr = check_split_merge(FI_temp_4, FI_temp_0, FI_temp_1, actual_min_intensity)

                    position_0 = test[test[:,4+number_of_color] == temp_connectivity[0]][0][np.newaxis]

                    position_1 = test[test[:,4+number_of_color] == temp_connectivity[1]][-1][np.newaxis]

                    distance = distance_function(position_1,position_0, one_to_one = True)
                    dist_corr = sigmoid_distance_function_traj(distance, 2) 
                    total_corr =corr* dist_corr
                    connectivity[i][7] = (total_corr[0] * 100).astype(int)

                    GNN_inferred = somehow_connected(temp_1[-1,3+number_of_color],  temp_0[0,3+number_of_color], 5, 0, edges_condition)

                    if ((GNN_inferred > 0) | (total_corr[0] > 0.35)) & ((test_initial(temp_0, 20, 150) > initial_speed_tres) | (test_initial(temp_4, 20, 150) > initial_speed_tres)):
                        connectivity[i][5] = 1
                    else:
                        connectivity[i][5] = -1
                    temp = test[test[:,4+number_of_color] == temp_connectivity[1]][4:]
        connectivity[(connectivity[:,[0,1,4]]==-1).any(axis=1),5] = -1
        edge_list = []
        source_split = []

        for j in np.arange(connectivity.shape[0]):
            i = connectivity[j]
            if i[0] != -1:
                temp_0 = test[test[:,4+number_of_color] == i[0]]
                temp_0_end   = temp_0[-1][0:2][np.newaxis]
                temp_0_start   = temp_0[0][0:2][np.newaxis]
                FI_0 = temp_0.mean(axis=0)[np.newaxis]
            if i[1] != -1:
                temp_1 = test[test[:,4+number_of_color] == i[1]]
                temp_1_end   = temp_1[-1][0:2][np.newaxis]
                temp_1_start   = temp_1[0][0:2][np.newaxis]
                FI_1 = temp_1.mean(axis=0)[np.newaxis]
            if i[4] != -1:
                temp_4 = test[test[:,4+number_of_color] == i[4]]
                temp_4_end   = temp_4[-1][0:2][np.newaxis]
                temp_4_start   = temp_4[0][0:2][np.newaxis]       
                FI_4 = temp_4.mean(axis=0)[np.newaxis]   
            if (i[5] == 1):
                if i[2] == 0:
                    if(i[0] != -1) & (i[4] != -1):
                        if i[6] == 0:
                            GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_0[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                            if len(GNN_inferred) > 0:
                                if GNN_inferred.max() > 0:
                                    edge_list.append([i[0], i[4]])
                        else:
                            edge_list.append([i[0], i[4]])
                    if(i[1] != -1) & (i[4] != -1):
                        if i[6] == 0:
                            GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                            if len(GNN_inferred) > 0:
                                if GNN_inferred.max() > 0:
                                    edge_list.append([i[1], i[4]])
                        else:
                            edge_list.append([i[1], i[4]])
                if i[2] == 1:
                    if (i[1] != -1) & (i[0] != -1):
                        if i[6] == 0:
                            GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_0[:,3+number_of_color]))][:,2]
                            if len(GNN_inferred) > 0:
                                if GNN_inferred.max() > 0:
                                    edge_list.append([i[1], i[0]])
                        elif np.isin(temp_1[:,2], temp_0[:,2]).sum() <3:
                            edge_list.append([i[1], i[0]])
                    if(i[1] != -1) & (i[4] != -1):
                        if i[6] == 0:
                            GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                            if len(GNN_inferred) > 0:
                                if GNN_inferred.max() > 0:
                                    edge_list.append([i[1], i[4]])
                        elif np.isin(temp_1[:,2], temp_4[:,2]).sum() <3:
                            edge_list.append([i[1], i[4]])
            elif (i[4] == -1):        
                if(i[1] != -1) & (i[0] != -1):
                    edge_list.append([i[1], i[0]])
            elif (i[5] == -1):
                if (i[2] == 0) & (i[1] != -1) & (i[4] != -1):
                    min_FI_1_4_merge = min(temp_1[:,Range_a:Range_b].max(), temp_4[:,Range_a:Range_b].max())
                    sensitivity_value_1_4_merge = sensitivity_corr(min_FI_1_4_merge,intensity_threshold )
                    FI_correlation_1_4 = correlation_function(FI_1, FI_4 , Range_a,  Range_b, sensitivity_value_1_4_merge, one_to_one = True)

                    distance_1_4_merge = distance_function(temp_1_end,temp_4_start, one_to_one = True )

                    dist_corr_1_4_merge = sigmoid_distance_function_traj(distance_1_4_merge, 0.1) 

                    intensity_corr_1_4_merge = intensity_function(FI_1,FI_4,Range_a , Range_b, sensitivity_value_1_4_merge, one_to_one = True) 

                    corr_1_4_merge = FI_correlation_1_4 * dist_corr_1_4_merge * intensity_corr_1_4_merge

                if (i[2] == 0) & (i[0] != -1) & (i[4] != -1):
                    min_FI_0_4_merge = min(temp_0[:,Range_a:Range_b].max(), temp_4[:,Range_a:Range_b].max())
                    sensitivity_value_0_4_merge = sensitivity_corr(min_FI_0_4_merge,intensity_threshold )
                    FI_correlation_0_4 = correlation_function(FI_0, FI_4 , Range_a,  Range_b, sensitivity_value_0_4_merge, one_to_one = True)

                    distance_0_4_merge = distance_function(temp_0_end,temp_4_start, one_to_one = True )

                    dist_corr_0_4_merge = sigmoid_distance_function_traj(distance_0_4_merge, 0.1) 

                    intensity_corr_0_4_merge = intensity_function(FI_0,FI_4,Range_a , Range_b, sensitivity_value_0_4_merge, one_to_one = True) 

                    corr_0_4_merge = FI_correlation_0_4 * dist_corr_0_4_merge * intensity_corr_0_4_merge

                if (i[2] == 1) &(i[1] != -1) & (i[0] != -1):
                    min_FI_1_0_split = min(temp_1[:,Range_a:Range_b].max(), temp_0[:,Range_a:Range_b].max())
                    sensitivity_value_1_0_split = sensitivity_corr(min_FI_1_0_split,intensity_threshold )
                    FI_correlation_1_0_split = correlation_function(FI_0, FI_1 , Range_a,  Range_b, sensitivity_value_1_0_split, one_to_one = True)

                    distance_1_0_split = distance_function(temp_1_end,temp_0_start, one_to_one = True )

                    dist_corr_1_0_split = sigmoid_distance_function_traj(distance_1_0_split, 0.1) 

                    intensity_corr_1_0_split = intensity_function(FI_1,FI_0,Range_a , Range_b, sensitivity_value_1_0_split, one_to_one = True) 

                    corr_1_0_split = FI_correlation_1_0_split * dist_corr_1_0_split * intensity_corr_1_0_split                

                if (i[2] == 1) &(i[1] != -1) & (i[4] != -1):
                    min_FI_1_4_split = min(temp_1[:,Range_a:Range_b].max(), temp_4[:,Range_a:Range_b].max())
                    sensitivity_value_1_4_split = sensitivity_corr(min_FI_1_4_split,intensity_threshold )
                    FI_correlation_1_4_split = correlation_function(FI_1, FI_4 , Range_a,  Range_b, sensitivity_value_1_4_split, one_to_one = True)

                    distance_1_4_split = distance_function(temp_1_end,temp_4_start, one_to_one = True )

                    dist_corr_1_4_split = sigmoid_distance_function_traj(distance_1_4_split, 0.1) 

                    intensity_corr_1_4_split = intensity_function(FI_1,FI_4,Range_a , Range_b, sensitivity_value_1_4_split, one_to_one = True) 

                    corr_1_4_split = FI_correlation_1_4_split * dist_corr_1_4_split * intensity_corr_1_4_split     

                if (i[2] == 0) & (i[1] != -1) & (i[4] != -1) :
                    GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                    if len(GNN_inferred) > 0:
                        if GNN_inferred.max() > merging_tres:
                            edge_list.append([i[1], i[4]])
                        elif (GNN_inferred.max() > -4) & (FI_correlation_1_4*intensity_corr_1_4_merge * dist_corr_1_4_merge> 0.5):
                            edge_list.append([i[1], i[4]])
                        elif (test_initial(test[test[:,4+number_of_color] == i[1]], 10,150) < initial_speed_tres) & (len(test[test[:,4+number_of_color] == i[1]]) > 10):
                            source_split.append(i[1])
                if (i[2] == 0) & (i[0] != -1) & (i[4] != -1) :
                    GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_0[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                    if len(GNN_inferred) > 0:
                        if GNN_inferred.max() > merging_tres:
                            edge_list.append([i[0], i[4]])
                        elif (GNN_inferred.max() > -4) & (FI_correlation_0_4*intensity_corr_1_4_merge * dist_corr_0_4_merge> 0.5):
                            edge_list.append([i[0], i[4]])
                        elif (test_initial(test[test[:,4+number_of_color] == i[0]], 10,150) < initial_speed_tres) & (len(test[test[:,4+number_of_color] == i[0]]) > 10):
                            source_split.append(i[0])
                if (i[2] == 1) & (i[1] != -1) & (i[0] != -1):
                    GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_0[:,3+number_of_color]))][:,2]
                    if len(GNN_inferred) > 0:
                        if GNN_inferred.max() > merging_tres:
                            if np.isin(temp_1[:,2], temp_0[:,2]).sum() <3:
                                edge_list.append([i[1], i[0]])
                        if (back_search(connectivity,test, 0.8,number_of_color, j) == True) & (dist_corr_1_0_split > 0.9):
                                edge_list.append([i[1], i[0]])
                        if (test_initial(temp_0, 20,150) < initial_speed_tres):
                            source_split.append(i[0])                                
                if (i[2] == 1) & (i[1] != -1) & (i[4] != -1):
                    GNN_inferred = edges_condition[(np.isin(edges_condition[:,0], temp_1[:,3+number_of_color])) & (np.isin(edges_condition[:,1], temp_4[:,3+number_of_color]))][:,2]
                    if len(GNN_inferred) > 0:
                        if GNN_inferred.max() > merging_tres:
                            if np.isin(temp_1[:,2], temp_4[:,2]).sum() <3:
                                edge_list.append([i[1], i[4]])
                        if ((test_initial(temp_4, 20,150) < initial_speed_tres) & (GNN_inferred.max() < 10)) :
                            source_split.append(i[4])                                  
                    else:
                        if corr_1_4_split > 0.9:
                            edge_list.append([i[1], i[4]])
        source_split = np.unique(source_split)
        not_sources = connectivity[(connectivity[:,2] == 1) , :][:,4]
        not_sources = np.unique(not_sources)
        source_split_filtered = source_split

        if len(edge_list) == 0:
            test = np.copy(sub_traj)
            traj_inferred_corrected = [[i] for i in np.unique(test[:,4+number_of_color]).astype(int)]
        else:
            edge_list = np.array(edge_list)
            edge_list = edge_list[edge_list[:,0] != -1]
            edge_list = edge_list[edge_list[:,1] != -1]
            unique_intensity = np.zeros((0,2+number_of_color))
            for i in np.unique(test[:,4+number_of_color]):
                temp = test[test[:,4+number_of_color] == i]
                intensity = temp[:10,3:3+number_of_color].mean(axis = 0)
                intensity_list =  list(intensity) 

                intensity_list = intensity_list + [sum(intensity_list)] + [temp[:,2].min()]

                unique_intensity = np.vstack((unique_intensity,   intensity_list))

            for i in range(0,len(connectivity)):
                if connectivity[i,2] == 1:
                    temp = test[test[:,4+number_of_color] == connectivity[i,1]]
                    if len(temp) > 10:
                        displacement = np.sqrt(((temp[1:,0]-temp[:-1,0])**2 + (temp[1:,1]-temp[:-1,1])**2))
                        averaged_displacement = displacement[-10:].mean()
                        x_difference = max(temp[-50:,0]) - min(temp[-50:,0])
                        y_difference = max(temp[-50:,1]) - min(temp[-50:,1])

                        xy_difference = np.sqrt(x_difference**2 + y_difference**2)
                        travel_distance = displacement[-50:].sum()

                        if averaged_displacement > 4:
                            connectivity[i,5] = -1
                        if xy_difference > 30:
                            connectivity[i,5] = -1
                        if travel_distance > 70:
                            connectivity[i,5] = -1
            edge_list = edge_list[edge_list[:,0] !=edge_list[:,1]]
            graph = {}
            for i in np.unique(edge_list[:,0]):
                i = int(i)
                dst = edge_list[:,1][edge_list[:,0] == i]
                dst = np.unique(dst).astype(int)
                graph[i] = list(dst)
            max_segments = np.max(connectivity[:,[0,1,4]])
            segments_index_all = np.arange(max_segments+1)

            island_traj = segments_index_all[~np.isin(segments_index_all, edge_list.flatten())]

            sources = np.unique(edge_list[:,0][~np.isin(edge_list[:,0], edge_list[:,1])])

            sources = np.unique(np.concatenate((sources,island_traj,np.array(source_split_filtered) )))
            sources = sources[sources!=-1]
            sources = np.append(sources,independent_source)
            sources = sources.astype(int)


            def connection_onestep(sub_traj, graph, source): 
                return [sub_traj + [n] for n in graph[source]]
            traj = [[i] for i in sources]
            while len(set(graph.keys()) & set([k[-1] for k in traj])) > 0:
                for i in traj:
                    source = i[-1]
                    if source in graph.keys():
                        traj.remove(i)
                        diverted = connection_onestep(i, graph, source)
                        traj = traj + diverted
                    else:
                        pass
            lasso_segments = connectivity[connectivity[:,2]==0][:,4]
            lasso_segments = np.unique(lasso_segments)
            a,b = np.unique(edge_list[:,1], return_counts=True)
            a_b = np.column_stack((a,b))
            a_b = a_b[a_b[:,1] > 1]
            lasso_segments = a_b[np.isin(a_b,lasso_segments)]
            lasso_true = []

            for i in lasso_segments:
                temp = test[test[:,4+number_of_color] == i]
                displacement = np.sqrt(((temp[1:,0]-temp[:-1,0])**2 + (temp[1:,1]-temp[:-1,1])**2))
                x_difference = max(temp[:,0]) - min(temp[:,0])
                y_difference = max(temp[:,1]) - min(temp[:,1])
                xy_difference = np.sqrt(x_difference**2 + y_difference**2)
                travel_distance = displacement[-50:].sum()

                if len(temp) > 200:
                    lasso_true.append(i)
                elif travel_distance > 100:
                    lasso_true.append(i)
                elif xy_difference > 30:
                    lasso_true.append(i)
                elif test_initial(temp, 10,150) < initial_speed_tres:
                    lasso_true.append(i)
            lasso_true = np.unique(lasso_true)
            if len(traj) > 1:
                unique_list = []
                if len(traj) > 0:
                    for x in traj:
                        if x not in unique_list:
                            unique_list.append(x)    
                traj = [x for x in unique_list if x != []]
                traj_combination_matrix = np.zeros((len(traj),len(unique_intensity)))
                for i, j in enumerate(traj):
                    traj_combination_matrix[i][j] = 1
                traj_combination_matrix = traj_combination_matrix.T.astype(dtype=bool)
                traj_min_intensity = []    


                def is_included_intensity(unique_intensity_list, merit,number_of_color):
                    if len(unique_intensity[traj_combination_matrix[:,i],:])>1:
                        low = np.copy((unique_intensity_list -merit       )[:,:number_of_color])
                        high = np.copy((unique_intensity_list)[:,:number_of_color])
                        difference = (high[:,np.newaxis]-low )
                        minimum_list = ((difference > 0).all(axis=2)).all(axis=0)
                        arg_min_list = np.argwhere(minimum_list == True)

                        if len(arg_min_list)>0:
                            arg_min_index = arg_min_list[np.argmin(unique_intensity_list[arg_min_list, number_of_color])]
                            arg_min_unique = unique_intensity_list[arg_min_index,:number_of_color]
                        else:
                            arg_min_unique = np.repeat(999,number_of_color)
                    else:
                        arg_min_unique = unique_intensity_list[0,:number_of_color]
                    return arg_min_unique
                for i in range(0,traj_combination_matrix.shape[1]):
                    min_unique = is_included_intensity(unique_intensity[traj_combination_matrix[:,i],:], 50, number_of_color)
                    traj_min_intensity.append(min_unique)
                traj_min_intensity = np.vstack(traj_min_intensity)

                traj_min_intensity = np.repeat(traj_min_intensity.T[np.newaxis], repeats = traj_combination_matrix.shape[0], axis = 0)
                traj_min_intensity = np.swapaxes(traj_min_intensity, 1,2)

                traj_combination_matrix = np.repeat(traj_combination_matrix[:,:,np.newaxis], repeats = number_of_color, axis = 2)
                A = traj_min_intensity * traj_combination_matrix
                x = cp.Variable(A.shape[1], boolean=True)

                constraints =  [(traj_combination_matrix[:,:,0]@x)[i] >= 1 for i in range(0, traj_combination_matrix[:,:,0].shape[0])]
                epsilon = 1e-6
                objective = cp.Minimize(
                    sum(
                        cp.sum(

                            cp.abs( (A[:, :, i] @ x - unique_intensity[:, i]) / (unique_intensity[:, i] + epsilon) )
                        )

                        for i in range(A.shape[2])
                    )
                )
                prob = cp.Problem(objective,constraints)
                prob.solve()
                minimum_index = x.value >0.1


                traj_combination_matrix = np.zeros((len(traj),len(unique_intensity)))

                for i, j in enumerate(traj):
                    traj_combination_matrix[i][j] = 1
                traj_combination_matrix = traj_combination_matrix.T.astype(dtype=bool)

                traj_inferred_corrected = [i for j in np.argwhere(minimum_index)[:,0].tolist() for i in [traj[j]]]
            else:
                traj_inferred_corrected = [[i] for i in np.unique(test[:,4+number_of_color]).astype(int)]
        return test, traj_inferred_corrected
    


    
    final_trajectories = []    
    final_indexing = 0
    for i in np.unique(final_liposome[:,9+number_of_color]):
        sub_traj = final_liposome[final_liposome[:,9+number_of_color] == i]
        if len(np.unique(sub_traj[:,4+number_of_color])) > 1:
            traj_out, predict_traj_list = traj_optimization(sub_traj)
            for j in predict_traj_list:
                copy_traj = np.copy(traj_out)
                traj_segment_index = j
                copy_traj = copy_traj[np.isin(copy_traj[:,4+number_of_color] , traj_segment_index)]

                no_duplicated_traj = np.ones((len(np.unique(copy_traj[:,2])), 2)) * -1
                no_duplicated_traj[:,0] = np.unique(copy_traj[:,2])

                for unique_segment_index in j:
                    temp_traj = np.copy(copy_traj[copy_traj[:,4+number_of_color] == unique_segment_index])
                    temp_traj = temp_traj[np.argsort(temp_traj[:,2])]
                    no_duplicated_traj[np.isin(no_duplicated_traj[:,0], temp_traj[:,2]),1] = np.argwhere(copy_traj[:,4+number_of_color] == unique_segment_index)[:,0]
                copy_traj = copy_traj[no_duplicated_traj[:,1].astype(int)]
                copy_traj[:,4+number_of_color] = final_indexing
                final_trajectories.append(copy_traj)
                final_indexing = final_indexing+1

        else:
            sub_traj[:,4+number_of_color] = final_indexing
            final_trajectories.append(sub_traj)
            final_indexing = final_indexing+1            
    final_trajectories = np.vstack(final_trajectories)
    final_trajectories = final_trajectories[np.argsort(final_trajectories[:,2])]

    return final_trajectories 

