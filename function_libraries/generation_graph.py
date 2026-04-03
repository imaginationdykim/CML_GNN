# 0:x, 1:y, 2:time, 3:previous, 4:current, 5:unique, 6:FI, 7:FI, 8:FI, 9:SUM

import numpy as np
from tqdm import tqdm
from itertools import repeat
import itertools
from multiprocessing import Pool
from torch_geometric.data import Data
import torch
from function_libraries.training_function import simulated_liposome, point_mixture_jumping, point_mixture_exact, blinking_deletion, plot_graph, add_false_positive


traj = np.genfromtxt("sample_data/sample_data.csv", delimiter= ",")

def correlation_function(tail,head,Range_a, Range_b, sensitivity, one_to_one):   # The maximum sensitivity is 1, the minimum is 0
    if one_to_one == True:
        tail = tail[:,Range_a:Range_b]
        head = head[:,Range_a:Range_b]
        
        if (tail.sum() == 0) | (head.sum() == 0):
            corr = 0
        else:
            product = (tail * head).sum(axis=1)
            product = product/   np.sqrt( (tail**2).sum(axis=1) * (head**2).sum(axis=1)   ) 
            corr = product/abs(product + 0.000001) * abs(product) ** sensitivity
    else:    
        tail = tail[:,Range_a:Range_b][:,np.newaxis]
        head = head[:,Range_a:Range_b]
        product = (tail * head).sum(axis=2) 
        product = product/   np.sqrt( (tail**2).sum(axis=2) * (head**2).sum(axis=1)   ) 
        corr = product/abs(product+ 0.000001) * abs(product+ 0.000001) ** sensitivity
    return corr


def intensity_function_merging_split(tail,head,Range_a, Range_b, sensitivity, one_to_one):
        tail_intensity = tail[:,Range_a:Range_b]
        head_intensity = head[:,Range_a:Range_b]
        tail_mean =  np.sqrt((tail_intensity**2).sum())
        head_mean =  np.sqrt((head_intensity**2).sum())
        #tail_mean =  np.sqrt(((tail_intensity).sum())**2)
        #head_mean =  np.sqrt(((head_intensity).sum())**2)
        mean = (tail_mean + head_mean)/2

        substraction_intensity = tail_intensity.sum() -  head_intensity.sum() 
        corr = (mean-substraction_intensity)/mean**sensitivity
        return corr

def intensity_function(tail,head,Range_a, Range_b, sensitivity, one_to_one):

    if one_to_one == True:
        tail_intensity = tail[:,Range_a:Range_b]
        head_intensity = head[:,Range_a:Range_b]
        #tail_intensity = tail_intensity**2
        #tail_intensity = tail_intensity.sum(axis=1)
        tail_intensity = (tail_intensity.sum(axis = 1))**2

        #head_intensity = head_intensity**2
        #head_intensity = head_intensity.sum(axis=1)
        head_intensity = (head_intensity.sum(axis = 1))**2
        
        cor_matrix = (2* np.sqrt(tail_intensity * head_intensity) / (tail_intensity + head_intensity) ) **sensitivity 

    else:
        tail_intensity = tail[:,Range_a:Range_b]
        head_intensity = head[:,Range_a:Range_b]
        tail_intensity = tail_intensity**2
        tail_intensity = tail_intensity.sum(axis=1)

        tail_intensity = tail_intensity[:, np.newaxis] 
        head_intensity = (head_intensity**2)
        head_intensity = head_intensity.sum(axis=1)
        cor_matrix = (2* np.sqrt(tail_intensity * head_intensity) / (tail_intensity + head_intensity) ) **sensitivity 
    return cor_matrix 
     

def distance_function(tail,head, one_to_one):
    tail_x, tail_y = tail[:,0], tail[:,1]
    head_x, head_y = head[:,0], head[:,1]
    if one_to_one == True:
        pass
    else:
        tail_x, tail_y = tail_x[:, np.newaxis], tail_y[:, np.newaxis]    

    distance_matrix = np.sqrt( (tail_x - head_x)**2  +  (tail_y - head_y)**2)
    #distance_matrix = (600-distance_matrix)/600
    #distance_matrix = distance_matrix * (distance_matrix>0)*1
    return distance_matrix


def distance_corr(distance, thres_distance):
    distance_1 = np.copy(distance)
    distance_1[distance_1 < thres_distance] = 1
    distance_1[distance_1 >= thres_distance] = 0
    return distance_1


def sigmoid_distance_function(distance, sensitivity):
    slop = 1
    curve = 90
    

    exponent_term = -slop * (curve - distance) 
    
    exponent_term = np.clip(exponent_term, -40, 40)
    
    sigmoid = 1 / (1 + np.exp(exponent_term))

    decay = np.exp(-distance**0.62 / 100)

    result = (sigmoid * decay)**sensitivity
    
    return result

def time_function(tail, head, one_to_one):
    if one_to_one == True:
        time_corr = tail[:,2] - head[:,2] 
    else:
        time_corr = tail[:,2][:,np.newaxis] - head[:,2]
    time_corr = time_corr * -1
    time_corr[time_corr<0] = abs(time_corr[time_corr<0]) + 10 
    return time_corr

def time_function_minus(tail, head, one_to_one):
    if one_to_one == True:
        time_corr = tail[:,2] - head[:,2] 
    else:
        time_corr = tail[:,2][:,np.newaxis] - head[:,2]
    time_corr = time_corr * -1

    return time_corr


def sigmoid_time_function(time_corr, sensitivity,one_to_one):
    slop = 0.05
    time_corr = time_corr - 1
    time_corr[time_corr<=-1] = abs(time_corr[time_corr<=-1]) + 1
    translation = 50
    a_0 = np.exp( -slop*(0-translation))/(np.exp(-slop*(0-translation)) + 1) *  np.exp( -0**0.4/100)
    return (np.exp( -slop*(time_corr-translation))/(np.exp(-slop*(time_corr-translation)) + 1) *  np.exp( -time_corr**0.4/100)/a_0)** sensitivity

def make_continuous(liposome_noise):
    
    unique_liposome = np.unique(liposome_noise[:,4]).astype(int)
    unique_length = len(unique_liposome)
    continuous = np.arange(unique_length)

    one_to_one = np.column_stack((continuous, unique_liposome))
    transform = np.arange(unique_liposome.max() + 2)
    transform[-1] = -1
    transform[one_to_one[:,1]] = one_to_one[:,0]
    liposome_noise[:,4] = transform[liposome_noise[:,4].astype(int)]
    liposome_noise[:,3] = transform[liposome_noise[:,3].astype(int)]
    return liposome_noise


def sigmoid_distance_function_traj(distance, sensitivity):
    slop=0.5
    curve=15
    sigmoid = 1 / (1 + np.exp(-slop * (curve - distance)+ 0.00001))
    decay = np.exp(-distance**0.62 / 100)
    result = (sigmoid * decay)**sensitivity

    return result


def input_traj(number_initial_liposome,
               number_of_color,
               number_merging_liposome,
               merging_distance_thres,
               distance_thres,
               percentage_blinking,
               noise_percentage,
               number_of_weak_FI,
               blinking,
               initial_t,
               end_t,
               concentration,
               number_of_neighboring_liposome,
               merging_GT_strategy,
               combinatorial):
    training_data_completeness = 0

    while training_data_completeness ==0:
        seed_value = np.random.RandomState().randint(100000000)
        np.random.seed(seed_value)


        liposome_clean, merging_GT = simulated_liposome(traj,number_of_color, initial_t,end_t, number_initial_liposome, number_merging_liposome, concentration, number_of_neighboring_liposome,combinatorial) #300 100 test
        #print(1)
        if merging_GT_strategy == "exact":
            liposome_mixture = point_mixture_exact(liposome_clean, merging_distance_thres, number_of_color)
        if merging_GT_strategy == "jumping":
            liposome_mixture = point_mixture_jumping(liposome_clean,number_of_color, merging_distance_thres, blinking)
        #print(2)
        liposome_blinking = blinking_deletion(liposome_mixture, percentage_blinking)
        #print(3)
        liposome_noise  = add_false_positive(liposome_blinking,number_of_color, noise_percentage,number_of_weak_FI, traj,merging_distance_thres,combinatorial )    #input, % of noise relative to the total liposome
        #print(4)

        liposome_noise = make_continuous(liposome_noise)
        np.random.shuffle(liposome_noise)        



        #plot_graph(liposome_blinking)

        #print(liposome_clean.shape, liposome_mixture.shape, liposome_blinking.shape, liposome_noise.shape)




        Range_a = 6
        Range_b = 6 + number_of_color
        sensitivity = 1

        liposome_input = np.copy(liposome_noise)



        
        unique, index = np.unique(liposome_input[:,4], return_index=True)   #there are duplicated vertexes and need to delete, but the edges are also deleted by deleting vertex, there for need to recover the deleted edges

        deleted_but_GT_edges = liposome_input[~np.isin(np.arange(len(liposome_input)), index),:][:,3:5]
        deleted_but_GT_edges = deleted_but_GT_edges[deleted_but_GT_edges[:,0] != -1]
        liposome_input = liposome_input[index,:]

        total_edge_corr = []

        #input graph construction
        # 0:x, 1:y, 2:time, 3:previous, 4:current, 5:unique, 6:FI, 7:FI, 8:FI, 9:transition
        #list of functions: correlation_matrix, intensity_correlation, distance, sigmoid_distance, time_correlation, sigmoid_time
        for i in np.unique(liposome_input[:,2]):  
            #if i%500 ==0:
            #print(i)

            tail = liposome_input[liposome_input[:,2] == i]
            _, tail_unique_index = np.unique(tail[:,4], return_index=True)
            tail = tail[tail_unique_index]

            #this prevents duplication of edges
            head  = liposome_input[(i<liposome_input[:,2]) & (i+blinking>=liposome_input[:,2])]
            _, head_unique_index = np.unique(head[:,4], return_index = True)
            head = head[head_unique_index]

            #distance correlation
            distance = distance_function(tail, head, one_to_one = False)
            distance = distance.flatten()


            #time difference
            time_difference = abs(tail[:,2][:,np.newaxis] - head[:,2])
            time_difference = time_difference.flatten()


            #intensity difference
            tail_intensity = tail[:,Range_a:Range_b]
            tail_intensity = tail_intensity**2
            tail_intensity = tail_intensity.sum(axis=1)
            tail_intensity = tail_intensity[:,np.newaxis]
            head_intensity = head[:,Range_a:Range_b]
            head_intensity = head_intensity**2
            head_intensity = head_intensity.sum(axis=1)
            intensity_cor = (2* np.sqrt(tail_intensity * head_intensity) / (tail_intensity  + head_intensity) ) 
            intensity_cor = intensity_cor.flatten()


            #intensity product
            tail_norm = tail[:,Range_a:Range_b]/np.sqrt((tail[:,Range_a:Range_b]**2).sum(axis=1))[:,np.newaxis]
            head_norm = head[:,Range_a:Range_b]/np.sqrt((head[:,Range_a:Range_b]**2).sum(axis=1))[:,np.newaxis]
            product_norm = (tail_norm[:,np.newaxis] * head_norm).sum(axis=2)
            product_norm = product_norm.flatten()
            

            #tail and head index
            tail_index_list = np.repeat(tail[:,4], len(head)) 
            head_index_list = np.tile(head[:,4], len(tail))

            total_edge_corr_temp = np.column_stack((tail_index_list, head_index_list, distance, time_difference, intensity_cor,product_norm ))

            total_edge_corr_temp = total_edge_corr_temp[total_edge_corr_temp[:,2] < distance_thres]
            total_edge_corr.append(total_edge_corr_temp)

        total_edge_corr = np.vstack(total_edge_corr)

        GT_edges = np.copy(liposome_input[liposome_input[:,3] != -1][:,3:5])
        GT_edges = np.vstack((GT_edges, deleted_but_GT_edges))     #adding deleted edges due to the duplication of the vertexes from the biginning
        edge_corr = total_edge_corr[:,2:6]
        input_edges = np.copy(total_edge_corr[:,0:2]).astype(int)

        input_edges = input_edges[:,0] + input_edges[:,1]*1j
        GT_edges = GT_edges[:,0] + GT_edges[:,1]*1j

        GT = np.isin(input_edges, GT_edges) * 1
        input_GT_sum = GT.sum()
        number_of_GT = len(GT_edges)

        if input_GT_sum == number_of_GT:
            training_data_completeness = 1
            liposome_input = liposome_input[np.argsort(liposome_input[:,4].astype(int))]
        else:
            pass

            
    return liposome_input, total_edge_corr, GT , edge_corr, merging_GT

""" slow version of the above one
a,b = np.unique(input_edges, return_counts=True)
a,b = np.unique(total_edge_corr[:,0:2], return_counts = True, axis = 0)
c = np.column_stack((a,b))
c = np.column_stack((a,b))
c[c[:,2]>=2]
"""


# 0:x, 1:y, 2:time, 3:previous, 4:current, 5:unique, 6:FI, 7:FI, 8:FI, 9:SUM
#tail_index, edges_index, distance, product, intensity




def gen_graph(num_graph,  
              number_of_color,
              number_initial_liposome,
              number_merging_liposome,
              merging_distance_thres,
              distance_thres,
              percentage_blinking,
              noise_percentage,
              number_of_weak_FI,
              blinking,
              initial_t,
              end_t,
              concentration,
              number_of_neighboring_liposome,
              merging_GT_strategy,
              deterministic,
              combinatorial):

    

    if deterministic == False:
        number_of_neighboring_liposome = np.random.randint(0,number_of_neighboring_liposome + 1)
        noise_percentage = np.random.randint(0,noise_percentage + 1)
        percentage_blinking = np.random.randint(2,percentage_blinking + 1)
        number_of_weak_FI = np.random.randint(1, number_of_weak_FI + 1)
    else:
        number_of_neighboring_liposome = number_of_neighboring_liposome
        noise_percentage = noise_percentage 
        percentage_blinking = percentage_blinking 
        number_of_weak_FI = number_of_weak_FI
    
    
    input_nodes, input_edges, input_GT, edge_corr, merging_GT = input_traj(number_initial_liposome, 
                                                    number_of_color,
                                                    number_merging_liposome,
                                                    merging_distance_thres,
                                                    distance_thres,
                                                    percentage_blinking,
                                                    noise_percentage, 
                                                    number_of_weak_FI,
                                                    blinking,
                                                    initial_t,
                                                    end_t,
                                                    concentration,
                                                    number_of_neighboring_liposome,
                                                    merging_GT_strategy,
                                                    combinatorial)
    

    neighbor_edges = []

    for i in np.unique(input_nodes[:,2]):
        temp_index = np.argwhere(input_nodes[:,2] == i)[:,0]
        temp = input_nodes[temp_index,:]
        neighbor_distance = distance_function(temp,temp, one_to_one = False)

        temp_edges = np.argwhere((neighbor_distance <distance_thres) & (neighbor_distance !=0))
        temp_edges[:,0] = temp_index[temp_edges[:,0]]
        temp_edges[:,1] = temp_index[temp_edges[:,1]]
        neighbor_edges.append(temp_edges)

    neighbor_edges = np.vstack(neighbor_edges)    

    edge_index = input_edges[:,0:2].astype("int64")


    ### message propagation between adjacent vertices
    neighbor_edges_index = neighbor_edges[:,0:2].astype("int64")

    if neighbor_edges_index.shape[0] != 0:
        edge_neighbor_graph = neighbor_edges_index
    else:
        edge_neighbor_graph = np.empty((2,0))

        
    if number_of_color == 1:
        X = input_nodes[:,[0,1,2,6,3,4,5]] 
    if number_of_color == 2:
        X = input_nodes[:,[0,1,2,6,7,3,4,5]] 

    if number_of_color == 3:
        X = input_nodes[:,[0,1,2,6,7,8,3,4,5]] 
    if number_of_color == 4:

        X = input_nodes[:,[0,1,2,6,7,8,9,3,4,5]]
        
    data = Data(x = torch.from_numpy(X).float(), 
                y = torch.from_numpy(input_GT), 
                edge_index = torch.from_numpy(edge_index.T).type(torch.LongTensor) , 
                neighbor_edges = torch.from_numpy(edge_neighbor_graph.T).type(torch.LongTensor),
                edge_corr = torch.from_numpy(edge_corr).float() ) #,, neighbor_edges = edge_neighbor_graph

    return data, merging_GT





def parallel_gen_graph(num_graph,
                       number_of_color,
                    number_initial_liposome,
                    number_merging_liposome,
                    merging_distance_thres,
                    distance_thres,
                    percentage_blinking,
                    noise_percentage,
                    number_of_weak_FI,
                    blinking,
                    initial_t,
                    end_t,
                    random_time_interval,
                    concentration,
                    number_of_neighboring_liposome,
                    merging_GT_strategy,
                    deterministic,
                    combinatorial,
                      number_of_core):
    

    data = []
    

    num_cores = number_of_core
    
    if random_time_interval !=0:
        
        initial_t_repeat = np.random.randint(1, 2500 - random_time_interval, num_graph)
        end_t_repeat     = initial_t_repeat + random_time_interval
    
    elif random_time_interval == 0:
        
        initial_t_repeat = itertools.repeat(initial_t)
        end_t_repeat = itertools.repeat(end_t)
        

    with Pool(num_cores) as pool:
        data = pool.starmap(gen_graph, tqdm(zip(np.arange(num_graph),
                                                itertools.repeat(number_of_color),
                                                itertools.repeat(number_initial_liposome),
                                                
                                                itertools.repeat(number_merging_liposome), 
                                                itertools.repeat(merging_distance_thres),
                                                itertools.repeat(distance_thres),   
                                                itertools.repeat(percentage_blinking),
                                                itertools.repeat(noise_percentage), 
                                                itertools.repeat(number_of_weak_FI),
                                                itertools.repeat(blinking), 
                                                initial_t_repeat,
                                                end_t_repeat,
                                                itertools.repeat(concentration),
                                                itertools.repeat(number_of_neighboring_liposome),
                                                itertools.repeat(merging_GT_strategy),
                                                itertools.repeat(deterministic),
                                                itertools.repeat(combinatorial)),
                                                
                                            total= num_graph, desc = "Graph generation", position=0, leave=True))
        
    #elapsed_time = time.time() - start_time
    return data

def exp_traj(ex_peaks, number_of_color, blinking, distance_thres):

    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)

    #ex_peaks = ex_peaks_raw[:, [0,1,6,2,3,4]] #so that it becomes x y t FI FI FI
    #print(liposome_clean.shape, liposome_mixture.shape, liposome_blinking.shape, liposome_noise.shape)
    Range_a = 3
    Range_b = 6 + number_of_color


    liposome_input = np.copy(ex_peaks)
    total_edge_corr = []




    #input graph construction
    # 0:x, 1:y, 2:time, 3:previous, 4:current, 5:unique, 6:FI, 7:FI, 8:FI, 9:transition
    #list of functions: correlation_matrix, intensity_correlation, distance, sigmoid_distance, time_correlation, sigmoid_time
    for i in np.unique(liposome_input[:,2]):   


        tail_index = np.argwhere(liposome_input[:,2] == i)[:,0]
        head_index = np.argwhere((i<liposome_input[:,2]) & (i+blinking>=liposome_input[:,2]))[:,0]

        tail = liposome_input[tail_index]
        head = liposome_input[head_index]

        #distance correlation
        distance = distance_function(tail, head, one_to_one = False)
        distance = distance.flatten()


        #time difference
        time_difference = abs(tail[:,2][:,np.newaxis] - head[:,2])
        time_difference = time_difference.flatten()


        #intensity difference
        tail_intensity = tail[:,Range_a:Range_b]
        tail_intensity = tail_intensity**2
        tail_intensity = tail_intensity.sum(axis=1)
        tail_intensity = tail_intensity[:,np.newaxis]
        head_intensity = head[:,Range_a:Range_b]
        head_intensity = head_intensity**2
        head_intensity = head_intensity.sum(axis=1)
        intensity_cor = (2* np.sqrt(tail_intensity * head_intensity) / (tail_intensity  + head_intensity) ) 
        intensity_cor = intensity_cor.flatten()


        #intensity product
        tail_norm = tail[:,Range_a:Range_b]/np.sqrt((tail[:,Range_a:Range_b]**2).sum(axis=1))[:,np.newaxis]
        head_norm = head[:,Range_a:Range_b]/np.sqrt((head[:,Range_a:Range_b]**2).sum(axis=1))[:,np.newaxis]
        product_norm = (tail_norm[:,np.newaxis] * head_norm).sum(axis=2)
        product_norm = product_norm.flatten()
        

        #tail and head index
        tail_index_list = np.repeat(tail_index, len(head_index)) 
        head_index_list = np.tile(head_index, len(tail_index))
        #distance_correlation = distance_corr(distance, 100)
        total_edge_corr_temp = np.column_stack((tail_index_list, head_index_list, distance, time_difference, intensity_cor,product_norm ))
        #print(total_edge_corr_temp)

        total_edge_corr_temp = total_edge_corr_temp[total_edge_corr_temp[:,2] < distance_thres]
        total_edge_corr.append(total_edge_corr_temp)

    total_edge_corr = np.vstack(total_edge_corr)
    edge_corr = total_edge_corr[:,2:6]
    edge_index = total_edge_corr[:,0:2]
    return ex_peaks[:,:Range_b], edge_index, edge_corr




def exp_gen_graph(ex_peaks_raw, number_of_color, blinking, distance_thres):

    input_nodes, edge_index, edge_corr = exp_traj(ex_peaks_raw,number_of_color, blinking, distance_thres)


    neighbor_edges = []

    for i in np.unique(input_nodes[:,2]):
        temp_index = np.argwhere(input_nodes[:,2] == i)[:,0]
        temp = input_nodes[temp_index,:]
        neighbor_distance = distance_function(temp,temp, one_to_one = False)
        #neighbor_distance = np.tril(neighbor_distance)
        temp_edges = np.argwhere((neighbor_distance <distance_thres) & (neighbor_distance !=0))
        temp_edges[:,0] = temp_index[temp_edges[:,0]]
        temp_edges[:,1] = temp_index[temp_edges[:,1]]
        neighbor_edges.append(temp_edges)

    neighbor_edges = np.vstack(neighbor_edges)    


    ### message propagation between adjacent vertices
    neighbor_edges_index = neighbor_edges[:,0:2].astype("int64")

    if neighbor_edges_index.shape[0] != 0:

        edge_neighbor_graph = neighbor_edges_index   
    else:
        edge_neighbor_graph = torch.empty((2,0), dtype = torch.int64)

    edge_index = edge_index.astype("int64")

    X = input_nodes  #Features x,y,t,FI-1, FI-2, FI-3 


    data = Data(x = torch.from_numpy(X).float(),
                edge_index = torch.from_numpy(edge_index.T).type(torch.LongTensor), 
                edge_corr = torch.from_numpy(edge_corr).float(),
                neighbor_edges = torch.from_numpy(edge_neighbor_graph.T).type(torch.LongTensor))

    return data

    
    
    

    