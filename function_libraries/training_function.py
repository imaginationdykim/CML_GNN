
import numpy as np
import pandas as pd
from scipy.stats import gamma
import networkx as nx
from math import comb


def sampling_from_experimental_data(traj, traj_1, transition, fast):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
   
    traj_tran = np.column_stack((traj_1, transition))
    traj_fast = traj_tran[np.isin(traj_tran[:,0], fast )]
    internalization_distribution = []
    
    for i in traj_tran:
        if i[1] != -1:
            temp = traj[traj[:,7] == i[0]]
            into_time = temp[0,6]
            duration = i[1] - into_time
            internalization_distribution.append(duration)
     
    shape_internalization,b,scale_internalization = gamma.fit(internalization_distribution, floc=0)
    
    np.random.gamma(shape_internalization, scale_internalization)
    
            
    
    
    on_membrane_displacement = np.zeros((0,2))
    for i in traj_tran:
        temp = traj[traj[:,7] == i[0]]
        if i[1] != -1:
            
            temp = temp[temp[:,6] < i[1]]
            temp = temp[10:]
            temp_dis = temp[1:,0:2] - temp[:-1, 0:2]
            np.random.shuffle(temp_dis)
            relative_theta = np.random.uniform(low = 0, high = 2* np.pi)
            rotation_matrix = np.array([[np.cos(relative_theta), -np.sin(relative_theta)], 
                                [np.sin(relative_theta), np.cos(relative_theta)]])
            
            temp_dis = np.inner(rotation_matrix ,temp_dis).T
            on_membrane_displacement = np.vstack((on_membrane_displacement,temp_dis ))
        
        else:
            temp = temp[:200,]
            temp_dis = temp[1:,0:2] - temp[:-1, 0:2]
            on_membrane_displacement = np.vstack((on_membrane_displacement,temp_dis ))
    
    
    
        
    limit_dis = 10
    on_membrane_displacement = on_membrane_displacement[on_membrane_displacement[:,0] < limit_dis]
    on_membrane_displacement = on_membrane_displacement[on_membrane_displacement[:,0] > -limit_dis]
    on_membrane_displacement = on_membrane_displacement[on_membrane_displacement[:,1] < limit_dis]    
    on_membrane_displacement = on_membrane_displacement[on_membrane_displacement[:,1] > -limit_dis]    




    in_cytosol_displacement = np.zeros((0,2)) 
    for i in traj_tran:
        temp = traj[traj[:,7] == i[0]]
        if i[1] != -1:
            temp = temp[temp[:,6] > i[1]]
            
            temp_dis = temp[1:,0:2] - temp[:-1, 0:2]
            np.random.shuffle(temp_dis)
            in_cytosol_displacement = np.vstack((in_cytosol_displacement,temp_dis ))
        else:
            pass
    
    
    in_cytosol_displacement_fast = np.zeros((0,2)) 
    for i in traj_fast:
        temp = traj[traj[:,7] == i[0]]
        if i[1] != -1:
            temp = temp[temp[:,6] > i[1]]
            
            temp_dis = temp[1:,0:2] - temp[:-1, 0:2]
            np.random.shuffle(temp_dis)
            in_cytosol_displacement_fast = np.vstack((in_cytosol_displacement_fast,temp_dis ))
        else:
            pass

    return on_membrane_displacement, in_cytosol_displacement, in_cytosol_displacement_fast, shape_internalization, scale_internalization





def intensity_gamma_list(traj, num_list,traj_1, threshold):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    total_gamma_mean = pd.DataFrame()
    for i in traj_1:
        temp = traj[traj[:,7] == i]
            
        for j in range(2,5):
            a,b,c = gamma.fit(temp[:,j], floc=0)
            d = temp[:,j].mean()
            
            temp_list = pd.DataFrame([[i, j, a, c, d]])
            total_gamma_mean = pd.concat([total_gamma_mean, temp_list])
            
    
    total_gamma_mean.columns = ["Merging number", "Channel", "Shape", "Scale", "Mean"]
    total_gamma_mean = total_gamma_mean[total_gamma_mean["Mean"] > threshold]
    shape_mean,b,scale_mean= gamma.fit(total_gamma_mean["Mean"])
    simulated_mean = np.random.gamma(shape_mean, scale_mean,num_list)
    slope = total_gamma_mean["Shape"]/total_gamma_mean["Mean"] 
    
    theta_rad = np.arctan(slope)/np.pi * 4
    theta_rad_mean = theta_rad.mean()
    theta_rad_std = theta_rad.std()
    
    
    
    shape_simu = simulated_mean* np.tan(np.random.normal(theta_rad_mean-0.05, theta_rad_std *1, len(simulated_mean)))

    
    scale_simu = simulated_mean/ shape_simu
    
    simu_gamma = np.column_stack((shape_simu, scale_simu,shape_simu* scale_simu))
    

    simu_gamma = simu_gamma[simu_gamma[:,0] < total_gamma_mean["Shape"].max()]
    simu_gamma = simu_gamma[simu_gamma[:,0] > total_gamma_mean["Shape"].min()]
    simu_gamma = simu_gamma[simu_gamma[:,1] < total_gamma_mean["Scale"].max()]
    simu_gamma = simu_gamma[simu_gamma[:,1] > total_gamma_mean["Scale"].min()]

    simu_gamma = simu_gamma[simu_gamma[:,2] > total_gamma_mean["Mean"].min()]
    
    
    
    simu_gamma = np.tile([[100,0.2,20]],(num_list,1)) 

    
    
    simu_gamma = pd.DataFrame(simu_gamma)
    
    
    simu_gamma.columns = ["Shape", "Scale","Mean"]
    return simu_gamma




def on_membrane(on_membrane_displacement,length_traj_menbrane,  inverse_x, inverse_y):
    
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    temp = np.zeros((0,2))
    
    #on membrane movement
    on_membrane_distance_sample_number = on_membrane_displacement.shape[0]
    
    
    if length_traj_menbrane <20:
        length_traj_menbrane = 20
    cuting = np.random.choice(np.arange(1, length_traj_menbrane -1), 10, replace=False)
    cuting = np.append(cuting, [0,length_traj_menbrane])
    cuting = cuting[np.argsort(cuting)]
    cuting_number_list = cuting[1:]-cuting[:-1]
    
    for j in cuting_number_list:        
    
        cuting_start = np.random.choice(on_membrane_distance_sample_number -j)
        cuting_end = cuting_start + j

        segment = np.copy(on_membrane_displacement[cuting_start:cuting_end,:])
        
        
        segment_x = np.cumsum(segment[:,0]) + np.random.normal(0,1, segment.shape[0])
        segment_y = np.cumsum(segment[:,1]) + np.random.normal(0,1, segment.shape[0])
        segment_line = np.column_stack((segment_x, segment_y))
        inverse_xy = np.column_stack((inverse_x, inverse_y))[0]
        segment_dis = segment_line[-1,:]
        relative_theta = np.arctan2(segment_dis[1], segment_dis[0]) - np.arctan2(inverse_xy[1], inverse_xy[0]) + np.random.normal(0,0.5)
        rotation_matrix = np.array([[np.cos(-relative_theta), -np.sin(-relative_theta)], 
                                    [np.sin(-relative_theta), np.cos(-relative_theta)]])
        xy = np.inner(rotation_matrix ,segment).T
        temp = np.vstack((temp,xy))
    return temp




def in_membrane(in_cytosol_displacement, length_traj_cytosol, in_cytosol_displacement_fast,  inverse_x, inverse_y):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    temp = np.zeros((0,2))

    in_cytosole_sample_number_fast = in_cytosol_displacement_fast.shape[0]
    in_cytosole_sample_number = in_cytosol_displacement.shape[0]
    
    if length_traj_cytosol <20:
        length_traj_cytosol = 20    
    
    cuting = np.random.choice(np.arange(1, length_traj_cytosol -1), 10, replace=False)
    cuting = np.append(cuting, [0,length_traj_cytosol])
    cuting = cuting[np.argsort(cuting)]
    cuting_number_list_in = cuting[1:]-cuting[:-1]

    if np.random.random() > 0.5:
        for j in cuting_number_list_in:
            cuting_start = np.random.choice(in_cytosole_sample_number -1)
            cuting_end = cuting_start + j
                
            segment = np.copy(in_cytosol_displacement[cuting_start:cuting_end,:])
            segment = segment + np.random.normal(0, np.random.random(), (segment.shape[0], segment.shape[1]))
            
            
            relative_theta = np.random.uniform(0,np.pi*2, segment.shape[0])
            
            rotation_matrix = np.array([[np.cos(-relative_theta), -np.sin(-relative_theta)], 
                                        [np.sin(-relative_theta), np.cos(-relative_theta)]])
            
            segment = np.einsum("ij...,...i", rotation_matrix,segment)
            
            segment_x = np.cumsum(segment[:,0]) 
            segment_y = np.cumsum(segment[:,1]) 
            segment_line = np.column_stack((segment_x, segment_y))
            inverse_xy = np.column_stack((inverse_x, inverse_y))[0]
            segment_dis = segment_line[-1,:]
            relative_theta = np.arctan2(segment_dis[1], segment_dis[0]) - np.arctan2(inverse_xy[1], inverse_xy[0]) + np.random.normal(0,1)
            rotation_matrix = np.array([[np.cos(-relative_theta), -np.sin(-relative_theta)], 
                                        [np.sin(-relative_theta), np.cos(-relative_theta)]])
            xy = np.inner(rotation_matrix ,segment).T
            temp = np.vstack((temp,xy))
            
            
    #fast liposome movement in cytosol     
    else:
        for j in cuting_number_list_in:
            cuting_start = np.random.choice(in_cytosole_sample_number_fast -1)
            cuting_end = cuting_start + j
                
            segment = np.copy(in_cytosol_displacement_fast[cuting_start:cuting_end,:])
            segment = segment + np.random.normal(0, np.random.random(), (segment.shape[0], segment.shape[1]))
            
            relative_theta = np.random.uniform(0,np.pi*2, segment.shape[0])
            
            rotation_matrix = np.array([[np.cos(-relative_theta), -np.sin(-relative_theta)], 
                                        [np.sin(-relative_theta), np.cos(-relative_theta)]])
            
            segment = np.einsum("ij...,...i", rotation_matrix,segment)
            
            segment_x = np.cumsum(segment[:,0]) #+ np.random.normal(0,2, segment.shape[0])
            segment_y = np.cumsum(segment[:,1]) #+ np.random.normal(0,2, segment.shape[0])
            segment_line = np.column_stack((segment_x, segment_y))
            inverse_xy = np.column_stack((inverse_x, inverse_y))[0]
            segment_dis = segment_line[-1,:]
            relative_theta = np.arctan2(segment_dis[1], segment_dis[0]) - np.arctan2(inverse_xy[1], inverse_xy[0]) + np.random.normal(0,1)
            rotation_matrix = np.array([[np.cos(-relative_theta), -np.sin(-relative_theta)], 
                                        [np.sin(-relative_theta), np.cos(-relative_theta)]])
            xy = np.inner(rotation_matrix ,segment).T
            temp = np.vstack((temp,xy))        
    
    return temp









def intensity_new(temp_shape,number_of_color, intensity_sampled_list, selected_channel):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    
    intensity_list = np.zeros((temp_shape,number_of_color))
    selected =  np.copy(selected_channel)
    background = np.where(np.isin(np.arange(number_of_color),selected, invert = True))[0]

    for i in selected:
        intensity_list[:,i] = np.random.gamma(intensity_sampled_list[i,0], intensity_sampled_list[i,1],temp_shape)

        intensity_list_background = np.copy(intensity_list)
    for i in background:
        intensity_list_background[:,i]  = np.random.gamma(1, number_of_color, temp_shape)

    return intensity_list + intensity_list_background



def single_trajectory(unique_liposome,
                      number_of_color,
                      object_liposome,
                      image_size_x,
                      image_size_y, 
                      direction, 
                      on_membrane_displacement, 
                      in_cytosol_displacement,
                      in_cytosol_displacement_fast,
                      length_traj_menbrane,
                      length_traj_cytosol,
                      initial_length,
                      initial_theta,
                      initial_x_m,
                      initial_y_m,
                      initial_t,
                      end_t,
                      merging_time,
                      intensity_sampled_list,
                      combinatorial
                      ):

    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    
    intensity_sampled = np.array(intensity_sampled_list.sample(n=number_of_color, random_state = None))
    


    partition = np.array([comb(number_of_color,i) for i in range(1,number_of_color + 1)])
    partition = partition/sum(partition)

    if combinatorial == True:
        color_combination = np.random.choice(number_of_color, 1, p = partition) + 1
    else:
        color_combination = 1
    selected_channel = np.random.choice(np.arange(number_of_color),color_combination, replace=False )


    center_x = image_size_x/2
    center_y = image_size_y/2


    if direction == "Forward":
        
        initial_x = initial_length * np.cos(initial_theta)
        initial_y = initial_length * np.sin(initial_theta)
    elif direction =="Backward":
        initial_x = initial_x_m
        initial_y = initial_y_m

    
    if direction == "Forward":
        inverse_x = - initial_x
        inverse_y = - initial_y
        
    elif direction == "Backward":
        inverse_x =  initial_x
        inverse_y =  initial_y
        
    
    if direction == "Forward":    
        temp_on_membrane = on_membrane(on_membrane_displacement, length_traj_menbrane, inverse_x = inverse_x, inverse_y = inverse_y)
        temp_in_membrane = in_membrane(in_cytosol_displacement, length_traj_cytosol, in_cytosol_displacement_fast,  inverse_x, inverse_y)
        
        temp = np.vstack((temp_on_membrane, temp_in_membrane))
        temp = np.cumsum(temp, axis = 0)
        temp_shape = temp.shape[0]
        intensity_list = intensity_new(temp_shape, number_of_color, intensity_sampled, selected_channel)
            
        #time encoding
        starting_time = np.random.randint(initial_t, initial_t + int((end_t - initial_t)/2)) 
        if unique_liposome == 0:
            starting_time = initial_t
        time_sequence = np.arange(starting_time, temp.shape[0] + starting_time)
    elif direction == "Backward":

        temp_in_membrane = in_membrane(in_cytosol_displacement , length_traj_cytosol, in_cytosol_displacement_fast ,  inverse_x = inverse_x, inverse_y = inverse_y)
        temp_on_membrane = on_membrane(on_membrane_displacement, length_traj_menbrane, inverse_x = inverse_x, inverse_y = inverse_y)

        temp = np.vstack((temp_in_membrane, temp_on_membrane))
        
        
        if initial_t < merging_time-temp_on_membrane.shape[0]:
            starting_merging_liposome = np.random.randint(initial_t, merging_time-temp_on_membrane.shape[0])
        else:
            starting_merging_liposome = initial_t
            
        distance_between_mergingtime_and_starting = merging_time - starting_merging_liposome
        temp = temp[-distance_between_mergingtime_and_starting:] #cutting merging_liposome traj to fit into initial_t
        temp = np.cumsum(temp, axis = 0)
        intensity_list = intensity_new(temp.shape[0],number_of_color, intensity_sampled, selected_channel)    

        time_sequence = np.arange(merging_time,  merging_time-len(temp), -1)
        starting_time = merging_time


        
        
        
    liposome_temp = np.column_stack((temp[:,0], temp[:,1], time_sequence, 
                         np.repeat(0, temp.shape[0]), np.repeat(0, temp.shape[0]), np.repeat(unique_liposome, temp.shape[0]) ,intensity_list[:,0:number_of_color],
np.repeat(temp_on_membrane.shape[0] + starting_time, temp.shape[0])))
        
        

    
    liposome_temp[:,0:2] = liposome_temp[:,0:2] + np.array([initial_x, initial_y]) 
    
    
    liposome_temp = liposome_temp[np.argsort(liposome_temp[:,2])]
    
    liposome_temp = liposome_temp[liposome_temp[:,2] < end_t]
    
    
    return liposome_temp,intensity_sampled ,  selected_channel



def neighboring_trajectory(max_liposome_number,
                           number_of_color,
                      on_membrane_displacement, 
                      in_cytosol_displacement,
                      in_cytosol_displacement_fast,
                      length_traj_menbrane,
                      length_traj_cytosol,
                      initial_x_m,
                      initial_y_m,
                      starting_t,
                      end_t,
                      FI
                      ):

    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    
    
    if number_of_color == 1:
        shape_0, loc, scale_0 = gamma.fit(FI[:len(FI),0], floc=0)

    if number_of_color == 2:
        shape_0, loc, scale_0 = gamma.fit(FI[:len(FI),0], floc=0)
        shape_1, loc, scale_1 = gamma.fit(FI[:len(FI),1], floc=0)

    if number_of_color == 3:
        shape_0, loc, scale_0 = gamma.fit(FI[:len(FI),0], floc=0)
        shape_1, loc, scale_1 = gamma.fit(FI[:len(FI),1], floc=0)
        shape_2, loc, scale_2 = gamma.fit(FI[:len(FI),2], floc=0)
        
    if number_of_color == 4:
        shape_0, loc, scale_0 = gamma.fit(FI[:len(FI),0], floc=0)
        shape_1, loc, scale_1 = gamma.fit(FI[:len(FI),1], floc=0)
        shape_2, loc, scale_2 = gamma.fit(FI[:len(FI),2], floc=0)        
        shape_3, loc, scale_3 = gamma.fit(FI[:len(FI),3], floc=0)   
        
        
    initial_x = initial_x_m
    initial_y = initial_y_m

    inverse_x = - initial_x
    inverse_y = - initial_y

    temp_on_membrane = on_membrane(on_membrane_displacement, length_traj_menbrane, inverse_x = inverse_x, inverse_y = inverse_y)
    temp_in_membrane = in_membrane(in_cytosol_displacement, length_traj_cytosol, in_cytosol_displacement_fast,  inverse_x, inverse_y)

    temp = np.vstack((temp_on_membrane, temp_in_membrane))
    temp = np.cumsum(temp, axis = 0)
    temp_shape = temp.shape[0]

    if number_of_color == 1:
        intensity_list_0 = np.random.gamma(shape_0, scale_0, temp_shape)

        
    if number_of_color == 2:
        intensity_list_0 = np.random.gamma(shape_0, scale_0, temp_shape)
        intensity_list_1 = np.random.gamma(shape_1, scale_1, temp_shape) 

    if number_of_color == 3:
        intensity_list_0 = np.random.gamma(shape_0, scale_0, temp_shape)
        intensity_list_1 = np.random.gamma(shape_1, scale_1, temp_shape) 
        intensity_list_2 = np.random.gamma(shape_2, scale_2, temp_shape)
    if number_of_color == 4:
        intensity_list_0 = np.random.gamma(shape_0, scale_0, temp_shape)
        intensity_list_1 = np.random.gamma(shape_1, scale_1, temp_shape) 
        intensity_list_2 = np.random.gamma(shape_2, scale_2, temp_shape)    
        intensity_list_3 = np.random.gamma(shape_3, scale_3, temp_shape)    
        
    #time encoding

    starting_time = starting_t

    time_sequence = np.arange(starting_time, temp.shape[0] + starting_time)

    if number_of_color == 1:    
        liposome_temp = np.column_stack((temp[:,0], temp[:,1], time_sequence, 
                         np.repeat(0, temp.shape[0]), np.repeat(0, temp.shape[0]), np.repeat(max_liposome_number, temp.shape[0]) ,intensity_list_0, np.repeat(temp_on_membrane.shape[0] + starting_time, temp.shape[0])))

    if number_of_color == 2:    
        liposome_temp = np.column_stack((temp[:,0], temp[:,1], time_sequence, 
                         np.repeat(0, temp.shape[0]), np.repeat(0, temp.shape[0]), np.repeat(max_liposome_number, temp.shape[0]) ,intensity_list_0,
                         intensity_list_1, np.repeat(temp_on_membrane.shape[0] + starting_time, temp.shape[0])))
    

    if number_of_color == 3:    
        liposome_temp = np.column_stack((temp[:,0], temp[:,1], time_sequence, 
                         np.repeat(0, temp.shape[0]), np.repeat(0, temp.shape[0]), np.repeat(max_liposome_number, temp.shape[0]) ,intensity_list_0,
                         intensity_list_1, intensity_list_2, np.repeat(temp_on_membrane.shape[0] + starting_time, temp.shape[0])))
        
        
    if number_of_color == 4:
        
        liposome_temp = np.column_stack((temp[:,0], temp[:,1], time_sequence, 
                         np.repeat(0, temp.shape[0]), np.repeat(0, temp.shape[0]), np.repeat(max_liposome_number, temp.shape[0]) ,intensity_list_0,
                         intensity_list_1, intensity_list_2,intensity_list_3, np.repeat(temp_on_membrane.shape[0] + starting_time, temp.shape[0])))        

    liposome_temp[:,0:2] = liposome_temp[:,0:2] + np.array([initial_x, initial_y]) 
    
    liposome_temp = liposome_temp[np.argsort(liposome_temp[:,2])]
    
    liposome_temp = liposome_temp[liposome_temp[:,2] < end_t]

    return liposome_temp


def gen_simulated(num_liposome,
                  number_of_color,
                  object_liposome,
                  image_size_x,
                  image_size_y,
                  direction,
                  on_membrane_displacement,
                  in_cytosol_displacement,
                  in_cytosol_displacement_fast,
                  length_traj_menbrane,
                  length_traj_cytosol,
                  initial_x_m,
                  initial_y_m,
                  initial_t,
                  end_t,
                  intensity_sampled_list,
                  concentration,
                  number_of_neighboring_liposome,
                  combinatorial):  
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    liposome = np.zeros((0, 10))

    center_x = image_size_x/2
    center_y = image_size_y/2
    initial_length = center_x/2 
    initial_theta  = 0
    
    if concentration == 1:
        initial_length_list = np.random.uniform(low=center_y/2, high = center_y, size = num_liposome)
        initial_theta_list = np.random.uniform(low=0, high = 2*np.pi, size = num_liposome)
    else:
        initial_length_list = np.repeat(initial_length, num_liposome)
        initial_theta_list = np.repeat(initial_theta, num_liposome).astype(float)
        length_additional = len(initial_length_list)

        initial_length_list =  np.random.uniform(low=center_y/1.1, high = center_y, size=length_additional)
        initial_theta_list = initial_theta_list + np.random.uniform(low=-concentration/360*np.pi, high = concentration/360*np.pi, size=length_additional)
    liposome = []
    for i in range(0,num_liposome):
        liposome_temp, _ ,_ = single_trajectory(unique_liposome = i, 
                                                number_of_color = number_of_color,
                                                object_liposome = object_liposome,
                                              image_size_x = image_size_x, image_size_y = image_size_y,
                                              direction = "Forward",
                                              on_membrane_displacement = on_membrane_displacement,
                                              in_cytosol_displacement = in_cytosol_displacement,
                                              in_cytosol_displacement_fast = in_cytosol_displacement_fast,
                                              length_traj_menbrane = length_traj_menbrane[i],
                                              length_traj_cytosol = length_traj_cytosol,
                                              initial_length = initial_length_list[i],
                                              initial_theta = initial_theta_list[i],
                                              initial_x_m = initial_x_m,
                                              initial_y_m = initial_y_m,
                                              initial_t = initial_t,
                                              end_t = end_t,
                                              merging_time = 0, # this variable is only used in Backward
                                              intensity_sampled_list = intensity_sampled_list,
                                              combinatorial = combinatorial)
        liposome.append(liposome_temp)
    liposome = np.vstack((liposome)) 

        
        
    #### neighboring liposome
    list_of_unique_liposome, counts = np.unique(liposome[:,5], return_counts=True)   #finding initial attachements
    list_of_unique_liposome = np.column_stack((list_of_unique_liposome, counts))
    list_of_unique_liposome = list_of_unique_liposome[:,0][list_of_unique_liposome[:,1] > 20] #finding liposomes that have more than 20 time points
    
    if len(list_of_unique_liposome)> 0:
        selected_liposome = np.random.choice(list_of_unique_liposome, number_of_neighboring_liposome, replace = True)
        max_liposome_number = list_of_unique_liposome.max() + 1
        liposome_neighbor = []
        for i in selected_liposome:
            selected_liposome_attachment = liposome[liposome[:,5] == i]
            xy = selected_liposome_attachment[0,0:2]
            t = selected_liposome_attachment[0,2] + np.random.randint(10)

            starting_t = int(t + np.random.normal(0, 10))
            if starting_t < initial_t:
                starting_t = initial_t

            selected_liposome_attachment = liposome[liposome[:,5] == np.random.choice(selected_liposome,1)]
            FI = selected_liposome_attachment[:20,6:6+number_of_color]



            liposome_neighbor_temp = neighboring_trajectory(max_liposome_number,
                                                            number_of_color = number_of_color,
                                                          on_membrane_displacement = on_membrane_displacement, 
                                                          in_cytosol_displacement = in_cytosol_displacement,
                                                          in_cytosol_displacement_fast = in_cytosol_displacement_fast,
                                                          length_traj_menbrane = int(np.random.gamma(10,10 )),
                                                          length_traj_cytosol = end_t - starting_t ,
                                                          initial_x_m = xy[0] + np.random.normal(0, 10),
                                                          initial_y_m = xy[1] + np.random.normal(0, 10),
                                                          starting_t = starting_t ,
                                                          end_t = end_t,
                                                          FI = FI
                                                          )
            liposome_neighbor.append(liposome_neighbor_temp)
            max_liposome_number = max_liposome_number + 1 
        if number_of_neighboring_liposome > 0:
            liposome_neighbor = np.vstack(liposome_neighbor)
            liposome = np.vstack((liposome, liposome_neighbor))
    


    return liposome

def merging_liposome(liposome, 
                     number_of_color,
                     on_membrane_displacement, 
                     in_cytosol_displacement, 
                     in_cytosol_displacement_fast, 
                     shape_internalization, 
                     scale_internalization, 
                     intensity_sampled_list, 
                     number_merging_liposome,
                     initial_t, 
                     end_t,
                     combinatorial):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    #merging
    
    max_time = end_t-2
    unique_liposome, indexes_liposome = np.unique(liposome[:,5], return_index = True)
    
    liposome_transition = np.column_stack((unique_liposome, liposome[indexes_liposome,6+number_of_color]))
    
    liposome_transition = liposome_transition[liposome_transition[:,1] < max_time]
    
    j = int(liposome[:,5].max()) + 1
    
    list_merged = []
    merging_target = np.zeros([0,3])
    
    if len(liposome_transition[:,0]) > 0:
        list_merged = np.random.choice(liposome_transition[:,0], number_merging_liposome, replace = True)
    
    if len(list_merged) > 0 :
        for i in list_merged:
            
            
            temp = liposome[liposome[:,5] == i]
            if temp.shape[0] > 0:
                starting_a = int(temp[0,2])       #Stating time of liposome a

                transition_a = int(temp[0,6+ number_of_color])     #time of internalization of liposome a

                merging_time_b = np.random.randint(transition_a, max_time) #merging of liposome b into liposome a happens after a is internalized

                length_traj_membrane = merging_time_b  #the possible length of liposome b on membrane before merging into liposome a
                length_traj_cytosol = merging_time_b  #the possible length of liposome b in membrane before mering into liposome a 

                while (length_traj_membrane +length_traj_cytosol)  >=  merging_time_b:
                    #length_traj_membrane = int(np.random.gamma(shape_internalization, scale_internalization))
                    length_traj_membrane = int(np.random.gamma(10, 10))
                    length_traj_cytosol = np.random.randint(1, end_t - initial_t)


                index_merging = temp[:,2] == merging_time_b 
                if len(temp[index_merging, 0])>0:
                    merging_liposome_traj, intensity_sampled, selected_channel = single_trajectory(unique_liposome = j, 
                                                                                              number_of_color = number_of_color,
                                                                                              object_liposome = i, image_size_x = 1500, image_size_y =    1500,                                                   
                                                                                   direction = "Backward",
                                                                                   on_membrane_displacement = on_membrane_displacement,
                                                                                   in_cytosol_displacement = in_cytosol_displacement,
                                                                                   in_cytosol_displacement_fast = in_cytosol_displacement_fast,
                                                                                   length_traj_menbrane = length_traj_membrane,
                                                                                   length_traj_cytosol = length_traj_cytosol,
                                                                                   initial_length = 0,
                                                                                   initial_theta = 0,  
                                                                                   initial_x_m = temp[index_merging, 0][0],
                                                                                   initial_y_m = temp[index_merging, 1][0],
                                                                                   initial_t = initial_t,
                                                                                   end_t = end_t,
                                                                                   merging_time = merging_time_b,
                                                                                   intensity_sampled_list = intensity_sampled_list,
                                                                                   combinatorial = combinatorial)                                    

                    object_rest_length = (temp[:,2] >= merging_time_b).sum()

                    adding_intensity = intensity_new(object_rest_length, number_of_color, intensity_sampled, selected_channel)


                    temp[:,6:6+number_of_color][temp[:,2] >= merging_time_b] = temp[:,6:6+number_of_color][temp[:,2] >=merging_time_b] + adding_intensity*0.8

                    merging_liposome_traj = merging_liposome_traj[:-1]

                    merging_liposome_traj = start_end_assigment(merging_liposome_traj, liposome.shape[0])
                    merged_moment_a = np.copy(temp[index_merging ])
                    merged_moment_a[0][3] = merging_liposome_traj[-1,4]
                    merged_moment_a[0][5] = j 

                    merging_liposome_traj = np.vstack((merging_liposome_traj, merged_moment_a))
                    liposome[liposome[:,5] == i] = temp           
                    temp_merging_target = np.array([j , i, merging_time_b])
                    merging_target = np.vstack((merging_target, temp_merging_target))

                    liposome = np.vstack((liposome, merging_liposome_traj))    
                    j = j + 1
    return liposome, merging_target


def start_end_assigment(liposome, initial):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    j = initial
    for i in np.unique(liposome[:,5]):
        
        len_temp = (liposome[:,5]==i).sum()


        s_e = np.zeros((len_temp,2))    
        s_e[:,0] = np.arange(j, j + len_temp)
        s_e[:,1] = np.arange(j+1, j + len_temp+1)
        s_e[0,0] = -1
        
        j = j + len_temp
        liposome[:,3:5][liposome[:,5] == i] = s_e
    return liposome


def simulated_liposome(traj, 
                       number_of_color,
                       initial_t, 
                       end_t, 
                       number_initial_liposome, 
                       number_merging_liposome, 
                       concentration, 
                       number_of_neighboring_liposome,
                       combinatorial): #concentration 20: the degree of liposome localzation. 1 = random for whole radial direction
    
    
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    ## experimental data input
    # traj_1: selected liposome number
    # transition: liposome internalization time
    # fast: liposome numbers that move fast in the cytosol
    traj_1 = [54, 70, 23, 159, 1, 100, 89, 137,
              77,81,115, 6, 49, 308, 162, 292,
              65, 47, 97, 163, 20, 161, 34, 85,
              2, 42, 84, 76, 117, 74, 119, 324,
              520, 313, 415, 43, 431,  186,
              323, 51, 166, 24, 344, 46, 98,
              101, 363, 361, 287, 368, 481,
              480, 5, 416, 94, 64, 199, 133, 192,
              128, 83, 320, 150, 135, 52, 111, 131,
              103, 75, 366, 66, 160, 174, 169, 72,
              173, 114, 71, 80, 1000, 1001, 1002, 1003, 1004] #
              
    
    
    transition = [497, -1, -1, -1, 214, 828, 352, 762,
                  -1, 756, 945, 839, -1, 1561, 334, 1325,
                  -1, -1, -1, -1, 854, -1, -1, 776,
                  1163, 827, 250, 683, 1526, 1262, 472, -1,
                  -1, 1472, -1, 1195, -1, -1,
                  -1, 534, -1, 388, -1, 1360, 996,
                  -1, 1860, -1, 1675,  -1, 1881,
                  -1, 259, 1761, -1, 170, 568, 333, 1071,
                  1234, 410, -1, 750, 414, -1, 338, 300,
                  -1, -1, 1677, -1, 303, 1015, 381, 520,
                  -1, -1, -1,518, 203, 630, 221, 265, 191  ]
    
    fast = [1000 , 1001, 1002, 1003, 1004] 
    
    intensity_sampled_list = intensity_gamma_list(traj, 1000 , traj_1, 0)  #input traj, sampled gamma distribution, selected liposome, threshold
    
    
    
    on_membrane_displacement, in_cytosol_displacement, in_cytosol_displacement_fast, shape_internalization, scale_internalization = sampling_from_experimental_data(traj, traj_1, transition, fast)
    #need to know shape and scale for internalization
    #simulated individual liposoem trajectories
    liposome= gen_simulated(num_liposome = number_initial_liposome,
                            number_of_color = number_of_color,
                            object_liposome = False,
                            image_size_x = 1500,
                            image_size_y = 1500,
                            direction = "Forward",
                            on_membrane_displacement = on_membrane_displacement,
                            in_cytosol_displacement = in_cytosol_displacement,
                            in_cytosol_displacement_fast = in_cytosol_displacement_fast,
                            length_traj_menbrane = (np.random.gamma(10,10, number_initial_liposome )).astype(int) ,  
                            length_traj_cytosol = end_t - initial_t,
                            initial_x_m = 0,
                            initial_y_m = 0,
                            initial_t = initial_t,
                            end_t = end_t,
                            
                            intensity_sampled_list = intensity_sampled_list,
                            concentration = concentration,
                            number_of_neighboring_liposome = number_of_neighboring_liposome,
                            combinatorial = combinatorial)

    liposome = start_end_assigment(liposome, initial = -1)
    

    #addition of merging liposomes
    liposome, merging_GT = merging_liposome(liposome, 
                                number_of_color,
                                on_membrane_displacement, 
                                in_cytosol_displacement, 
                                in_cytosol_displacement_fast, 
                                shape_internalization, 
                                scale_internalization, 
                                intensity_sampled_list, 
                                number_merging_liposome, initial_t,end_t,combinatorial)

    #set the center x = 750, y = 750
    liposome[:,0] = liposome[:,0] + 750 
    liposome[:,1] = liposome[:,1] + 750 

    final_liposome = []
    for i in np.unique(liposome[:,5]):
        temp = liposome[liposome[:,5] == i]

        temp = temp[temp[:,2] <end_t]
        temp = temp[temp[:,2] >0]
        final_liposome.append(temp)

    final_liposome = np.vstack(final_liposome)
    
    return final_liposome, merging_GT


def ID_change_duplication(list_ID_changing):
    i = 0
    while i < len(list_ID_changing):
        
        j = i + 1
        while j < len(list_ID_changing):
            
            overlap = set(list_ID_changing[i]) & set(list_ID_changing[j])
            if len(overlap) > 0:
                list_ID_changing[i] = list( set(list_ID_changing[i]) | set(list_ID_changing[j]))
                list_ID_changing.pop(j)
           
            if len(overlap) == 0:
                j = j + 1
        
        i = i + 1
        
    for i in list_ID_changing:
        i.sort()
    
    return list_ID_changing

def distance(current,previous):
    current_x, current_y = current[:,0], current[:,1]
    previous_x, previous_y = previous[:,0], previous[:,1]
    current_x, current_y = current_x[:, np.newaxis], current_y[:, np.newaxis]    
    distance_matrix = np.sqrt( (current_x - previous_x)**2  +  (current_y - previous_y)**2)
    return distance_matrix

#graph edge contraction
def point_mixture_jumping(liposome_raw,number_of_color, merging_distance_thres, blinking):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    point_mixed = []

    unique_frame = np.unique(liposome_raw[:,2])
    unique_liposome = np.unique(liposome_raw[:,5])
    intensity_list = np.zeros((len(unique_liposome), 2))
    intensity_list[:,0] = unique_liposome
    for i ,j in enumerate(np.unique(liposome_raw[:,5])):
        temp = liposome_raw[liposome_raw[:,5] == j]
        average_intensity = temp[:,6:6+number_of_color].sum(axis = 0).mean()
        intensity_list[i,1] = average_intensity
        

    intensity_list = intensity_list[np.argsort(intensity_list[:,1]),:][::-1]
    intensity_rank = intensity_list[:,0].astype(int)

    for i in unique_frame:
        temp = liposome_raw[liposome_raw[:,2] == i]
        distance_matrix = distance(temp,temp)    
        data = distance_matrix< merging_distance_thres 

        connection = np.tril(data, k = -1)
        con_list = np.argwhere(connection)
        list_ID_changing = con_list.tolist()
        list_ID_changing = ID_change_duplication(list_ID_changing)
        list_ID_changing = ID_change_duplication(list_ID_changing)
        list_ID_changing = ID_change_duplication(list_ID_changing)
        
        lost_index = np.zeros((0))
        for j in list_ID_changing:
            
            
            mean_xy = np.copy(temp[j,0:2].mean(axis=0))    


            intensity_sum = temp[j,6:6+number_of_color].sum(axis = 0)
            intensity = [np.random.uniform(intensity_sum[i]/2, intensity_sum[i]) for i in range(0,number_of_color)] #choosing mixture color between [sum(intensity), mean(intensity)]        
            unique_liposome = temp[j,5]
            index_max = np.argmin([np.argwhere(intensity_rank == k)[0][0] for k in unique_liposome])
            max_liposome = unique_liposome[index_max]
            arg_maxliposome_in_temp = np.argwhere(temp[:,5] == max_liposome)
            arg_maxliposome_in_j = np.argwhere(temp[j,5] == max_liposome)
            lost_index = np.append(lost_index, np.delete(j, arg_maxliposome_in_j)) 
            temp[arg_maxliposome_in_temp,0:2] = mean_xy
            temp[arg_maxliposome_in_temp,6:6+number_of_color] = intensity

        temp = np.delete(temp, lost_index.astype(int), axis = 0)
        point_mixed.append(temp)
    point_mixed = np.vstack(point_mixed)



    max_unique = np.unique(point_mixed[:,5]).max()
    max_unique = max_unique + 1

    counting = 0

    for q in np.unique(point_mixed[:,5]):

        temp = point_mixed[point_mixed[:,5] == q]
        temp[:,3] = np.arange(counting, counting + len(temp))
        temp[:,4] = np.arange(1 + counting,  len(temp) + counting + 1 )
        counting = counting + len(temp)
        temp[0,3] = -1
        time_series = temp[:,2]
        index_change = np.argwhere((time_series[1:] - time_series[:-1])> blinking)[:,0]
        for i in index_change:
            temp[i+1:,5] = max_unique
            temp[i+1,3] = -1
            max_unique = max_unique + 1
        point_mixed[point_mixed[:,5] == q] = temp
    return point_mixed



def point_mixture_exact(liposome_raw, merging_distance_thres, number_of_color):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    point_mixed = []
    liposome = np.copy(liposome_raw)
    unique_frame = np.unique(liposome[:,2])
    
    for i in unique_frame:

        temp = liposome[liposome[:,2] == i]


        distance_matrix = distance(temp,temp)    
        data = distance_matrix< merging_distance_thres
        connection = np.tril(data, k = -1)
        con_list = np.argwhere(connection)
        list_ID_changing = con_list.tolist()
        list_ID_changing = ID_change_duplication(list_ID_changing)
        list_ID_changing = ID_change_duplication(list_ID_changing)
        list_ID_changing = ID_change_duplication(list_ID_changing)       
        merged_index = []
        original_index   = []
        
        for j in list_ID_changing:
            
            mean_xy = np.copy(temp[j,0:2].mean(axis=0))  
            
            intensity_max = temp[j,6:6+number_of_color].max(axis = 0)
            intensity_sum = temp[j,6:6+number_of_color].sum(axis = 0)
            intensity = [np.random.uniform(intensity_max[i], intensity_sum[i]) for i in range(0,number_of_color)] #choosing mixture color in between [max intensity, sum intensity]     
            
            base_liposome = temp[j[0],4]

            temp[j,0:2] = mean_xy
            temp[j,6:6+number_of_color] = intensity
            
            merged_index.append(temp[j[1:],4])
            original_index.append([int(base_liposome)]*len(j[1:]))
    
        merged_index = [a for b in merged_index for a in b]
        original_index =  [a for b in original_index for a in b]
        
        index = np.column_stack((merged_index, original_index))
        

        for k in index:
            merged_liposome = k[0]
            original_liposome = k[1]
            temp[:,4][temp[:,4] == merged_liposome] = original_liposome

            liposome[:,3][liposome[:,3] == merged_liposome] = original_liposome
            unique, index_edge_2 = np.unique(temp[:,3:5], axis = 0, return_index=True) #eliminating dluplicated edges
            temp = temp[index_edge_2,:]
        point_mixed.append(temp)
    point_mixed = np.vstack(point_mixed)

    return point_mixed


def blinking_deletion(liposome, percentage):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)

    liposome_mixture = np.copy(liposome)

    unique_frame = np.unique(liposome_mixture[:,2])
    blink_deleted = []
    for i in unique_frame:

        temp = liposome_mixture[liposome_mixture[:,2] == i]
        len_temp = len(temp)
        integer_delete = int(percentage*len_temp/100)
        remain_delete = (percentage*len_temp)%100
        if remain_delete > np.random.randint(100):
            remain_delete = 1
        else:
            remain_delete = 0
        unfortunate_liposome_index = np.random.choice(np.arange(len_temp), integer_delete + remain_delete , replace = False) 
        unfortunate_liposome = temp[unfortunate_liposome_index]
        temp = np.delete(temp, unfortunate_liposome_index, axis = 0)
                                                   
        blink_deleted.append(temp)
        
        
        from_value = np.copy(unfortunate_liposome[:,3].astype(int))
        to_value   = np.copy(unfortunate_liposome[:,4].astype(int))
        
        from_to = np.arange(liposome_mixture[:,4].max()+2, dtype = int)
        from_to[-1] = -1  
        from_to[to_value] = from_value
        liposome_mixture[:,3] = from_to[liposome_mixture[:,3].astype(int)]

    blink_deleted = np.vstack(blink_deleted)
    
    return blink_deleted




def add_false_positive(liposome_raw, number_of_color, number_of_noise, number_of_weak_FI, traj,merging_distance_thres,combinatorial):
    seed_value = np.random.RandomState().randint(100000000)
    np.random.seed(seed_value)
    liposome = np.copy(liposome_raw)
    liposome[:,6+ number_of_color ] = 0

    noise_per_frame = number_of_noise
    
    xmax_liposome = liposome[:,0].max()
    xmin_liposome = liposome[:,0].min()
    ymax_liposome = liposome[:,1].max()
    ymin_liposome = liposome[:,1].min()

    liposome_noise = []
    
    for i in np.unique(liposome[:,2]):
        if noise_per_frame == 0:
            num_noise =0
        else:
            num_noise = np.random.normal(noise_per_frame,3)
            num_noise = abs(num_noise)
            num_noise = int(num_noise)
        
        index_noise = np.random.choice(liposome_raw.shape[0], num_noise, replace = True)
        sampled_noise = np.zeros((num_noise, liposome.shape[1]))
        sampled_noise[:,2] = i
        sampled_noise[:,5] = -1 
        sampled_noise[:,0] = np.random.uniform(xmin_liposome, xmax_liposome, num_noise )
        sampled_noise[:,1] = np.random.uniform(ymin_liposome, ymax_liposome, num_noise )
        sampled_noise[:,6+number_of_color] = -1
        sampled_noise[:,3] = -1
        sampled_noise[:,6:6+number_of_color] = liposome_raw[index_noise, 6:6+number_of_color] + abs(np.random.normal(0, 0.1, size=(len(index_noise), number_of_color)))
        liposome_noise.append(sampled_noise)

    number_of_blinking_noise = np.random.randint(int(number_of_weak_FI/2),number_of_weak_FI + 1)
    time_min = liposome[:,2].min()
    time_max = liposome[:,2].max()
    
    intensity_sampled_list = intensity_gamma_list(traj, 1000, np.unique(traj[:,7]), 20)
    intensity_list = np.array(intensity_sampled_list)     
    
    index_blinking_noise = np.random.choice(len(intensity_list), number_of_blinking_noise, replace = False)
    
    intensity_list = intensity_list[index_blinking_noise,:]
    
    for i in range(0, number_of_blinking_noise):
        length_false_traj = np.random.randint(30,70)
        displacement = np.random.normal(0,np.random.uniform(0,3),size = (length_false_traj,2))
        traj_xy = np.cumsum(displacement, axis =0)
        
        end_cut = time_max-length_false_traj
        if time_min >= end_cut:
            end_cut = time_min + 1
        starting_time = np.random.randint(time_min, end_cut)
        sampled_noise = np.zeros((length_false_traj, liposome.shape[1]))
        sampled_noise[:,0:2] = traj_xy
        sampled_noise[:,0] = sampled_noise[:,0] + np.random.uniform(xmin_liposome, xmax_liposome, 1 )
        sampled_noise[:,1] = sampled_noise[:,1] + np.random.uniform(ymin_liposome, ymax_liposome, 1 )
        sampled_noise[:,2] = np.arange(starting_time, starting_time + length_false_traj)
        sampled_noise[:,5] = -1 
        sampled_noise[:,6+number_of_color] = -1
        sampled_noise[:,3] = -1
        sampled_noise[:,6:6+number_of_color] = np.random.gamma(intensity_list[i,0], intensity_list[i,1], size=(length_false_traj, number_of_color))

        partition = np.array([comb(number_of_color,i) for i in range(1,number_of_color + 1)])
        partition =partition/sum(partition)
        
        if combinatorial == True:
            color_combination = np.random.choice(number_of_color, 1, p = partition) + 1
        else:
            color_combination = 1
    
        selected_channel = np.random.choice(np.arange(number_of_color),color_combination, replace=False )
        selected_channel = np.setdiff1d(np.arange(number_of_color),selected_channel)
        if len(selected_channel) >0:
            for i in selected_channel:
                sampled_noise[:,6 + i] = abs(np.random.normal(0, 2, length_false_traj))


        
        index_remain = np.random.choice(len(sampled_noise), int(np.random.randint(0,25)/100 * len(sampled_noise)), replace = False)
        sampled_noise = sampled_noise[index_remain, :]
        liposome_noise.append(sampled_noise)
        
    liposome_noise = np.vstack(liposome_noise)
    max_node_num = liposome[:,4].max() + 1
    liposome_noise[:,4] = np.arange(len(liposome_noise[:,4])) + max_node_num

    liposome = np.vstack((liposome, liposome_noise))
    
    liposome = liposome[np.argsort(liposome[:,2])]
    
    return liposome


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

def plot_graph(liposome_mixture):

    merging_point = np.argwhere(liposome_mixture[:,5]==1)[-1][0]
    merging_time = liposome_mixture[merging_point,2]
    data_raw = liposome_mixture[liposome_mixture[:,2] > merging_time-10]
    data_raw = data_raw[data_raw[:,2] <merging_time +10]

    
    nodes = data_raw[:,3:5].astype(int).flatten()
    nodes = np.unique(nodes)
    nodes = nodes[nodes != -1]   
    
    position = []
    for i in nodes:        
        position.append(liposome_mixture[:,0:2][liposome_mixture[:,4] == i][0])
    position = np.vstack(position)

    G = nx.Graph()
    j = 0 

    for i in nodes:
        G.add_node(i, pos = position[j])
        j = j + 1
    
    
    data_raw = data_raw[data_raw[:,9] !=-1]
    edges = data_raw[data_raw[:,0] !=-1][:,3:5]
       
    G.add_edges_from(edges)
    nx.draw(G,  node_size = 70, font_size = 7, with_labels=True, node_color = "green")
