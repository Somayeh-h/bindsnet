import os
import sys 
import cv2
import re
import math 
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
import scipy.io
from pathlib import Path 
import pandas as pd
import seaborn as sn
from sklearn.metrics import auc
import argparse

prefs.codegen.target = "numpy"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     

def get_recognized_number_ranking(assignments, spike_rates, true_label, all_assignments):

    '''
    Given the assignments (calculated from training response), and the number of spikes fired for the img with given label  
    '''

    summed_rates = [0] * num_unique_test_labels
    num_assignments = [0] * num_unique_test_labels

    data = [] 

    for i in range(num_unique_test_labels):
        assigned_neurons_indices = np.where(assignments == i)[0]  
        num_assignments[i] = len(assigned_neurons_indices)

        if num_assignments[i] > 0:            
            test_spikes = spike_rates[assignments == i]

            if weighted_assignments: 
                train_spikes = np.array([all_assignments[x][i] for x in assigned_neurons_indices])
                norm_factor = ( np.sum(train_spikes[test_spikes != 0]) ) / (np.sum(train_spikes)) 

                if upweight_assignments:                         
                    summed_rates[i] = ( np.sum(test_spikes) / num_assignments[i] ) * (1 / norm_factor) if norm_factor != 0 else norm_factor

                if use_weighted_test_spikes:                     
                    all_train_spikes_all = np.array([ [x, all_assignments[x]] for x in range(len(all_assignments)) if i in all_assignments[x]])

                    all_train_spikes = np.array([ x for x in all_train_spikes_all if (x[1].get(i) / sum(list(x[1].values()))) > 0.1 ])    

                    sum_train_spikes = np.array([ sum(list(x[1].values()))  for x in all_train_spikes ])
                    train_spikes = np.array([ x[1].get(i) for x in all_train_spikes ])
                    learnt_neurons = np.array([ x[0] for x in all_train_spikes ])
                    len_learnt_labels = [ len(list(x[1].values()))  for x in all_train_spikes ]
                    test_spikes = np.array([ spike_rates[x] for x in range(len(spike_rates)) if x in learnt_neurons ])  

                    norm_factor = ( np.sum(train_spikes[test_spikes != 0])  /  np.sum(train_spikes) ) if np.any(train_spikes) else 0 

                    if use_first: 
                        test_spikes = np.array([ test_spikes[x] if len_learnt_labels[x] <= 0.02*num_unique_test_labels else test_spikes[x]*(1/len_learnt_labels[x]) for x in range(len(test_spikes))  ]) 

                    if use_second: 
                        test_spikes = np.array([test_spikes[x] * ( train_spikes[x] / (sum_train_spikes[x]) ) for x in range(len(test_spikes)) ]) 

                    if use_third:
                        summed_rates[i] =  ( np.sum(test_spikes) ) * norm_factor if len(learnt_neurons) > 0 else 0  
                    else:
                        summed_rates[i] =  ( np.sum(test_spikes) ) if len(learnt_neurons) > 0 else 0  

                else: 
                    summed_rates[i] = ( np.sum(test_spikes) / num_assignments[i] ) * norm_factor
                
            else: 
                summed_rates[i] = ( np.sum(test_spikes) / num_assignments[i] )

            if summed_rates[i] > 0:
                spiked_rates = spike_rates[assignments == i]
                data.append([i, summed_rates[i], assigned_neurons_indices, spiked_rates])                   

    summed_rates = np.array(summed_rates)

    if np.all(summed_rates == 0):
        sorted_summed_rates = np.arange(-1, num_unique_test_labels-1)
    else: 
        sorted_summed_rates = np.argsort(summed_rates)[::-1]

    # num spikes fired by neurons that were correctly assigned to the label i 
    true_assigned_neurons_spike_rates = spike_rates[assignments == true_label]
    tp_rates = dict(zip( np.where(assignments == true_label)[0], true_assigned_neurons_spike_rates))

    highest_avg_neurons_spike_rates = spike_rates[assignments == sorted_summed_rates[0]]

    # num spikes fired by neurons that weren't assigned to label i 
    fp_spike_rate_indices = [x for x in np.where(assignments != true_label)[0] if spike_rates[x] != 0]
    fp_rates = dict(zip( fp_spike_rate_indices, spike_rates[fp_spike_rate_indices]))

    return sorted_summed_rates, summed_rates, true_assigned_neurons_spike_rates, highest_avg_neurons_spike_rates, tp_rates, fp_rates, data


def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)

    # initialize them as not assigned
    assignments = np.ones(n_e) * -1 
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    

    for j in range(num_unique_test_labels):
        num_inputs = len(np.where(input_nums == j)[0])

        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs

        # weighted assignments gets updated here 
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j

    return assignments


def get_new_assignments_weighted(result_monitor, input_numbers, hard_assignments):
    print(result_monitor.shape)
    print(hard_assignments.shape)

    # initialize them as not assigned
    assignments = [{} for _ in range(n_e)] 
    input_nums = np.asarray(input_numbers)

    for j in range(num_unique_test_labels):
        num_inputs = len(np.where(input_nums == j)[0])
        maximum_rate = [0] * n_e

        if num_inputs > 0:

            # avg num spikes fired by 400 neurons after showing all examples of label j 
            num_spikes = np.sum(result_monitor[input_nums == j], axis = 0)

            rate = num_spikes / num_inputs


        for i in range(n_e):
            if rate[i] > maximum_rate[i]:

                maximum_rate[i] = rate[i]                

                if j in assignments[i]:
                    assignments[i][j] += num_spikes[i]
                else:  
                    assignments[i][j] = num_spikes[i] 

    top_assignments = np.array([-1 if not assignments[i] else max(assignments[i], key=assignments[i].get) for i in range(len(assignments))])

    return top_assignments, assignments


def compute_distance_matrix(dMat, data_path, name):
    
    fig, ax = plt.subplots()

    sn.set_context("paper", font_scale=2)
    
    im = ax.matshow(dMat, cmap=plt.cm.Blues)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "Distance"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel("Output labels")
    plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig(data_path + name)
    plt.savefig(data_path + name + ".pdf")


def compute_distance_matrix_v2(dMat, input_nums_mat, matchIndices, name, data_path=""):

    plt.figure()

    data = {'Reference': input_nums_mat, 
            'Query': matchIndices}
    
    df = pd.DataFrame(data, columns=['Reference', 'Query'])
    confusion_matrix = pd.crosstab(df['Reference'], df['Query'])

    # sn.heatmap(confusion_matrix, annot=False, cbar=True)

    sn.heatmap(dMat, annot=False, cmap='Blues', xticklabels = 10, yticklabels = 10, cbar=True)
    plt.xlabel("Query")
    plt.ylabel("Reference")

    if save:
        plt.savefig(data_path + name + "_v2")
        plt.savefig(data_path + name + "_v2.pdf")


def compute_binary_distance_matrix():

    input_nums = testing_input_numbers
    input_nums = [int(input_nums[i]) for i in range(len(input_nums)) ]
    input_nums_mat = np.zeros( (len(input_nums), num_unique_test_labels) )
    j = 0
    for row in range(input_nums_mat.shape[0]):
        jj = input_nums[j]
        input_nums_mat[row, jj] = 1
        j += 1

    test_res = test_results[0,:]
    test_res = [int(test_results[0,i]) for i in range(len(test_results[0,:])) ]
    test_res_mat = np.zeros( (len(test_res), num_unique_test_labels) )

    i = 0
    for row in range(test_res_mat.shape[0]):
        ii = test_res[i]
        test_res_mat[row, ii] = 1 
        i += 1

    return input_nums_mat, test_res_mat


def processImage(im,reso=None):
    """
    imPath: full image path
    reso: (width,height)
    """
    
    if reso is not None:  
        # im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)      
        im = cv2.resize(im,reso)
    return im


def add_subplot_border(ax, width=1, color=None ):

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, color=color,
        transform=ax.transAxes, zorder=-1, lw=2*width+1, fill=None)

    fig.patches.append(rect)


def get_weights_matrix(weightsPath, name, n_input, n_e):

    XeAe_w = np.load(weightsPath + name)
    value_arr = np.nan * np.ones((n_input, n_e))

    # for each row of XeAe ==> (313600, 3)
    for conn in XeAe_w:

        # get i, j, and value and assign to value array ==> (784, 400)
        src, tgt, value = int(conn[0]), int(conn[1]), conn[2]
        value_arr[src, tgt] = value

    values = np.asarray(value_arr)
    XA_values = np.copy(values)

    return XeAe_w

    
def show_qry_match_gt_imgs_basic(testImgs, trainImgs, orgTestImgs, orgTrainImgs, testing_labels, matchIndices, imgIdx, data_path, saveFig=False):  

    qryImgPN = processImage(testImgs[imgIdx])
    matchImgPN = processImage(trainImgs[int(matchIndices[imgIdx]), :, :])
    orgImgPN = processImage(trainImgs[imgIdx])

    qryImg = processImage(orgTestImgs[imgIdx], reso)
    matchImg = processImage(orgTrainImgs[int(matchIndices[imgIdx]), :, :], reso)
    orgImg = processImage(orgTrainImgs[imgIdx], reso)

    imgIdxLabel = testing_labels[imgIdx]
    matchIdxLabel = testing_labels[int(matchIndices[imgIdx])]

    fig = plt.figure()

    f, axarr = plt.subplots(2, 3)

    axarr[0,0].imshow(qryImgPN, cmap='gray')
    axarr[0,0].set_title("Query PN7")
    axarr[0,0].set_xlabel("Query Index: {}".format(imgIdxLabel))

    axarr[0,1].imshow(matchImgPN, cmap='gray')
    axarr[0,1].set_title("Matched Reference")
    axarr[0,1].set_xlabel("Matched Index: {}".format(matchIdxLabel))

    w = 4 
    if abs(imgIdxLabel - matchIdxLabel) == 0:        
        add_subplot_border(axarr[0,1], w, 'g')
    else: 
        add_subplot_border(axarr[0,1], w, 'r')

    axarr[0,2].imshow(orgImgPN, cmap='gray')
    axarr[0,2].set_title("Ground Truth")
    axarr[0,2].set_xlabel("Ground Truth Index: {}".format(imgIdxLabel))


    axarr[1,0].imshow(qryImg, cmap='gray')
    axarr[1,0].set_title("Query in colour")
    axarr[1,0].set_xlabel("Query Index: {}".format(imgIdxLabel))

    axarr[1,1].imshow(matchImg, cmap='gray')
    axarr[1,1].set_title("Matched Reference")
    axarr[1,1].set_xlabel("Matched Index: {}".format(matchIdxLabel))

    w = 4 
    if abs(imgIdxLabel - matchIdxLabel) == 0:        
        add_subplot_border(axarr[1,1], w, 'g')
    else: 
        add_subplot_border(axarr[1,1], w, 'r')

    axarr[1,2].imshow(orgImg, cmap='gray')
    axarr[1,2].set_title("Ground Truth")
    axarr[1,2].set_xlabel("Ground Truth Index: {}".format(imgIdxLabel))

    plt.tight_layout()

    if saveFig:
        plt.savefig(data_path + "plots/Img{}".format(imgIdxLabel)) 

    plt.close(fig)


def show_qry_match_gt_imgs(testImgs, trainImgs, orgTestImgs, orgTrainImgs, testing_labels, imgIdx, num_unique_labels, data_path, saveFig=False):  

    if imgIdx >= num_true_labels and place_familiarity: 
        matchIdxFall = int(matchIndices[imgIdx]) 
        orgIdxFall = imgIdx
        ORC_offset = int(num_true_labels/2) if unfamiliar_ORC else 0 
    else: 

        offset = num_unique_test_labels if not place_familiarity else num_true_labels
        matchIdxFall = int(matchIndices[imgIdx]) + offset if int(matchIndices[imgIdx]) < 100 else int(matchIndices[imgIdx])
        orgIdxFall = imgIdx + offset
        ORC_offset = 0

    qryImgPN = processImage(testImgs[imgIdx])
    matchImgPN = processImage(trainImgs[int(matchIndices[imgIdx]), :, :])
    orgImgPN = processImage(trainImgs[imgIdx-ORC_offset]) 
    matchImgPNFall = processImage(trainImgs[ matchIdxFall, :, :])
    orgImgPNFall = processImage(trainImgs[orgIdxFall-ORC_offset])

    qryImg = processImage(orgTestImgs[imgIdx], reso)
    matchImg = processImage(orgTrainImgs[int(matchIndices[imgIdx]), :, :], reso)
    orgImg = processImage(orgTrainImgs[imgIdx-ORC_offset], reso) 
    matchImgFall = processImage(orgTrainImgs[matchIdxFall, :, :], reso)
    orgImgFall = processImage(orgTrainImgs[orgIdxFall-ORC_offset], reso) 

    imgIdxLabel = testing_labels[imgIdx]
    matchIdxLabel = testing_labels[int(matchIndices[imgIdx])]

    sn.set_context("paper", font_scale=2 )
    fig = plt.figure(figsize=(22,10))
    fig.suptitle("")
    outer = gridspec.GridSpec(3, 1, wspace=0.001, hspace=0.2)

    innerFirst = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.001, hspace=0.001)
    imgs = np.array([qryImgPN, matchImgPN, orgImgPN, matchImgPNFall, orgImgPNFall])    
    titles = ["Query L{}", "Matched Spr L{}", "Ref Spr L{}", "Matched Fall L{}", "Ref Fall L{}"]
    w = 4 

    for j in range(imgs.shape[0]):

        ax = plt.subplot(innerFirst[j])     
        ax.imshow(imgs[j], cmap='gray')       
        ax.set_xticks([])
        ax.set_yticks([])

        if j % 2 == 0: 
            ax.set_title(titles[j].format(imgIdxLabel), pad=20, fontsize=30, wrap=True)
        else:
            ax.set_title(titles[j].format(matchIdxLabel), pad=20, fontsize=30, wrap=True)

        if j == 1 or j == 3: 
            if abs(imgIdxLabel - matchIdxLabel) == 0:        
                add_subplot_border(ax, w, 'g')
            else: 
                add_subplot_border(ax, w, 'r')     

        ax.set_aspect('equal')
        fig.add_subplot(ax)

    innerMid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.001, hspace=0.001)
    imgs = np.array([qryImg, matchImg, orgImg, matchImgFall, orgImgFall])
    xlabels = ["Query Label {}", "Matched Label {}", "True Label {}", "Matched Label {}", "True Label {}"] 
    
    for j in range(imgs.shape[0]):

        ax = plt.subplot(innerMid[j])     
        ax.imshow(imgs[j], cmap='gray')       
        ax.set_xticks([])
        ax.set_yticks([])

        # if j % 2 == 0: 
        #     ax.set_xlabel(xlabels[j].format(imgIdxLabel), labelpad=10, fontsize=30, wrap=True)
        # else:
        #     ax.set_xlabel(xlabels[j].format(matchIdxLabel), labelpad=10, fontsize=30, wrap=True)

        if j == 1 or j == 3: 
            if abs(imgIdxLabel - matchIdxLabel) == 0:        
                add_subplot_border(ax, w, 'g')
            else: 
                add_subplot_border(ax, w, 'r')
        
        ax.set_aspect('equal')
        fig.add_subplot(ax)

    neurons_i = np.where(assignments == imgIdxLabel)
    
    neuron_i_rep = XA_values[:, neurons_i]
    neuron_i_rep = np.squeeze(neuron_i_rep)

    if len(neuron_i_rep.shape) == 1:
        ncol = 1 
    else: 
        ncol = neuron_i_rep.shape[1]

    if ncol != 0: 
        innerLast = gridspec.GridSpecFromSubplotSpec(1, ncol, subplot_spec=outer[2], wspace=0.001, hspace=0.001)

        for j in range(ncol):
            ax = plt.subplot(innerLast[j])
            
            if len(neuron_i_rep.shape) == 1:
                img = neuron_i_rep.reshape(imWidth, imHeight)
            else:
                img = neuron_i_rep[:, j].reshape(imWidth, imHeight)

            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Neuron {}".format(neurons_i[0][j]), labelpad=20, fontsize=30 )

            if abs(imgIdxLabel - matchIdxLabel) == 0:        
                add_subplot_border(ax, w, 'g')
            else: 
                add_subplot_border(ax, w, 'r')

            # if j == 0:
                # ax.set_title("Learnt feature representations of Label {} by {} neurons".format(imgIdxLabel, ncol), loc='left', pad=10, fontsize=30)

            ax.set_aspect('equal')
            fig.add_subplot(ax)

    if saveFig:
        
        plt.axis('on')

        plt.tight_layout()

        path_to_save = data_path + "plots_v2/"
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save + "Img{}".format(imgIdxLabel)) 
        plt.savefig(path_to_save + "Img{}.pdf".format(imgIdxLabel)) 

    plt.close('all')


def plot_feature_representations(i, assignments, min_diags, min_off_diags, data_path):

    neurons_i = np.where(assignments == i)
    
    neuron_i_rep = XA_values[:, neurons_i]
    neuron_i_rep = np.squeeze(neuron_i_rep)

    fig = plt.figure()

    if len(neuron_i_rep.shape) == 1:
        ff, axarr = plt.subplots(1, len(neuron_i_rep.shape) )
    elif neuron_i_rep.shape[1] == 0:
        return
    else: 
        ff, axarr = plt.subplots(1, neuron_i_rep.shape[1])

    for imgListIdx in range( len(ff.axes) ): 
        
        if len(neuron_i_rep.shape) == 1:
            img = neuron_i_rep.reshape(imWidth, imHeight)
            axarri = axarr
        else: 
            img = neuron_i_rep[:, imgListIdx].reshape(imWidth, imHeight)
            axarri = axarr[imgListIdx]

        axarri.imshow(img, cmap='gray')
        axarri.set_title("N{}".format(neurons_i[0][imgListIdx]))

        w = 4  
        if i in list(min_diags.keys()):        
            add_subplot_border(axarri, w, 'g')

        elif i in list(min_off_diags.keys()): 
            add_subplot_border(axarri, w, 'r')
    
    plt.tight_layout()
    plt.suptitle('Learnt feature representations of Label {} by {} neurons'.format(i, len(ff.axes)  ))  
    plt.savefig(data_path + "figures/Img{}".format(i)) 
    plt.close(fig)


def plot_neurons_labels(min_diags, min_off_diags):

    n = list(range(n_e))
    plt.figure()
    plt.plot(n, assignments, '.b')

    for i in range(num_unique_test_labels):

        if i in list(min_diags.keys()):
            plt.axhline(y=i, linewidth=0.3, color='gray')
        elif i in list(min_off_diags.keys()):
            plt.axhline(y=i, linewidth=0.3, color='red')

    plt.xlabel("Neurons")
    plt.ylabel("Labels (Incorrect testing results are red)")
    plt.title("The neurons and the corresponding learnt labels")
    plt.savefig(data_path + "learning")


def plot_neurons_learnt_labels():
    plt.figure()
    plt.bar(list(range(len(unique_assignments))), unique_counts)
    plt.xlabel("Labels")
    plt.ylabel("Number of learnt neurons")
    plt.title("Number of neurons learning each label")
    plt.savefig(data_path + "Num neurons learning labels")


def get_accuracy(difference, tolerance):

    correct = np.where(difference <= tolerance)[0]
    incorrect = np.where(difference > tolerance)[0]
    sum_accurracy = len(correct)/float(num_testing_imgs) * 100

    return correct, incorrect, sum_accurracy


def get_highest_summed_rates(rates_matrix, type):

    if type == "min": 
        top_col_val_idx = rates_matrix.argmin(axis=0)
        mat_val = (max(map(max, rates_matrix))) 
    elif type == "max":
        top_col_val_idx = rates_matrix.argmax(axis=0)
        mat_val = (min(map(min, rates_matrix))) 

    diags = {}                      # tp
    off_diags = {}                  # fp 
    off_diags_on_diags_vals = {}    # val of fp for true labels 
    off_diag_vals_unfamiliar = []   # fp for unfamiliar imgs 

    all_fp_rates_unfamiliar = []    # all col vals of fp matches for unfamiliar places 
    all_fp_rates_familiar = []      # all col vals of fp matches for familiar places 

    for colIdx in range(rates_matrix.shape[1]):

        top_col_val = rates_matrix[top_col_val_idx[colIdx], colIdx]
        used_indices = np.where(rates_matrix[:, colIdx] != mat_val)[0]

        if top_col_val_idx[colIdx] == colIdx:
            diags[colIdx] = top_col_val   

        elif place_familiarity and colIdx >= num_true_labels:
            off_diag_vals_unfamiliar.append(top_col_val)
            all_fp_rates_unfamiliar.append(rates_matrix[used_indices, colIdx])

        else:
            off_diags[colIdx] = top_col_val
            off_diags_on_diags_vals[colIdx] = rates_matrix[colIdx, colIdx]
            all_fp_rates_familiar.append(rates_matrix[used_indices, colIdx])
    
    # normalise for histogram plotting 
    diag_vals = np.array( list(diags.values()) )
    off_diag_vals = np.array( list(off_diags.values()) )
    off_diag_vals_unfamiliar = np.array(off_diag_vals_unfamiliar)

    all_fp_rates_unfamiliar = np.array(all_fp_rates_unfamiliar)
    all_fp_rates_familiar = np.array(all_fp_rates_familiar)

    return diags, off_diags, off_diags_on_diags_vals, diag_vals, off_diag_vals, off_diag_vals_unfamiliar, all_fp_rates_unfamiliar, all_fp_rates_familiar


def plot_histogram(min_diag_vals, min_off_diag_vals, min_off_diag_vals_unfamiliar, min_all_fp_rates_unfamiliar, min_all_fp_rates_familiar, type):
    '''
    Given min diag vals (tp), and min off diag vals (fp),
    plots a histogram of true matches and false matches 
    '''

    if len(min_all_fp_rates_familiar) > 0: 
        min_all_fp_rates_familiar = np.concatenate(min_all_fp_rates_familiar).ravel()
    num_bins = 10

    bins = np.histogram(np.hstack((min_diag_vals, min_off_diag_vals)), bins=num_bins)[1]

    if place_familiarity: 
        min_all_fp_rates_unfamiliar = np.concatenate(min_all_fp_rates_unfamiliar).ravel()
        bins = np.histogram(np.hstack((min_diag_vals, min_off_diag_vals, min_all_fp_rates_unfamiliar)), bins=num_bins)[1] 

    plt.figure()
    plt.hist(min_diag_vals, density=False, bins=bins, alpha=0.5, histtype='bar', ec='black', color='g', label='tm')
    plt.hist(min_off_diag_vals, density=False, bins=bins, alpha=0.5, histtype='bar', ec='black', color='r', label='fm')
    if place_familiarity:
        plt.hist(min_off_diag_vals_unfamiliar, density=False, bins=bins, alpha=0.2, histtype='bar', ec='black', color='k', label='unseen fm')

    plt.legend(loc='upper right')
    plt.xlabel('Highest spike rates')
    plt.ylabel('Counts')
    plt.title('Spike rates of tm and fm using {} distances for tm'.format(type))
    plt.savefig(data_path + "histogram_{}".format(type))


    if len(min_all_fp_rates_familiar) > 0:
        bins = np.histogram(np.hstack((min_diag_vals, min_all_fp_rates_familiar)), bins=num_bins)[1]
    else: 
        bins = 10 

    plt.figure()
    plt.hist(min_diag_vals, density=True, bins=bins, alpha=0.5, histtype='bar', ec='black', color='g', label='tm')
    plt.hist(min_all_fp_rates_familiar, density=True, bins=bins, alpha=0.5, histtype='bar', ec='black', color='r', label='fm')
    if place_familiarity: 
        bins = np.histogram(np.hstack((min_diag_vals, min_all_fp_rates_familiar, min_all_fp_rates_unfamiliar)), bins=num_bins)[1]
        plt.hist(min_all_fp_rates_unfamiliar, density=True, bins=bins, alpha=0.5, histtype='bar', ec='black', color='k', label='unseen fm')

    plt.legend(loc='upper right')
    plt.xlabel('Spike rates')
    plt.ylabel('Counts')
    plt.title('Spike rates of tm and all fm using {} distances for tm'.format(type))
    plt.savefig(data_path + "histogram_{}_tm_all_fm".format(type))



def plot_num_neurons_learning_each_label(unique_counts, unique_assignments):
    # remove -1 assignments 
    unique_counts_labels = np.delete(unique_counts, 0)
    label_list = list(range(len(unique_assignments)-1)) 

    plt.figure()
    barlist = plt.bar(label_list, unique_counts_labels)

    for barIdx in range(len(label_list)):

        if barIdx in min_diags.keys():
            barlist[barIdx].set_color('g')
        
        elif barIdx in min_off_diags.keys():
            barlist[barIdx].set_color('r')

    plt.xlabel("Labels")
    plt.ylabel("Number of learnt neurons")
    plt.title("Number of neurons learning each label")
    plt.savefig(data_path + "Num neurons learning labels and summed rates")


def plot_labels_and_spike_rates_bargraph(max_diags, max_off_diags, max_off_diags_on_diags_vals):
    plt.figure(figsize=(18,10))
    plt.bar(list(max_diags.keys()), list(max_diags.values()), color='g', label='tp spike rates')
    plt.bar(list(max_off_diags.keys()), list(max_off_diags.values()), color='r', label='fp spike rates')
    plt.bar(list(max_off_diags_on_diags_vals.keys()), list(max_off_diags_on_diags_vals.values()), color='b', label='true spike rates')
    plt.legend()
    plt.xlabel("Labels")
    plt.ylabel("Average spike rate of learnt neurons")
    plt.title("Labels and spike rates")
    plt.savefig(data_path + "tp and fp spike rates")


def plot_labels_and_spike_rates_boxplots(summed_rates, num_testing_imgs, min_diags, min_off_diags):
    fig, ax = plt.subplots(figsize=(18,10))
    summed_rates_nonzeros = [] 

    for colIdx in range(summed_rates.shape[1]):
        nonzero_summed_rates = [i for i in summed_rates[:, colIdx] if i != 0]
        summed_rates_nonzeros.append(nonzero_summed_rates)

    meanlineprop = dict(linestyle='--', linewidth=2.5, color='blue')
    bp = ax.boxplot(summed_rates_nonzeros, 0, 'k.', patch_artist=True, meanprops=meanlineprop, meanline=True, showmeans=True)
    xaxis = list(range(num_testing_imgs))
    ax.set_xticklabels(xaxis)
    ax.set_xlabel("Image labels", fontsize=12)
    ax.set_ylabel("Spike rates of learnt neurons", fontsize=12)
    plt.suptitle("Labels and spike rates", fontsize=16)

    i = 0
    for patch in bp['boxes']:

        if i in min_diags.keys():
            patch.set_facecolor('g')
        elif i in min_off_diags.keys():
            patch.set_facecolor('r')
        i += 1

    plt.savefig(data_path + "tp and fp spike rates boxplots")



def get_rates():

    value_tp, count_tp = np.unique(all_tp_spike_rates[i], return_counts=True)
    value, count = np.unique(all_highest_spike_rates[i], return_counts=True)
    # count_tp = [c-0.5 for c in count_tp]
    # value = [v+0.5 for v in value]

    plt.figure(figsize=(18,10))
    if i == test_results[0,i]:
        plt.bar(count_tp, value_tp, color='g', ec='black', label='tp spike rates', width=1)
    
    else:
        plt.bar(count_tp, value_tp, color='b', ec='black', label='tp spike rates', width=1)
        plt.bar(count, value, color='r', ec='black', label='highest spike rates', width=1)

    plt.legend()
    plt.xlabel("Number of neurons that fired/were assigned label {}".format(i))
    plt.ylabel("Spike rates")
    plt.title("Labels and spike rates")
    plt.xlim(0, 5)

    path_to_save = data_path + "rates/"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save + "tp and highest spike rates {}".format(i))

    plt.close('all')


def plot_label_assignments(j):

    unique_j_assignment_dic = {}

    plt.figure()
    for i, aPath in enumerate(assignment_list):

        if 'assignments' not in aPath:
            continue

        assignments_i = hard_assignments[i]

        unique_i_assignments, unique_i_counts = np.unique(assignments_i, return_counts=True)

        unique_j_assignment_indices = np.where(j < len(unique_i_assignments) and assignments_i == unique_i_assignments[j])

        unique_j_assignment_dic[ i ] = unique_j_assignment_indices[0]

        if i == len(assignment_list)-1: 
            plt.plot( unique_j_assignment_indices[0], np.zeros_like(unique_j_assignment_indices[0]) + i, '.r')
        else:
            plt.plot( unique_j_assignment_indices[0], np.zeros_like(unique_j_assignment_indices[0]) + i, '.b')

    if -1 in unique_assignments: 
        unique_assignment_indices = np.where(assignments == unique_assignments[j+1])
    else:
        unique_assignment_indices = np.where(assignments == unique_assignments[j])

    plt.plot(unique_assignment_indices[0], np.zeros_like(unique_assignment_indices[0]) + i+1, '*g')
    plt.title("Neuron indices that learnt label {} over assignment updates".format(j))
    plt.xlabel("Neurons")
    plt.ylabel("Assignment update interval")
    plt.xlim((0, n_e))

    path_to_save = data_path + "labels/"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save + "label{}".format(j))

    plt.close('all')


def plot_neuron_assignments(j):

    plt.figure(figsize=(18,10))
    for i, aPath in enumerate(assignment_list):

        if 'assignments' not in aPath:
            continue

        assignments_i = hard_assignments[i]

        # label learnt by neuron j 
        neuron_j_label = assignments_i[j]

        plt.plot(neuron_j_label, np.zeros_like(neuron_j_label) + i, '.b')
        plt.axvline(x=neuron_j_label, linewidth=0.3, color='gray')
    
    neuron_label = assignments[j]

    plt.plot(neuron_label, np.zeros_like(neuron_label) + i+1, '*g')
    plt.title("Labels learnt by neuron {} over assignment updates".format(j))
    plt.xlabel("Labels")
    plt.ylabel("Assignment update interval")
    plt.xlim((0, num_iter))

    path_to_save = data_path + "neurons/"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save + "neuron{}".format(j))

    plt.close('all')


def plot_highest_rates():
    sorted_diag_vals = -np.sort(-max_diag_vals)
    sorted_off_diag_vals = -np.sort(-max_off_diag_vals)
    sorted_off_diag_vals_unfamiliar = -np.sort(-max_off_diag_vals_unfamiliar)

    plt.figure()
    plt.plot(list(range(len(sorted_diag_vals))), sorted_diag_vals, label='tp rates')
    plt.plot(list(range(len(sorted_off_diag_vals))), sorted_off_diag_vals, label='fp rates')

    if place_familiarity: 
        plt.plot(list(range(len(sorted_off_diag_vals_unfamiliar))), sorted_off_diag_vals_unfamiliar, label='highest fp rates unfamiliar')
    plt.xlabel('Number of images')
    plt.ylabel('Highest spike rates')
    plt.title('Highest tp and fp spike rates')
    plt.legend()
    plt.savefig(data_path + "highest rates")


def plot_mean_spike_rates():

    mean_all_tp_spike_rates = np.zeros((len(all_tp_spike_rates),))
    mean_all_highest_spike_rates = np.zeros((len(all_highest_spike_rates),))
    
    for row in range(num_testing_imgs):

        mean_i = np.mean(all_tp_spike_rates[row])
        mean_all_tp_spike_rates[row] = mean_i

        mean_ii = np.mean(all_highest_spike_rates[row])
        mean_all_highest_spike_rates[row] = mean_ii

    mean_all_tp_spike_rates_sorted = -np.sort(-mean_all_tp_spike_rates)
    mean_all_tp_spike_rates_indices = np.argsort(mean_all_tp_spike_rates)[::-1]

    mean_all_highest_spike_rates_sorted = [mean_all_highest_spike_rates[mean_all_tp_spike_rates_indices[x]] for x in range(len(mean_all_highest_spike_rates))]

    plt.figure()
    plt.plot(range(num_testing_imgs), mean_all_highest_spike_rates_sorted, 'r', label='fp rates')
    plt.plot(range(num_testing_imgs), mean_all_tp_spike_rates_sorted, 'g', label='tp rates')
    plt.xlabel("Number of testing images")
    plt.ylabel("Mean spike rates")
    plt.title("Average spike rates of test images sorted in descending order")
    plt.legend()
    plt.savefig(data_path + "mean spike rates")


def plot_tp_spike_rates():

    ig, ax = plt.subplots(figsize=(18,10))
    meanlineprop = dict(linestyle='--', linewidth=2.5, color='blue')
    bp = ax.boxplot(all_tp_spike_rates, 0, 'k.', patch_artist=True, meanprops=meanlineprop, meanline=True, showmeans=True)
    xaxis = list(range(num_testing_imgs))
    ax.set_xticklabels(xaxis)
    ax.set_xlabel("Image labels", fontsize=12)
    ax.set_ylabel("Spike rates of neurons", fontsize=12)
    plt.suptitle("Labels and spike rates", fontsize=16)
    for patch in bp['boxes']:
        patch.set_facecolor('0.9')
    plt.savefig(data_path + "tp spike rates boxplots")

    fig, ax = plt.subplots(figsize=(18,10))
    meanlineprop = dict(linestyle='--', linewidth=2.5, color='blue')
    bp = ax.boxplot(all_highest_spike_rates, 0, 'k.', patch_artist=True, meanprops=meanlineprop, meanline=True, showmeans=True)
    xaxis = list(range(num_testing_imgs))
    ax.set_xticklabels(xaxis)
    ax.set_xlabel("Image labels", fontsize=12)
    ax.set_ylabel("Spike rates of neurons", fontsize=12)
    plt.suptitle("Labels and spike rates", fontsize=16)
    i = 0
    for patch in bp['boxes']:

        if not all(all_highest_spike_rates[i] == all_tp_spike_rates[i]):
            patch.set_facecolor('r')
        else:
            patch.set_facecolor('0.9')
        i += 1
    plt.savefig(data_path + "tp spike rates boxplots fps")


def plot_rates_detailed(i):

    data = all_data[i]
    fig, ax = plt.subplots(2, 1, figsize=(15,8), dpi=300)

    if data == []:
        plt.savefig(data_path + "rates_v3/Image_label_{}".format(i))
        return 
    
    summed_rates_i = [data[row][1] for row in range(len(data))] 
    vals, counts = np.unique(np.array(summed_rates_i), return_counts=True)

    max_spike_rate_idx = np.argmax(vals)
    max_spike_rate_counts = counts[int(max_spike_rate_idx)]

    for row in range(len(data)):

        idx = np.where(vals == data[row][1])[0][0]
        is_tp = abs(data[row][1] - vals[int(max_spike_rate_idx)]) < 0.1
        c = 'g' if is_tp else 'b'

        num_learnt_neurons = len(data[row][2])
        num_spiked_neurons = len( np.where(np.array(data[row][3]) > 0)[0] )
        y_data = num_spiked_neurons / num_learnt_neurons

        if data[row][0] == i: 
            rects = ax[0].bar(data[row][1], [max_ylim], color=c, ec='black', alpha=0.5, label='tm avg rate', width=1)

            ax[1].plot(data[row][1], y_data, color=c, marker='*')
            ax[1].text(data[row][1], 1.02*y_data, 'L{}'.format(data[row][0]), ha='center', va='bottom')

        else: 
            rects = ax[0].bar(data[row][1], [max_ylim], color='b', ec='black', alpha=0.1, label='fm avg rate', width=1)

            ax[1].plot(data[row][1], y_data, color='b', marker='*')
            ax[1].text(data[row][1], 1.02*y_data, 'L{}'.format(data[row][0]), ha='center', va='bottom')

        
        for rect in rects:
            height = rect.get_height()
            x = rect.get_x() + rect.get_width()/2.
            ax[0].text(x, 1.02*max_ylim, '{:.2f}'.format(rect.get_x() + rect.get_width()/2.), ha='center', va='bottom')
            ax[0].axvline(x=x, linewidth=0.3, color='gray', linestyle=':')
        

        for neuron_idx in range(len(data[row][2])):
            xx = data[row][1] - 0.5 + (data[row][2][neuron_idx]/n_e)
            yy = data[row][3][neuron_idx]

            if data[row][0] == i:
                if yy == 0:
                    ax[0].plot(xx, yy+0.1, '.r')
                else: 
                    ax[0].bar(xx, yy, color=c, label='tm', width=(5/n_e) )
            else:
                if yy != 0:
                    ax[0].bar(xx, yy, color='b', alpha=1, label='fm', width=(1/n_e) )

    ax[0].set_xlim([-0.5, max_summed_rates])    
    ax[0].set_xticks(np.arange(-0.5, max_summed_rates, step=0.5))
    ax[0].set_xlabel("Average spike rates")

    ax[0].set_ylim([0, max_ylim+0.2])
    ax[0].set_yticks(np.arange(0, max_ylim+0.2, step=1))
    ax[0].set_ylabel('Number of spikes fired')
    ax[0].legend()
    
    ax[1].set_xlim([-0.5, max_summed_rates])    
    ax[1].set_xticks(np.arange(-0.5, max_summed_rates, step=0.5))
    ax[1].set_xlabel("Average spike rates")

    y_lim = 1 
    ax[1].set_ylim([0, y_lim + 0.02])
    ax[1].set_yticks(np.arange(0, y_lim + 0.02, step=0.1))
    ax[1].set_ylabel('Percentage of learnt neurons that fired')

    ax[0].set_title("Average spike rates for test image with label {}".format(i), y=1.1)

    path_to_save = data_path + "rates_v2/"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save + "Image_label_{}".format(i))
    
    plt.close('all')


def getAUCPR(precision, recall):
    AUC_PRs = auc(recall, precision)
    AUC_PR_trap = np.trapz(precision, recall)
    # AUC_PR_simp = simpson(precision, recall)

    return AUC_PRs, AUC_PR_trap


def getRat99P(precision, recall):

    if not np.any(precision > 0.99): 
        return 0
    Rat99P = np.max(recall[precision > 0.99])

    return Rat99P


def plot_precision_recall(dMat, num_true_labels, data_path, fig_name='', label='', show_points=False, fig_num=12):

    all_labels = np.arange(len(dMat))
    new_label = len(dMat)-num_true_labels
    true_labels = np.arange(0, num_true_labels) if len(dMat)==num_true_labels else np.concatenate( (np.arange(0,num_true_labels), np.zeros(new_label,)) )

    gt = np.abs(all_labels - np.argmin(dMat, axis=0)) 
    
    # if place_familiarity:
    #     true_labels = np.abs(all_labels[0:num_true_labels] - np.argmin(dMat[0:num_true_labels, 0:num_true_labels], axis=0)) 
    #     new_labels = -all_labels[num_true_labels:-1]
    #     gt = np.concatenate((true_labels, new_labels))
    
    # get the indices and values of min elements in each row of dist mat 
    mInds = np.argmin(dMat, axis=0)
    mDists = np.min(dMat, axis=0)

    # define thresholds to sweep over based on range of dist mat 
    min_val = np.min(dMat)
    max_val = np.max(dMat)

    thresholds = np.linspace(min_val, max_val, 100)
    precision = []
    recall = []
    f1_scores = [] 

    print("Min value: {} Max value: {}\n".format(min_val, max_val))
    print("Threshold shape: ", len(thresholds))

    for threshold in thresholds:

        # get boolean of all indices of dists whose value is <= threshold 
        matchFlags = mDists <= threshold

        # set all unmatched items to -1 in a fresh copy of mInds 
        mInds_filtered = np.copy(mInds)
        mInds_filtered[~matchFlags] = -1 

        # get positives: matched mInds whose distance <= threshold 
        positives = np.argwhere(mInds_filtered!=-1)[:,0]
        tps = np.sum( gt[positives] == 0 )
        fps = len(positives) - tps 

        if tps == 0:
            precision.append(0)
            recall.append(0)

        # get negatives: matched mInds whose distance > threshold 
        negatives = np.argwhere(mInds_filtered==-1)[:,0]
        tns = np.sum( gt[negatives] < 0 )
        fns = len(negatives) - tns 

        assert(tps+tns+fps+fns==len(gt))

        precision_i = tps / float(tps+fps)
        recall_i = tps / float(tps+fns)
        f1_score = (2*precision_i*recall_i) / (precision_i+recall_i)

        precision.append( precision_i )
        recall.append(recall_i)
        f1_scores.append(f1_score)
        print( 'tps: {}, fps:{}, fns:{}, tns:{}, tot:{}, T:{:.2f}, P:{:.2f}, R:{:.2f}'.format(tps, fps, fns, tns, tps+fps+fns+tns, threshold, precision_i, recall_i) ) 
    
    precision = np.array(precision)
    recall = np.array(recall)
    f1_scores = np.array(f1_scores)

    AUC_PRs, AUC_PR_trap = getAUCPR(precision, recall)
    print("\nArea under precision recall curve: \nsklearn AUC: {}\nnumpy trapz: {}\n".format(AUC_PRs, AUC_PR_trap))

    Rat99P = getRat99P(precision, recall)
    print("\nR @ 99% P is: {}\n".format(Rat99P))

    Pat100R = f1_scores[-1]
    print("\nP a@ 100% R (last item of f1-score) is: {}\n".format(Pat100R))

    print("\n\nf1 scores with wa: {}\n\n".format(f1_scores))

    sn.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})

    plt.figure(fig_num)
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # plt.title("Precision and Recall")  

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_path + "{}.png".format(fig_name))
    plt.savefig(data_path + "{}.pdf".format(fig_name))

    if show_points:
        fig_num += 1
        plt.figure(fig_num)
        plt.plot(recall, precision, label=label)
        plt.plot(recall, precision, '*r')

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision and Recall")  

        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(data_path + "{}_v2.png".format(fig_name))

    return Rat99P, AUC_PRs


def plot_features(label_i, x_values):

    # get all rows of the training res mon for this label 
    train_spikes = training_result_monitor[training_input_numbers[0:training_result_monitor.shape[0]] == label_i, :]
    mean_train_spikes = np.mean(train_spikes, axis=0)

    # get the row of the testing res mon for this label 
    test_spikes = testing_result_monitor[testing_input_numbers == label_i, :][0]

    fig, ax = plt.subplots(2, 1, figsize=(15,8), dpi=300)

    for xxx in x_values: 
        if mean_train_spikes[xxx] == 0:
            continue

        ax[0].bar(x_values[xxx], mean_train_spikes[xxx], color='b', alpha=1, label='train spikes', width=0.5)

    for xxx in x_values: 
        if test_spikes[xxx] == 0:
            continue

        ax[1].bar(x_values[xxx], test_spikes[xxx], color='b', alpha=1, label='test spikes', width=0.5)

    ax[0].set_xlim([0, 400])    
    ax[0].set_xticks(np.arange(0, 400, step=20))
    ax[0].set_xlabel("Number of spikes")

    ax[1].set_xlim([0, 400])    
    ax[1].set_xticks(np.arange(0, 400, step=20))
    ax[1].set_xlabel("Number of spikes")

    path_to_save = data_path + "features/"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_save + "Image_label_{}".format(label_i))

    plt.close('all')


def save_npy2mat(array, workspaceName, fileName):
    scipy.io.savemat('{}.mat'.format(fileName), {workspaceName: array})



                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=8, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=0, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default='NRD', help='Folder name of dataset to be used.')
    parser.add_argument('--num_train_imgs', type=int, default=200, help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=100, help='Number of entire testing images.')
    parser.add_argument('--first_epoch', type=int, default=12000, help='For use of neuronal assignments, the first training iteration number in saved items.')
    parser.add_argument('--last_epoch', type=int, default=12001, help='For use of neuronal assignments, the last training iteration number in saved items.')
    parser.add_argument('--update_interval', type=int, default=600, help='The number of iterations to save at one time in output matrix.')
    parser.add_argument('--epochs', type=int, default=60, help='Simulation time step for every simulated object unless specified locally within object.')
    parser.add_argument('--use_all', type=int, default=4, help='Simulation time step for every simulated object unless specified locally within object.')
    parser.add_argument('--n_e', type=int, default=400, help='Number of excitatory output neurons. The number of inhibitory neurons are the same.')
    parser.add_argument('--plot', dest="plot", action="store_true", help='Boolean indicator for plotting per-label figures.')
    parser.set_defaults(plot=True)

    args = parser.parse_args()

    MNIST_data_path = './data/mnist/'
    data_path = './outputs/outputs_ne{}_L{}/'.format(args.n_e, args.num_test_imgs) # './outputs/outputs210812/Im28x28L100T100MS1TS60TS1-TNRD-UI600-S8-O300/outputs/'

    assignments_path = data_path 

    path_id = 'L{}_S{}_O{}'.format(args.num_test_imgs, args.skip, args.offset_after_skip)
    folder_id = args.folder_id 

    main_folder_path = data_path 
    Path(main_folder_path).mkdir(parents=True, exist_ok=True)

    weightsPath = './weights/weights_ne{}_L{}/'.format(args.n_e, args.num_test_imgs)
    name = "XeAe.npy"

    init_val_w = 6000   

    num_training_imgs = args.num_train_imgs 
    num_testing_imgs = args.num_test_imgs    
    num_validtn_imgs = 50   

    first_epoch = args.first_epoch          # 3600 9000
    last_epoch = args.last_epoch            # 6001 11401
    update_interval = args.update_interval  

    num_training_sweeps = args.epochs 
    num_testing_sweeps = 1 
    num_validtn_sweeps = 1 

    num_multi_imgs = 1
    num_multi_imgs_test = 1
    num_multi_imgs_val = 1 

    imWidth = 28   
    imHeight = 28   

    n_e = args.n_e
    n_input = imWidth * imHeight

    shuffled = False 
    val_mode = False
    place_familiarity = False
    unfamiliar_ORC = False
    num_true_labels = num_testing_imgs if not place_familiarity else int(num_training_imgs/2)

    plot_ref_qry = args.plot 
    do_extended_plotting = args.plot 
    
    upweight_assignments = False
    scale_neurons = False
    scale_neurons_v2 = False
    use_one_label_neurons = False  
    spiked_frac = 0.1

    weighted_assignments = True 
    use_weighted_test_spikes = True 
    multiple_UI_assignments = True 

    use_standard = args.use_all == 0  
    use_first = args.use_all == 1
    use_second = args.use_all == 2
    use_third = args.use_all == 3
    use_all = args.use_all == 4 

    if use_standard: 
        data_path += "standard/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        tag = "_0"

        weighted_assignments = False 
        use_weighted_test_spikes = False 
        multiple_UI_assignments = False 

    if use_first:
        data_path += "use_first/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        tag = "_1"
    
    if use_second:
        data_path += "use_second/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        tag = "_2"
    
    if use_third:
        data_path += "use_third/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        tag = "_3"

    if use_all:
        data_path += "wa_use_all/"
        Path(data_path).mkdir(parents=True, exist_ok=True)
        tag = "_4"

        use_first = True
        use_second = True 
        use_third = True 
    

    f = open(data_path + "evaluation.out", 'w')
    sys.stdout = f 

    print("\nArgument values:\n{}".format(args))

    epochs = np.arange(first_epoch, last_epoch, update_interval) # 9000, 11401

    print('load results')
    ending = '.npy'

    training_ending = str(init_val_w) if val_mode else str( int(num_training_imgs * num_training_sweeps * num_multi_imgs)) 
    testing_ending = str(init_val_w) if val_mode else str(num_testing_imgs * num_testing_sweeps * num_multi_imgs_test)  
    validation_ending = str( num_validtn_imgs * num_validtn_sweeps * num_multi_imgs_val )

    training_result_monitor = np.load(main_folder_path + 'resultPopVecs' + training_ending + ending)

    training_result_monitor = np.array([ training_result_monitor[x].sum(axis=0) for x in range(training_result_monitor.shape[0]) ])
    training_ending = str(3000) if num_training_sweeps == 24 else training_ending

    if os.path.isfile(main_folder_path + 'inputNumbers' + training_ending + ending): 
        training_input_numbers = np.load(main_folder_path + 'inputNumbers' + training_ending + ending)
    else:
        training_input_numbers = np.array( 2 * num_training_sweeps * num_multi_imgs * list(range(num_testing_imgs)) ) 

    if multiple_UI_assignments:
        for x in epochs:
            training_result_monitors_i = np.load(assignments_path + 'resultPopVecs' + str(int(x)) + ending)
            training_result_monitors_i = np.array([ training_result_monitors_i[x].sum(axis=0) for x in range(training_result_monitors_i.shape[0]) ])

            if x == epochs[0]: 
                training_result_monitors = training_result_monitors_i
            else:
                training_result_monitors = np.append(training_result_monitors, training_result_monitors_i, axis=0)
    else:
        epochs = [1]

    if val_mode: 
        testing_result_monitor = np.load(main_folder_path + 'resultPopVecs' + validation_ending + ending)
        testing_input_numbers = np.load(main_folder_path + 'inputNumbers' + validation_ending + ending)

    else: 
        testing_result_monitor = np.load(main_folder_path + 'resultPopVecs' + testing_ending + ending)
        testing_result_monitor = np.array([ testing_result_monitor[x].sum(axis=0) for x in range(testing_result_monitor.shape[0]) ])
        testing_input_numbers = np.load(main_folder_path + 'inputNumbers' + testing_ending + ending)

    unique_test_labels = np.unique(testing_input_numbers)
    num_unique_test_labels = len(unique_test_labels)


    # load images 
    if do_extended_plotting or plot_ref_qry: 
        testImgs = np.load(main_folder_path + "test_frames" + ending)
        trainImgs = np.load(main_folder_path + "train_frames" + ending)
        orgTestImgs = np.load(main_folder_path + "org_test_frames" + ending)
        orgTrainImgs = np.load(main_folder_path + "org_train_frames" + ending)

    if not val_mode and (do_extended_plotting or plot_ref_qry): 
        XA_values = get_weights_matrix(weightsPath, name, n_input, n_e)

    sum_accurracy = 0  
    test_results = np.zeros((num_unique_test_labels, num_testing_imgs))
    summed_rates = np.zeros((num_unique_test_labels, num_testing_imgs))
    rates_matrix = np.zeros((num_unique_test_labels, num_testing_imgs))
    
    if place_familiarity:
        summed_rates_PF = np.zeros((num_unique_test_labels, num_testing_imgs))
        sorted_summed_rates_PF = np.zeros((num_unique_test_labels, num_testing_imgs))

    print("train res mon: {}, train input nums: {}".format ( training_result_monitor.shape, training_input_numbers.shape) ) 
    print("test res mon: {}, test input nums: {}".format ( testing_result_monitor.shape, testing_input_numbers.shape) ) 

    print('get assignments')
    assignments = get_new_assignments(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]]) 

    all_assignments = [{} for _ in range(n_e)]

    if weighted_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]], assignments) 
    
    if weighted_assignments and multiple_UI_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitors, training_input_numbers[0:len(epochs)*training_result_monitor.shape[0]], assignments) 

    unique_assignments, unique_counts = np.unique(assignments, return_counts=True)

    if shuffled: 
        testing_result_monitor = np.array([x for _,x in sorted(zip(testing_input_numbers, testing_result_monitor))])
        testing_input_numbers = sorted(testing_input_numbers)

    print("Neuron Assignments ( shape = {} ): \n{}".format( assignments.shape, assignments) )
    print("Unique labels learnt ( count: {} ): \n{}".format( len(unique_assignments), unique_assignments ) ) 



    all_tp_spike_rates = [] 
    all_highest_spike_rates = [] 

    all_tp_rates = []
    all_fp_rates = [] 
    all_data = [] 

    norm_summed_rates = np.zeros((num_unique_test_labels, num_testing_imgs)) 
    P_i = np.zeros((num_unique_test_labels, num_testing_imgs))    

    for i in range(num_testing_imgs):
        test_results[:,i], summed_rates[:,i], tp_spike_rates, highest_spike_rates, tp_rates, fp_rates, data = get_recognized_number_ranking(assignments, testing_result_monitor[i,:], i, all_assignments)

        norm_summed_rates[:,i] = (summed_rates[:,i] - np.min(summed_rates[:,i]) ) / ( np.max(summed_rates[:,i]) - np.min(summed_rates[:,i]) )

        P_i[:, i] = norm_summed_rates[:,i] / np.sum(norm_summed_rates[:,i])

        if place_familiarity:
            summed_rates_PF[:,i] = summed_rates[:,i] > 0.525

            if np.all(summed_rates == 0):
                sorted_summed_rates_PF[:,i] = np.arange(-1, num_unique_test_labels-1)
            else:
                sorted_summed_rates_PF[:,i] = np.argsort(summed_rates_PF[:,i])[::-1]

        all_tp_spike_rates.append(tp_spike_rates)
        all_highest_spike_rates.append(highest_spike_rates)

        all_tp_rates.append(tp_rates)    
        all_fp_rates.append(fp_rates)

        all_data.append(data)

    matchIndices = test_results[0,:]

    mean_P_i = np.mean(P_i)
    P_i_tps = [P_i[i,i] for i in range(P_i.shape[0]) if test_results[0, :][i] == i ] 
    mean_P_i_tps = np.mean(P_i_tps)

    print("\nMean P of true label: {}\nAll P of true labels for testing imgs:\n{}\nMean: {}\nTP P of true labels for testing imgs:\n{}".format(mean_P_i, P_i, mean_P_i_tps, P_i_tps))

    max_summed_rates = math.ceil(max(np.amax(np.array(summed_rates), axis=0)))
    max_ylim = 10

    plot_mean_spike_rates()
    plot_tp_spike_rates()

    difference = test_results[0,:] - testing_input_numbers[0:test_results.shape[0]]
    correct = len(np.where(difference == 0)[0])
    performance = correct / float(len(testing_input_numbers)) * 100

    ## stop here for validation 
    if val_mode:              
        np.save(data_path + "performance" + "_" + str(init_val_w), performance)
        sys.exit() 

    print( "Differences: \n{}\nSummed rates (shape = {} ): \n{}".format(difference, summed_rates.shape, summed_rates) )

    # accuracy with tolerance 0
    tolerance = 0
    difference = abs(difference)
    correct, incorrect, sum_accurracy = get_accuracy(difference, tolerance)
    print( "\nSum response - accuracy: {}, num correct: {}, num incorrect: {}".format(np.mean(sum_accurracy), len(correct), len(incorrect)) )
    print("Correctly predicted labels: \n{}\nIncorrectly predicted labels: \n{}\n".format(correct, incorrect))

    if place_familiarity:
        difference_PF = sorted_summed_rates_PF[0,:] - testing_input_numbers[0:sorted_summed_rates_PF.shape[0]]

        correct_PF, incorrect_PF, sum_accurracy_PF = get_accuracy(difference_PF, tolerance)
        print( "\nSum response - accuracy: {}, num correct: {}, num incorrect: {}".format(np.mean(sum_accurracy_PF), len(correct_PF), len(incorrect_PF)) )
        print("Correctly predicted labels: \n", correct_PF)

    # accuracy with tolerance 5
    tolerance = 5
    difference = abs(difference)
    correct, incorrect, sum_accurracy = get_accuracy(difference, tolerance)
    print( "\nSum response - accuracy with tol 5: {}, num correct: {}, num incorrect: {}".format(np.mean(sum_accurracy), len(correct), len(incorrect)) )
    print("Correctly predicted labels: \n", correct)

    # Binary distance matrix 
    input_nums_mat, test_res_mat = compute_binary_distance_matrix() 
    dMat = cdist(input_nums_mat, test_res_mat, 'euclidean')
    compute_distance_matrix(dMat, data_path, "distMatrix")

    # Distance matrix where 0 represents furthest 
    compute_distance_matrix(summed_rates, data_path, "distMatrix2")

    # Distance matrix where 0 represents closest 
    max_rate = (max(map(max, summed_rates)))
    max_norm = (max(map(max, norm_summed_rates)))
    max_P_i = (max(map(max, P_i)))

    rates_matrix_norm = np.zeros((num_unique_test_labels, num_testing_imgs))
    rates_matrix_P_i = np.zeros((num_unique_test_labels, num_testing_imgs))

    plot_name = "DM_{}_{}".format(folder_id, path_id)
    for i in range(num_testing_imgs):
        rates_matrix[:, i] = [max_rate - l for l in summed_rates[:, i] ]
        rates_matrix_norm[:, i] = [max_norm - l for l in norm_summed_rates[:, i] ]
        rates_matrix_P_i[:, i] = [max_P_i - l for l in P_i[:, i] ]

    compute_distance_matrix(rates_matrix, data_path, plot_name)
    # compute_distance_matrix(rates_matrix_norm, data_path, plot_name + "_norm")
    compute_distance_matrix(rates_matrix_P_i, data_path, plot_name + "_Pi")

    compute_distance_matrix_v2(rates_matrix, testing_input_numbers[0:test_results.shape[0]], matchIndices, plot_name, data_path)

    if place_familiarity:
        # Distance matrix where 0 represents furthest 
        compute_distance_matrix(1-summed_rates_PF, data_path, plot_name + "_T1")

    fig_num = 12 
    show_dotted = True
    fig_name = "PR_{}_{}".format(folder_id, path_id)

    Rat99P, AUC_PRs = plot_precision_recall(rates_matrix, num_true_labels, data_path, fig_name, path_id, show_dotted, fig_num)
    fig_num += 1 
    Rat99P_Pi, AUC_PRs_Pi = plot_precision_recall(rates_matrix_P_i, num_true_labels, data_path, fig_name + "_Pi", path_id, show_dotted, fig_num)
    
    Rat99P_name = "Rat99P_{}".format(fig_name) if not use_weighted_test_spikes else "Rat99P_{}_wa".format(fig_name)
    Rat99P_Pi_name = "Rat99P_Pi_{}".format(fig_name) if not use_weighted_test_spikes else "Rat99P_Pi_{}_wa".format(fig_name)

    AUC_PRs_name = "AUC_PRs_{}".format(fig_name) if not use_weighted_test_spikes else "AUC_PRs_{}_wa".format(fig_name)
    AUC_PRs_Pi_name = "AUC_PRs_Pi_{}".format(fig_name) if not use_weighted_test_spikes else "AUC_PRs_Pi_{}_wa".format(fig_name)

    dMat_name = "dMat_{}".format(path_id) if not use_weighted_test_spikes else "dMat_{}_wa".format(path_id)
    summed_rates_name = "summed_rates_{}".format(path_id) if not use_weighted_test_spikes else "summed_rates_{}_wa".format(path_id)
    rates_matrix_name = "rates_matrix_{}".format(path_id) if not use_weighted_test_spikes else "rates_matrix_{}_wa".format(path_id)
    rates_matrix_Pi_name = "rates_matrix_Pi_{}".format(path_id) if not use_weighted_test_spikes else "rates_matrix_Pi_{}_wa".format(path_id)
    matchIndices_name = "matchIndices_{}".format(path_id) if not use_weighted_test_spikes else "matchIndices_{}_wa".format(path_id)

    np.save(data_path + Rat99P_name + tag, Rat99P)
    np.save(data_path + Rat99P_Pi_name + tag, Rat99P_Pi)

    np.save(data_path + AUC_PRs_name + tag, AUC_PRs)
    np.save(data_path + AUC_PRs_Pi_name + tag, AUC_PRs_Pi)
    
    np.save(data_path + dMat_name + tag, dMat)
    np.save(data_path + summed_rates_name + tag, summed_rates)
    np.save(data_path + rates_matrix_name + tag, rates_matrix)
    np.save(data_path + rates_matrix_Pi_name + tag, rates_matrix_P_i)
    np.save(data_path + matchIndices_name + tag, matchIndices)

    print("Testing result: \n", test_results[0,:])
    print("Testing input numbers: \n", testing_input_numbers)
    print("Sum response - accuracy --> mean: {} --> standard deviation: {}\n".format(np.mean(sum_accurracy), np.std(sum_accurracy) ))

    print("Rates matrix: \n", rates_matrix)
    # based on distance matrix where 0 represents furthest
    max_diags, max_off_diags, max_off_diags_on_diags_vals, max_diag_vals, max_off_diag_vals, max_off_diag_vals_unfamiliar, max_all_fp_rates_unfamiliar, max_all_fp_rates_familiar = get_highest_summed_rates(summed_rates, "max")

    # based on distance matrix where 0 represents closest  
    min_diags, min_off_diags, min_off_diags_on_diags_vals, min_diag_vals, min_off_diag_vals, min_off_diag_vals_unfamiliar, min_all_fp_rates_unfamiliar, min_all_fp_rates_familiar = get_highest_summed_rates(rates_matrix, "min")

    plot_highest_rates()

    plot_histogram(min_diag_vals, min_off_diag_vals, min_off_diag_vals_unfamiliar, min_all_fp_rates_unfamiliar, min_all_fp_rates_familiar, "min")

    plot_histogram(max_diag_vals, max_off_diag_vals, max_off_diag_vals_unfamiliar, max_all_fp_rates_unfamiliar, max_all_fp_rates_familiar, "max")

    plot_neurons_labels(min_diags, min_off_diags)

    plot_neurons_learnt_labels()

    plot_num_neurons_learning_each_label(unique_counts, unique_assignments)

    plot_labels_and_spike_rates_bargraph(max_diags, max_off_diags, max_off_diags_on_diags_vals)

    plot_labels_and_spike_rates_boxplots(summed_rates, num_testing_imgs, min_diags, min_off_diags) 


    reso = (540, 540) 

    if plot_ref_qry:

        for imgIdx in range(len(testImgs)):  
            show_qry_match_gt_imgs(testImgs, trainImgs, orgTestImgs, orgTrainImgs, unique_test_labels, imgIdx, num_unique_test_labels, data_path, True) 
            # sys.exit()


    if do_extended_plotting: 

        x_values = np.arange(0, 400)
        for label_i in range(num_unique_test_labels):
            plot_features(label_i, x_values)

        for i in range(num_testing_imgs):
            plot_rates_detailed(i)

        for i in range(num_testing_imgs):
            get_rates()

        assignment_list = np.sort(os.listdir(assignments_path))
        assignment_list = [os.path.join(assignments_path,f) for f in assignment_list if 'assignments' in f]
        assignment_list.sort(key=lambda f: int( re.sub('\D', '', f)))
        num_iter = num_unique_test_labels if not place_familiarity else num_true_labels
        hard_assignments = [] 

        for i, aPath in enumerate(assignment_list):

            if 'assignments' not in aPath:
                continue

            assignments_i = np.load(aPath)
            hard_assignments.append(assignments_i)


        for j in range(num_iter): 
            plot_label_assignments(j)


        for j in range(n_e):
            plot_neuron_assignments(j)


    f.close()


    #     # show_qry_match_gt_imgs_basic(testImgs, trainImgs, orgTestImgs, orgTrainImgs, testing_labels, matchIndices, imgIdx, data_path, True)  
    #     # plot_feature_representations(testing_labels[imgIdx], assignments, min_diags, min_off_diags, data_path)



