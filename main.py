from os import name
from re import L
import numpy as np
import math
from numpy.core.fromnumeric import ptp
from numpy.linalg import inv, det, pinv
import pandas as pd
from ete3 import Tree
from Bio import Phylo
from six import b
import matplotlib.pyplot as plt
# Brownian motion likelihood computation
# Brownian motion models can be completely described by two parameters. The first is the starting value of the population mean trait, z¯(0).
# This is the mean trait value that is seen in the ancestral population at the start of the simulation, before any trait change occurs.
# The second parameter of Brownian motion is the evolutionary rate parameter, σ^2. This parameter determines how fast traits will randomly walk through time.

# Under Brownian motion, changes in trait values over any interval of time are always drawn from a normal distribution
# with mean 0 and variance proportional to the product of the rate of evolution and the length of time (variance = σ^2t).
# x is an n x 1 vector of trait values for the n tip species in the tree

np.seterr(over='raise')
def Brownian_motion_likelihood(X,Z0, deltaSquare, C):
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    Z0_vector = Z0 * one
    XSubZ0_vector = X - Z0_vector
    # This is because pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
    temp = np.dot(np.dot(np.transpose(XSubZ0_vector), pinv(deltaSquare * C)), XSubZ0_vector)
    numerator = math.exp(-1/2 * temp) # This is correct
    try:
        denominator = math.sqrt(((2 * math.pi) ** len(X)) * det(deltaSquare * C)) # This is correct
        #denominator = math.sqrt(((2 * math.pi) ** len(X)) * deltaSquare**len(C) * det(C))
    except FloatingPointError: # OK to have this error and continue executing next time
        return 0
    likelihood = numerator / denominator
    return likelihood # correct

def Brownian_motion_maximumlikelihood(X,Z0, deltaSquare, C):
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    z0hat_front = pinv(one.T @ pinv(C) @ one)
    z0hat_end = one.T @ pinv(C) @ X
    # estimated root state for the character
    z0hat = z0hat_front * z0hat_end

    # maximum likelihood delta square
    numerator = (X - z0hat * one).T @ pinv(C) @ (X - z0hat * one)
    denominator = len(X)
    # estimated net rate of evolution
    deltaSquarehat = numerator / denominator
    return z0hat, deltaSquarehat

# Based on Pagel's lambda to transform the phylogenetic variance-covariance matrix.
# compresses internal branches while leaving the tip branches of the tree unaffected
def lambdaCovarience(C, lambdaVal):
    n = len(C)
    for i in range(0,n):
        for j in range(0,n):
            # Off diagonal times lambda
            if i != j:
                C[i][j] = C[i][j] * lambdaVal
    return C

# Compute MLE for a given lambda value
def Pagel_lambda_MLE(X,Z0, deltaSquare, C, lambdaVal):
    # Compute new covarience matrix
    C_lambda = lambdaCovarience(C, lambdaVal)
    z0hat, deltaSquarehat = Brownian_motion_maximumlikelihood(X, Z0, deltaSquare, C_lambda)
    # Compute likelihood
    Pagel_likelihood = Brownian_motion_likelihood(X,z0hat,deltaSquarehat, C_lambda)
    return Pagel_likelihood, z0hat, deltaSquarehat, lambdaVal

# Searching with different step sizes. Finding out lambda's value to maximize likelihood
def Found_Pagel_Maximumlikelihood(X,Z0, deltaSquare, tree, stepSize, startSearch = 0, EndSearch = 1):
    try:
        # Simple input checking.
        if stepSize <= 0 or startSearch < -1 or EndSearch > 2 or (startSearch > EndSearch):
            return
        # Initialization
        lambdaVal = startSearch
        maxlikelihood = -math.inf
        maxlikelihood_lambda = -math.inf
        # Record all likelihood and lambda value
        likelihoodSave = []
        lambdaValSave = []
        # Try different lambda values and try to find its corresponding MLE
        while lambdaVal <= EndSearch:
            # print(lambdaVal)
            # Recompute C every time because it will be overwrited
            C = Covariance(tree)
            tmp_likelihood,tmp_z0hat, tmp_deltaSquarehat, tmp_lambdaVal= Pagel_lambda_MLE(X,Z0,deltaSquare, C, lambdaVal)
            # If tmp value is larger
            if maxlikelihood < tmp_likelihood:
                maxlikelihood = tmp_likelihood
                maxlikelihood_lambda = lambdaVal
            likelihoodSave.append(tmp_likelihood)
            lambdaValSave.append(lambdaVal)
            lambdaVal += stepSize
    except:
        return None, None, [], []
    # Return all of them
    return maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave
# This function will return covariance
def Covariance(bac_tree):
    # Create n by n matrix Make sure pruning the tree or it maybe too large and have error occur.
    C = np.zeros(shape=(len(bac_tree), len(bac_tree)))
    # Used to tranverse through the matrix
    i_counter = -1
    j_counter = -1
    # Tranverse through all leaves
    for leaf_i in bac_tree:
        # Corresponding index
        i_counter += 1
        # Trnaverse through all leaves
        for leaf_j in bac_tree:
            j_counter += 1
            # If they are the same leaf
            if leaf_i == leaf_j:
                # Covariance is just its distance to the root
                C[i_counter][j_counter] = leaf_i.get_distance(bac_tree)
            else:
                # Get their first common ancestor and compute its distance to root
                commonAncestor = leaf_i.get_common_ancestor(leaf_j)
                C[i_counter][j_counter] = commonAncestor.get_distance(bac_tree)
        j_counter = -1
    return C

def traitsColumnReturn(df, traits_name):
    traits = list(df.loc[:,f'{traits_name}'])
    X = []
    for i in traits:
        X.append([i])
    X = np.array(X)
    return X

''' Simple testcase
C = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(lambdaCovarience(C,3))'''

'''Examination, we have a correct answer for covariance matrix
bac_tree = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")
print(bac_tree)
print(Covariance(bac_tree))'''


# IGG = pd.read_pickle('./Data/igg_haojun.pkl')

def Seedplant_sanity():
    seedplant = Tree("../seedplantData/seedplantsNew.tre", format = 0)
    df = pd.read_csv("../seedplantData/seedplants_Formatted.csv")
    # This works drop all rows with NAN value
    df = df.dropna()
    keep = list(df.loc[:,"Code"])
    # Save the tree befor prune
    seedplantSave = seedplant
    # Prune will modify the original tree (Pruning the tree), now seedplant is pruned
    seedplant.prune(keep, preserve_branch_length=True)
    C = Covariance(seedplant)

    # Preserve column names
    headers = list(df.columns)
    reorder = pd.DataFrame(columns=headers)
    # Reorder the data make them in the order of the tree labels
    for leaf in seedplant:
        row = df.loc[df['Code'] == leaf.name]
        reorder = reorder.append(row,ignore_index=True)

    # traits maxH
    maxH = list(reorder.loc[:,'maxH'])
    X = []
    for i in maxH:
        X.append([i])
    X = np.array(X)

    # maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, seedplant, 0.01, startSearch=0.0, EndSearch=1)
    # print(f"traits: maxH; ln Maximum likelihood: {np.log(maxlikelihood)}; lambda: {maxlikelihood_lambda};")

    # # traits
    # X = traitsColumnReturn(reorder, 'Wd')
    # maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, seedplant, 0.01, startSearch=0.0, EndSearch=1)
    # print(f"traits: Wd; ln Maximum likelihood: {np.log(maxlikelihood)}; lambda: {maxlikelihood_lambda};")

    X = traitsColumnReturn(reorder, 'Sm')
    maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, seedplant, 0.01, startSearch=0.0, EndSearch=1)
    print(f"traits: Sm;  lambda: {maxlikelihood_lambda};")

    # X = traitsColumnReturn(reorder, 'Shade')
    # maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, seedplant, 0.01, startSearch=0.0, EndSearch=1)
    # print(f"traits: Shade; ln Maximum likelihood: {np.log(maxlikelihood)}; lambda: {maxlikelihood_lambda};")

    # X = traitsColumnReturn(reorder, 'N')
    # maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, seedplant, 0.01, startSearch=0.0, EndSearch=1)
    # print(f"traits: N; ln Maximum likelihood: {np.log(maxlikelihood)}; lambda: {maxlikelihood_lambda};")

# This the maximum likelihood of a continuous function, thus maximum likelihood can be greater than 1.
# Seedplant_sanity()

# Step size 0.01
def Bacteria_Pagel():
    # Load trees
    # bac_tree is the root of the tree
    bac_tree = Tree('./Data/IGG_v1.0_bact_22515.tre')
    # Find correspondence
    IGG_Haojun = pd.read_csv("./Data/igg_haojun.csv")
    IGG_Haojun_altid2genomeid = IGG_Haojun.loc[:, ['genome_id', 'species_alt_id']]

    out_all = pd.read_csv("./Data/out-all-2.csv")
    out_all_genome_id = out_all.iloc[:,0]
    
    headers = list(IGG_Haojun_altid2genomeid.columns)
    translate = pd.DataFrame(columns=headers)

    # find correspondance
    for genome_id in out_all_genome_id:
        row = IGG_Haojun_altid2genomeid.loc[IGG_Haojun_altid2genomeid['genome_id'] == genome_id]
        translate = translate.append(row,ignore_index=True)
    # Slice all samples
    sample_id = list(out_all.columns)
    sample_id = sample_id[1:]
    # save for all maximum likelihood in a format of dictionary of dictionary
    maximumlikelihoodSave = {}
    for i in sample_id:
        out_all_new_sample = out_all.loc[:,i]
        tmp_translate = translate
        out = tmp_translate.join(out_all_new_sample)
        keep = out[pd.notnull(out[i])]
        keep = keep.reset_index(drop=True)
        keep_list = list(keep.iloc[:,1])
        # Make the become string
        for j in range(0, len(keep_list)):
            keep_list[j] = str(keep_list[j])
        # Create a tree to do pruning operation
        bac_tree_op = bac_tree.copy()
        bac_tree_op.prune(keep_list, preserve_branch_length=True)
        # Compute covarience
        C = Covariance(bac_tree_op)
        # Reorder the feature
        reorder_header = list(keep.columns)
        reorder = pd.DataFrame(columns=reorder_header)
        for leaf in bac_tree_op:
            row = keep.loc[keep['species_alt_id'] == int(leaf.name)]
            reorder = reorder.append(row,ignore_index=True)
        X = traitsColumnReturn(reorder, i)
        # Can be negative but be careful with the domain, I do not advise to do so because it is meaningless and may cause math domain error (math.sqrt(negative value))
        # Greater than 1 also has math domain error.
        maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X,1, 1, bac_tree_op, 0.01,startSearch=0,EndSearch=1)
        print(f'Sample: {i}; Maximum likelihood: {maxlikelihood}; Number of leaves: {len(keep)}; Lambda: {maxlikelihood_lambda}')

        if len(keep) not in maximumlikelihoodSave.keys():
            maximumlikelihoodSave[len(keep)] = {i:maxlikelihood_lambda}
        else:
            maximumlikelihoodSave[len(keep)][i] = maxlikelihood_lambda

        if maxlikelihood_lambda != 0 and maxlikelihood_lambda != None:
            plt.plot(lambdaValSave, likelihoodSave, 'ro')
            plt.xlabel("Lambda Value")
            plt.ylabel("Likelihood")
            plt.title(f'Pagel\'s Lambda: Lambda-Likelihood Plot \n \n\
                Sample: {i}; Maximum likelihood: {maxlikelihood}; \n \n Number of leaves: {len(keep)}; Lambda: {maxlikelihood_lambda}', fontweight='bold', fontsize=12)
            plt.xlim([0, 1])
            plt.savefig(f"./Plots/{i}%{len(keep)}.png", bbox_inches = 'tight')
            plt.clf()
    return maximumlikelihoodSave


maximumlikelihoodSave = Bacteria_Pagel()

# maximumlikelihoodSave needs a dictionary of dictionary
# numBins needs how many bins you want
def Lambda_Hist(maximumlikelihoodSave, numBins):
    lambdaList = []
    for leaf_num in maximumlikelihoodSave.keys():
        for sampleVal in maximumlikelihoodSave[leaf_num].keys():
            if maximumlikelihoodSave[leaf_num][sampleVal] != None:
                lambdaList.append(maximumlikelihoodSave[leaf_num][sampleVal])
    plt.hist(lambdaList, numBins)
    plt.title(f'Histogram of Maximum Likelihood Lambda \n \n Number of Valid Samples: {len(lambdaList)}', \
             fontweight='bold', fontsize=12)
    plt.xlabel("Lambda Values")
    plt.ylabel("Number of Samples")
    plt.savefig(f"./Plots/Lambda_Samples_Histogram.png", bbox_inches = 'tight')
    plt.clf()
    #plt.show()

Lambda_Hist(maximumlikelihoodSave, 10)

def leaves_lambda(maximumlikelihoodSave):
    leaves = []
    lambdas = []
    # {leaves: {sample:lambda}}
    for i in maximumlikelihoodSave.keys():
        for j in maximumlikelihoodSave[i].keys():
            if maximumlikelihoodSave[i][j] != None:
                leaves.append(i)
                lambdas.append(maximumlikelihoodSave[i][j])
    plt.plot(leaves, lambdas, 'ro')
    plt.xlabel("Number of Leaves")
    plt.ylabel("Lambda Values")
    plt.title(f"Pagel\'s Lambda -- Leaves-Lambda Distribution Plots \n \n\ Number of Samples: {len(lambdas)}")
    plt.savefig(f"./Plots/Leaves_lambdas.png", bbox_inches = 'tight')
    plt.clf()
    #plt.show()

leaves_lambda(maximumlikelihoodSave)