import tskit
import pyslim
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, required=True) # directory containing the tree-sequence files
parser.add_argument('--seed', type=str, required=True) # seed used for SLiM simulation
parser.add_argument('--P', type=str, required=True) # admixture proportion, comma-separated
parser.add_argument('--nDiploid', type=str, required=True) # number of diploid samples to include
parser.add_argument('--L', type=str, required=True) # block length
parser.add_argument('--nResampling', type=str, required=True) # block length

args = parser.parse_args()


# prepare arguments
dir_source = args.dir
seed = args.seed
P = [float(i) for i in ((args.P).split(','))] # initial admixture proportion
n_diploid_ind = int(args.nDiploid) # number of diploid individuals to sample
L = int(args.L) # length of each block
N_resampling = int(args.nResampling) # number of resamplings to perform 

# generations to work on
gen_range = np.concatenate([np.arange(1,101,1),
                            np.arange(105,501,5),
                            np.arange(520,1001,20),
                            np.arange(1050,2001,50)
                           ])

# change directionary to the simulation folder
os.chdir(dir_source+'/slim_seed'+seed)


# partition ancestor to K sources
def partition_ancestors(P,ind_ancestor):
    
    K = len(P)
    
    N = len(ind_ancestor)
    
    N_k = []
    
    for k in np.arange(0,K-1):
        
        N_k.append(int(np.round(N*P[k])))
        
    N_k.append(int(N-np.sum(N_k)))
    
    groups = np.array([])
    
    for k in np.arange(0,K):
        
        groups = np.concatenate([groups,np.zeros(N_k[k])+k])
        
    groups = np.random.permutation(groups)
            
    return groups.astype(int)

# sample nodes from the current generation
def sample_current_nodes(ts,n_ind=1):
    
    alive_inds = pyslim.individuals_alive_at(ts, 0)
    keep_indivs = np.random.choice(alive_inds, n_ind, replace=False)
    keep_nodes = []
    for i in keep_indivs:
          keep_nodes.extend(ts.individual(i).nodes)
            
    return keep_nodes

# get ancestry states for each haplotype
def get_ancestry_for_haplotypes(ts,
                                interval,
                                anc_ancestral_haplotype,
                                node_ancestral_haplotype
                                ):
    
    L = int(ts.sequence_length)
    l_left,l_right = interval #l_left: inclusive; l_right: exclusive
    
    ind_current = pyslim.individuals_alive_at(ts, 0)   
    hap_current = np.concatenate([ts.individual(i).nodes for i in ind_current])
    
    anc_blocks = []
    
    for h in hap_current:
        
        ts_singleton = ts.simplify([h],keep_input_roots=True)
        
        breakpoints = ts_singleton.breakpoints([0])
        
        roots = np.array([t.root for t in ts_singleton.trees()])
        
        # If roots is not a contiguous series of integer from 0:
        # roots_index = np.array([np.where(r==node_ancestral_haplotype)[0][0] for r in roots])
        
        # If roots is a contiguous series of integer from 0:
        # roots_index = roots
        
        roots_slim_id = [ts_singleton.node(root).metadata['slim_id'] for root in roots] # slim haplotype id
        
        ancestry = anc_ancestral_haplotype[roots_slim_id]
        
        anc_block = np.zeros(l_right-l_left)
        
        breakpoints_idx = np.where((breakpoints >= l_left) & (breakpoints < l_right))[0]
        
        if len(breakpoints_idx)==0:
            
            anc_block[:] = ancestry[np.where(breakpoints < l_left)[0][-1]]
            
        else:
        
            bp_included = (breakpoints[breakpoints_idx]).astype(int)
            bp_included = np.concatenate([bp_included,[l_right+1]]) # append a dummy right element
            anc_included = ancestry[breakpoints_idx]
                        
            anc_block[0 : bp_included[0] - l_left] = ancestry[breakpoints_idx[0]-1]

            for i in range(len(breakpoints_idx)):
                                
                anc_block[bp_included[i] - l_left : np.minimum(bp_included[i+1] - l_left,l_right-l_left)] = anc_included[i]

        anc_blocks.append(anc_block)
        
        
    return  anc_blocks

def anc_categorical_to_vectorized(anc_categorical,K):
    
    # K: number of distinct categories
    
    # convert categorical haplotype ancestry [0,1,2,1,0]
    # to vectors
    
    if len(anc_categorical.shape) == 2:
        
        n,y = anc_categorical.shape
        
        anc_vectorized = []
        for i in range(n):
            anc_vectorized.append([anc_categorical[i] == k for k in range(K)])
        
    else:
        
        anc_vectorized = [anc_categorical == k for k in range(K)]
        
    return np.array(anc_vectorized) * 1



def Sb2_haplotypes(block_ancestry):
    
    # block_ancestry is an numpy array of ancestry states along haplotype blocoks
    
    n = block_ancestry.shape[0]
    
    cij = np.array([])
    
    # only doing for i != j
    for i in np.arange(0,n-1):
        for j in np.arange(i+1,n):
            cij = np.append(cij,np.mean(block_ancestry[i] == block_ancestry[j]))
            
    Sb2 = 1 - (np.sum((cij)**2)*2 + n) / (n**2)

    return Sb2

def Sw2_haplotypes(vectorized_ancestry):
    
    # input must be numpy arrays of the form:
    
    #    k=0    k=1
    # [[[0,1],[1,0]], # haplotype 1
    #  [[1,1],[0,0]], # haplotype 2
    #  [[1,0],[0,1]]] # haplotype 3
    
    # n * K * L dimensional arrays
    
    n = vectorized_ancestry.shape[0]
    K = vectorized_ancestry.shape[1]
    L = vectorized_ancestry.shape[2]
    M = n*K
    
    anc = np.concatenate(vectorized_ancestry)
    
    # case: i != j
    
    aij = []
    
    for i in np.arange(0,M-1):
        for j in np.arange(i+1,M):
            aij.append(np.sum(anc[i]*anc[j]))
            
    aij = np.array(aij)
            
    aii = [np.sum(anc[i]) for i in range(M)]
    
    aii = np.array(aii)
    
    return 1 - (2*(np.sum(aij**2)) + np.sum(aii**2))/((L*n)**2)



def prepare_treesequences(filename,P):
    
    # load a tree-sequence file and impose ancestry in founder individuals
    
    # filename: filename of the .trees file
    # P: ancestry contribution in the 0-th generation (e.g., [0.2,0.7,0.1])
    
    ts = tskit.load(filename)
    
    gen = ts.metadata['SLiM']['tick']

    ind_ancestor = pyslim.individuals_alive_at(ts, gen)
    ind_current = pyslim.individuals_alive_at(ts, 0);
    
    # assigning ancestry to first-generation individuals

    anc_ancestor = partition_ancestors(P,ind_ancestor)
    anc_ancestral_haplotype = np.concatenate([[anc_ancestor[i],anc_ancestor[i]] for i in range(len(anc_ancestor))])
    node_ancestral_haplotype = np.concatenate([ts.individual(i).nodes for i in ind_ancestor])
    
    return ts, anc_ancestral_haplotype, node_ancestral_haplotype


def get_entropy_from_treesequences(ts,
                                   P,
                                   anc_ancestral_haplotype,
                                   node_ancestral_haplotype,
                                   interval=[0,10],
                                   n_diploid_ind=1):
    
    # sample from a .trees file and calculate entropy

    
    # n_diploid_ind: number of diploid individuals
    # interval: the interval to calculate entropy
    
    node_sample = sample_current_nodes(ts,n_ind=n_diploid_ind)
    sts = ts.simplify(node_sample, keep_input_roots=True)
    
    # extract ancestry states for each haplotype block

    block_ancestry = get_ancestry_for_haplotypes(sts,interval,anc_ancestral_haplotype,node_ancestral_haplotype)
    block_ancestry = np.array(block_ancestry)
    
    anc_vectorized = anc_categorical_to_vectorized(block_ancestry,len(P))
    
    Sb2 = Sb2_haplotypes(block_ancestry)
    Sw2 = Sw2_haplotypes(anc_vectorized)
    
    return Sb2,Sw2


# intervals = [[int(i),int(i)+int(L)] for i in np.arange(0,int(1e7),L)] # selected genomic intervals
intervals = [[int(i),int(i)+int(L)] for i in np.arange(0,int(1e7),L)] # selected genomic intervals


Sw_dict = dict()
Sb_dict = dict()

for gen in gen_range:
    
    print('generation: '+str(gen), flush=True)
    
    # # ======
    # # uncomment this code block if ancestry is determined once for all iteractions:
    
    # ts, ancestry_ancestral_haplotype, node_ancestral_haplotype = prepare_treesequences(
    #                     'slim_seed'+seed+'_gen'+str(gen)+'.trees',
    #                     P)
    
    # # ======
    
    Sb_ensemble = []
    Sw_ensemble = []
    
    # each i is a fresh group of samples
    for i in range(N_resampling):
        
        # ======
        # uncomment this code block if ancestry is redetermined for each iteraction:
        ts, ancestry_ancestral_haplotype, node_ancestral_haplotype = prepare_treesequences(
                        'slim_seed'+seed+'_gen'+str(gen)+'.trees',
                        P)
        # ======
        
        Sb_intervals = []
        Sw_intervals = []
        
        for interval in intervals:

                Sb_0, Sw_0 = get_entropy_from_treesequences(ts,
                                P,
                                ancestry_ancestral_haplotype,
                                node_ancestral_haplotype,
                                interval=interval,
                                n_diploid_ind=n_diploid_ind)

                Sb_intervals.append(Sb_0)
                Sw_intervals.append(Sw_0)
                
        Sb_ensemble.append(Sb_intervals)
        Sw_ensemble.append(Sw_intervals)
        
    Sb_dict['gen={},L={}'.format(gen,L)] = Sb_ensemble
    Sw_dict['gen={},L={}'.format(gen,L)] = Sw_ensemble
    

P_notation = ','.join([str(i) for i in P])

np.save('Entropy_Sb2_P_{}_nDiploid_{}_L_{}_nResampling_{}.npy'.format(P_notation,n_diploid_ind,L,N_resampling),  Sb_dict)    
np.save('Entropy_Sw2_P_{}_nDiploid_{}_L_{}_nResampling_{}.npy'.format(P_notation,n_diploid_ind,L,N_resampling),  Sw_dict)    






