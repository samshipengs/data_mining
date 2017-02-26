
## Hidden Markov Models in python

# Here we'll show how the Viterbi algorithm works for HMMs, assuming we have a trained model to start with. Further down we look at the forward and backward algorithms and Baum-Welch.

# In[1]:

import numpy as np


# Initialise the model parameters based on the example from the lecture slides

# In[2]:

A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
O = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])


# Now let's consider the example observation sequence UP, UP, DOWN, for which we'll try to discover the hidden state sequence.

# In[3]:

states = UP, DOWN, UNCHANGED = 0, 1, 2
observations = [UP, UP, DOWN]


# Now we'll code the Viterbi algorithm. It keeps a store of two components, the best scores to reach a state at a give time, and the last step of the path to get there. Scores alpha are initialised to -inf to denote that we haven't set them yet. 

# In[4]:

alpha = np.zeros((len(observations), len(states))) # time steps x states
alpha[:,:] = float('-inf')
backpointers = np.zeros((len(observations), len(states)), 'int')


# The base case for the recursion sets the starting state probs based on pi and generating the observation.

# In[5]:

# base case, time step 0
alpha[0, :] = pi * O[:,UP]
alpha


# Out[5]:

#     array([[ 0.35,  0.02,  0.09],
#            [ -inf,  -inf,  -inf],
#            [ -inf,  -inf,  -inf]])

# Now for the recursive step, where we maximise over incoming transitions reusing the best incoming score, computed above.

# In[6]:

# time step 1
for t1 in states:
    for t0 in states:
        score = alpha[0, t0] * A[t0, t1] * O[t1,UP]
        if score > alpha[1, t1]:
            alpha[1, t1] = score
            backpointers[1, t1] = t0
alpha


# Out[6]:

#     array([[ 0.35 ,  0.02 ,  0.09 ],
#            [ 0.147,  0.007,  0.021],
#            [  -inf,   -inf,   -inf]])

# Repeat with the next observation. (We'd do this as a loop in practice.)

# In[7]:

# time step 2
for t2 in states:
    for t1 in states:
        score = alpha[1, t1] * A[t1, t2] * O[t2,DOWN]
        if score > alpha[2, t2]:
            alpha[2, t2] = score
            backpointers[2, t2] = t1
alpha


# Out[7]:

#     array([[ 0.35   ,  0.02   ,  0.09   ],
#            [ 0.147  ,  0.007  ,  0.021  ],
#            [ 0.00882,  0.01764,  0.00882]])

# Now read of the best final state, and follow the backpointers to recover the full path.

# In[8]:

np.argmax(alpha[2,:])


# Out[8]:

#     1

# In[9]:

backpointers[2,1]


# Out[9]:

#     0

# In[10]:

backpointers[1,0]


# Out[10]:

#     0

# Phew. The best state sequence is [0, 0, 1]

### Formalising things

# Now we can put this all into a function to handle arbitrary length inputs 

# In[11]:

def viterbi(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s2 in range(S):
            for s1 in range(S):
                score = alpha[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
                if score > alpha[t, s2]:
                    alpha[t, s2] = score
                    backpointers[t, s2] = s1
    
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
        
    return list(reversed(ss)), np.max(alpha[M-1,:])


# In[12]:

viterbi((pi, A, O), [UP, UP, DOWN])


# Out[12]:

#     ([0, 0, 1], 0.017639999999999999)

# In[13]:

viterbi((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[13]:

#     ([0, 0, 2, 2, 2, 2, 0, 0], 6.2233919999999994e-06)

### Exhaustive method

# Let's verify that we've done the above algorithm correctly by implementing exhaustive search.

# In[14]:

from itertools import product

def exhaustive(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    # track the running best sequence and its score
    best = (None, float('-inf'))
    # loop over the cartesian product of |states|^M
    for ss in product(range(S), repeat=M):
        # score the state sequence
        score = pi[ss[0]] * O[ss[0],observations[0]]
        for i in range(1, M):
            score *= A[ss[i-1], ss[i]] * O[ss[i], observations[i]]
        # update the running best
        if score > best[1]:
            best = (ss, score)
            
    return best


# In[15]:

exhaustive((pi, A, O), [UP, UP, DOWN])


# Out[15]:

#     ((0, 0, 1), 0.017639999999999999)

# In[16]:

exhaustive((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[16]:

#     ((0, 0, 2, 2, 2, 2, 0, 0), 6.2233919999999994e-06)

# Yay, it got the same results as before. Note that the exhaustive method is practical on anything beyond toy data due to the nasty cartesian product. But it is worth doing to verify the Viterbi code above is getting the right results. 

### Supervised training, aka "visible" Markov model

# Let's train the HMM parameters on the Penn Treebank, using the sample from NLTK. Note that this is a small fraction of the treebank, so we shouldn't expect great performance of our method trained only on this data.

# In[17]:

from nltk.corpus import treebank


# Out[17]:

#     /Users/tcohn/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/numpy/core/fromnumeric.py:2499: VisibleDeprecationWarning: `rank` is deprecated; use the `ndim` attribute or function instead. To find the rank of a matrix see `numpy.linalg.matrix_rank`.
#       VisibleDeprecationWarning)
# 

# In[18]:

corpus = treebank.tagged_sents()
print corpus


# Out[18]:

#     [[(u'Pierre', u'NNP'), (u'Vinken', u'NNP'), (u',', u','), (u'61', u'CD'), (u'years', u'NNS'), (u'old', u'JJ'), (u',', u','), (u'will', u'MD'), (u'join', u'VB'), (u'the', u'DT'), (u'board', u'NN'), (u'as', u'IN'), (u'a', u'DT'), (u'nonexecutive', u'JJ'), (u'director', u'NN'), (u'Nov.', u'NNP'), (u'29', u'CD'), (u'.', u'.')], [(u'Mr.', u'NNP'), (u'Vinken', u'NNP'), (u'is', u'VBZ'), (u'chairman', u'NN'), (u'of', u'IN'), (u'Elsevier', u'NNP'), (u'N.V.', u'NNP'), (u',', u','), (u'the', u'DT'), (u'Dutch', u'NNP'), (u'publishing', u'VBG'), (u'group', u'NN'), (u'.', u'.')], ...]
# 

# We have to first map words and tags to numbers for compatibility with the above methods.

# In[19]:

word_numbers = {}
tag_numbers = {}

num_corpus = []
for sent in corpus:
    num_sent = []
    for word, tag in sent:
        wi = word_numbers.setdefault(word.lower(), len(word_numbers))
        ti = tag_numbers.setdefault(tag, len(tag_numbers))
        num_sent.append((wi, ti))
    num_corpus.append(num_sent)
    
word_names = [None] * len(word_numbers)
for word, index in word_numbers.items():
    word_names[index] = word
tag_names = [None] * len(tag_numbers)
for tag, index in tag_numbers.items():
    tag_names[index] = tag


# Now let's hold out the last few sentences for testing, so that they are unseen during training and give a more reasonable estimate of accuracy on fresh text.

# In[20]:

training = num_corpus[:-10]
testing = num_corpus[-10:]


# Next we compute relative frequency estimates based on the observed tag and word counts in the training set. Note that smoothing is important, here we add a small constant to all counts. 

# In[21]:

S = len(tag_numbers)
V = len(word_numbers)

# initalise
eps = 0.1
pi = eps * np.ones(S)
A = eps * np.ones((S, S))
O = eps * np.ones((S, V))

# count
for sent in training:
    last_tag = None
    for word, tag in sent:
        O[tag, word] += 1
        if last_tag != None:
            pi[tag] += 1
        else:
            A[last_tag, tag] += 1
        last_tag = tag
        
# normalise
pi /= np.sum(pi)
for s in range(S):
    O[s,:] /= np.sum(O[s,:])
    A[s,:] /= np.sum(A[s,:])


# Now we're ready to use our Viterbi method defined above

# In[22]:

predicted, score = viterbi((pi, A, O), map(lambda (w,t): w, testing[0]))


# In[23]:

print '%20s\t%5s\t%5s' % ('TOKEN', 'TRUE', 'PRED')
for (wi, ti), pi in zip(testing[0], predicted):
    print '%20s\t%5s\t%5s' % (word_names[wi], tag_names[ti], tag_names[pi])


# Out[23]:

#                    TOKEN	 TRUE	 PRED
#                        a	   DT	   DT
#                    white	  NNP	  NNP
#                    house	  NNP	  NNP
#                spokesman	   NN	   NN
#                     said	  VBD	  VBD
#                     last	   JJ	   JJ
#                     week	   NN	   NN
#                     that	   IN	  WDT
#                      the	   DT	   DT
#                president	   NN	   NN
#                       is	  VBZ	  VBZ
#              considering	  VBG	  VBG
#                      *-1	-NONE-	-NONE-
#                declaring	  VBG	  VBG
#                     that	   IN	  WDT
#                      the	   DT	   DT
#             constitution	  NNP	  NNP
#               implicitly	   RB	  SYM
#                    gives	  VBZ	  VBZ
#                      him	  PRP	  PRP
#                      the	   DT	   DT
#                authority	   NN	   NN
#                      for	   IN	   IN
#                        a	   DT	   DT
#                line-item	   JJ	   JJ
#                     veto	   NN	   VB
#                      *-2	-NONE-	-NONE-
#                       to	   TO	   TO
#                  provoke	   VB	   VB
#                        a	   DT	   DT
#                     test	   NN	   NN
#                     case	   NN	   NN
#                        .	    .	    .
# 

# Hey, not bad, only three errors. Can you explain why these might have occurred?

### Marginalisation in Hidden Markov Models

# A related problem is marginalisation, when we wish to find the probability of an observation sequence *under any hidden state sequence*. This allows hidden Markov models to function as language models, but also is key to unsupervised training and the central algorithm for training.

# As with the Viterbi algorithm, we'll need to start with the mathematical definition and attempt to factorise it (to follow a recursion, thus allowing for dynamic programming). The quantity we wish to compute is $$p(\vec{w}) = \sum_{\vec{t}} p(\vec{t}, \vec{w})$$
# where $w$ are the observations (words) and $t$ are the states (tags).

# Let's start by expanding the summation 
# $$
# p(\vec{w})  = \sum_{t_1} \sum_{t_2} \cdots \sum_{t_{N-1}} \sum_{t_N} p(\vec{t}, \vec{w})
# $$
# and expand the HMM probability 
# $$
# p(\vec{w})  = \sum_{t_1} \sum_{t_2} \cdots \sum_{t_{N-1}} \sum_{t_N} p(t_1) p(w_1 | t_1) p(t_2 | t_1) p(w_2| t_2) \cdots p(t_{N-1} | t_{N-2}) p(w_{N-1}| t_{N-1}) p(t_{N} | t_{N-1}) p(w_{N}| t_{N})
# $$

# Let's compare the full marginal probability $p(\vec{w})$ and the probability up to position $N-1$, finishing with tag $t_{N-1}$
# $$p(w_1, w_2, \ldots, w_{N-1}, t_{N-1}) = \sum_{t_1} \sum_{t_2} \cdots \sum_{t_{N-1}} p(t_1) p(w_1 | t_1) p(t_2 | t_1) p(w_2| t_2) \cdots p(t_{N-1} | t_{N-2}) p(w_{N-1}| t_{N-1})
# $$

# They look rather similar, and in fact we can express $p(\vec{w})$ more simply as
# $$
# p(\vec{w})  = \sum_{t_N} p(w_1, w_2, \ldots, w_{N-1}, t_{N-1}) p(t_{N} | t_{N-1}) p(w_{N}| t_{N})
# $$

# We can continue further by defining $p(w_1, w_2, \ldots, w_{N-1}, t_{N-1})$ in terms of $p(w_1, w_2, \ldots, w_{N-2}, t_{N-2})$ and so forth. (This is the same process used in the Viterbi algorithm, albeit swapping a max for a sum.)

# Formally we store a matrix of partial marginals, $\alpha$ defined as follows
# $$\alpha[i, t_i] = p(w_1, w_2, \ldots, w_i, t_i)$$
# computed using the recursion
# $$  
# \alpha[i, t_i] = \sum_{t_{i-1}} \alpha[i-1, t_i] p(t_i | t_{i-1}) p(w_i| t_i) 
# $$
# and the base case for $i=1$,
# $$
# \alpha[1, t_1] = p(t_1) p(w_1 | t_1)
# $$

# Now we have computed the formulation, we can put this into an iterative algorithm: compute the vector of alpha[1] values, then alpha[2] etc until we reach the end of our input

# In[24]:

def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((N, S))
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
    
    return (alpha, np.sum(alpha[N-1,:]))


# In[27]:

A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
O = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])

forward((pi, A, O), [UP, UP, DOWN])


# Out[27]:

#     (array([[ 0.35    ,  0.02    ,  0.09    ],
#            [ 0.1792  ,  0.0085  ,  0.0357  ],
#            [ 0.012605,  0.025176,  0.016617]]),
#      0.054398000000000002)

# In[28]:

forward((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[28]:

#     (array([[  3.50000000e-01,   2.00000000e-02,   9.00000000e-02],
#            [  1.79200000e-01,   8.50000000e-03,   3.57000000e-02],
#            [  1.26050000e-02,   2.51760000e-02,   1.66170000e-02],
#            [  5.35956000e-03,   3.52065000e-03,   6.34588000e-03],
#            [  1.50288260e-03,   8.28808500e-04,   1.97959280e-03],
#            [  2.10797093e-04,   4.48307010e-04,   4.36840386e-04],
#            [  3.67757541e-04,   2.20335560e-05,   1.05072304e-04],
#            [  1.91590157e-04,   9.06688053e-06,   3.91483114e-05]]),
#      0.00023980534876399999)

# Let's confirm we did this correctly by implementing an exhaustive equivalent

# In[29]:

def exhaustive_forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    total = 0.0
    # loop over the cartesian product of |states|^N
    for ss in product(range(S), repeat=N):
        # score the state sequence
        score = pi[ss[0]] * O[ss[0],observations[0]]
        for i in range(1, N):
            score *= A[ss[i-1], ss[i]] * O[ss[i], observations[i]]
        total += score
            
    return total


# In[30]:

exhaustive_forward((pi, A, O), [UP, UP, DOWN])


# Out[30]:

#     0.054397999999999988

# In[31]:

exhaustive_forward((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[31]:

#     0.00023980534876400081

### Backward algorithm

# The same process but working from left to right rather than right to left give us the backward algorithm.

# In[32]:

def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    
    beta = np.zeros((N, S))
    
    # base case
    beta[N-1, :] = 1
    
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
    
    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))


# Let's confirm the it gets the same marginal probability as the forward algorithm 

# In[33]:

backward((pi, A, O), [UP, UP, DOWN])


# Out[33]:

#     (array([[ 0.1216,  0.1077,  0.1076],
#            [ 0.24  ,  0.29  ,  0.25  ],
#            [ 1.    ,  1.    ,  1.    ]]),
#      0.054397999999999995)

### Unsupervised training

# Unsupervised training of a HMM involves running forward and backward to estimate the *expected* probability of taking various state sequences, then updates the model to match these *expectations*. This repeats many times until things stabilise (covergence). Note that it's non-convex, so the starting point often affects the converged solution.

# In[34]:

def baum_welch(training, pi, A, O, iterations):
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O) # take copies, as we modify them
    S = pi.shape[0]
    
    # do several steps of EM hill climbing
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
        
        for observations in training:
            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), observations)
            beta, zb = backward((pi, A, O), observations)
            assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
            
            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
                                                                    
        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
    
    return pi, A, O


# Let's test it out by training on our example from above

# In[35]:

pi2, A2, O2 = baum_welch([[UP, UP, DOWN]], pi, A, O, 10)


# In[36]:

forward((pi2, A2, O2), [UP, UP, DOWN])


# Out[36]:

#     (array([[  9.99894603e-01,   8.36665388e-12,   8.22616418e-05],
#            [  2.22272332e-01,   3.12183957e-02,   4.43718595e-01],
#            [  4.20221660e-07,   4.52046779e-01,   2.00391873e-02]]),
#      0.47208638604110348)

# Looks like it memorised the sequence, and has assigned it a very high probability. The downside is that it won't be very accepting of other sequences

# In[37]:

forward((pi2, A2, O2), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[37]:

#     (array([[  9.99894603e-01,   8.36665388e-12,   8.22616418e-05],
#            [  2.22272332e-01,   3.12183957e-02,   4.43718595e-01],
#            [  4.20221660e-07,   4.52046779e-01,   2.00391873e-02],
#            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00]]),
#      0.0)

# This looks strangely reminiscent of many other learning problems.... Can you think how we might deal with this?

# Incidentally training on both sequences leads to perhaps a better model

# In[38]:

pi3, A3, O3 = baum_welch([[UP, UP, DOWN], [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP]], pi, A, O, 10)


# In[39]:

forward((pi3, A3, O3), [UP, UP, DOWN])


# Out[39]:

#     (array([[  9.97616199e-01,   6.82049129e-15,   6.29715798e-14],
#            [  5.90761154e-01,   7.47740892e-03,   8.78089899e-06],
#            [  8.43266173e-04,   2.34964056e-01,   6.52309204e-04]]),
#      0.23645963152993088)

# In[40]:

forward((pi3, A3, O3), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])


# Out[40]:

#     (array([[  9.97616199e-01,   6.82049129e-15,   6.29715798e-14],
#            [  5.90761154e-01,   7.47740892e-03,   8.78089899e-06],
#            [  8.43266173e-04,   2.34964056e-01,   6.52309204e-04],
#            [  5.82459529e-09,   2.71931576e-08,   1.19461094e-01],
#            [  8.75785986e-10,   1.60469759e-06,   5.79399172e-02],
#            [  1.69690262e-05,   1.79695695e-02,   4.16867527e-03],
#            [  7.92352114e-03,   2.46221157e-05,   9.62381867e-05],
#            [  4.71403139e-03,   5.99516484e-05,   5.81125813e-07]]),
#      0.004774564161046658)

### Keeping (making?) in real

# This is a toy implementation of a HMM for pedagogical purposes. In reality we use several tricks to make things faster (e.g., matrix-vector operations) and to avoid floating point issues of underflow. These tricks complicate the code a fair bit. See the [Rabiner tutorial](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) for details, especifically the section about scaling factors. Another trick is to [work in log-space](http://machineintelligence.tumblr.com/post/4998477107/the-log-sum-exp-trick), which is easy for Viterbi but a bit more painful (and slower) for forward-backward.
