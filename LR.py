#LR related functions
import numpy as np 


"""
xx_M means a matrix
xx_S means a set
------------------------------------------------------------------------
A_M     : constraint matrix |b|x|E|, A_M x = b; b = <1, 1, 1, ...>
b       : encoding b constraints
alpha   : subgradient rates
beta    : threshold for pruning
delta   : parameters of the language model mapping a bigram of target-language words to a real-valued score.
E_S     : set of hyperedge
e       : hyperedge, ((v2, v3, ..., vn), v1)
h(e)    : v1, head of hyperedge
K       :
t(e)    : (v2, v3, ..., vn)
theta   : weight vector for the hypergraph
thetaP  : theta'
theta(e): The weight of this hyperedge e, theta(e) = translation model score + language model score
tau     : weight offset
tauP    : tau'
lb      : LR lower bound
omega   : parameters for the translation model mapping each pair in P_S to a real-valued score
opt     : certificate of optimality
P_S     : a set of pairs (q,r), where q1, q2... q|q| is a sequence of source-language words, 
          r1, r2, ..., r|r| is a sequence of target-language words drawn from the target vocabulary Sigma
p       : sequence of phrases = p1 p2 p3...
p(e)    : phrase of e
pn      : (q, r, j, k): source words (q=wj...wk) are translated to r
pi      : possible hypothesis
f(p)    : The score of a derivation is the sum of the translation score of each phrase 
          plus the language model score of the target sentence
q       : source sentence = wj...wk
r       : target words r1, r2...
Sigma   : target vocabulary
sig     :
sig_i   :
TP      :
theta   :
u       : sequence of words in sigma formed by concatenating the phrases r(p1)...r(pn), 
          with boundary cases u0 = <s> and u_{|u|+1} = </s>
ub      : LR upper bound
V_S     :
v = tm  : vertex,  (c,u) where c in {1, 2, ...|w|} is the count of source words translated;
                               u is the last target-language word produced by a partial hypothesis at this vertex
X_S     : hyperpath set
x       : x(e) = 1 if hyperedge e is used in the hyperpath, otherwise 0
w       : source word
"""

def LRRound(alpha_k, lamda):
   #compute the subgra- dient, update the dual vector, and check for a certificate
   for x in X: 
     x = argmax

   lamdaP = lamda - alpha_k(AM*x - b)
   opt = (AM*x == B)
   ub = theta*x + tau
   return (lamdaP, ub, opt)


def LR(alpha):	
   #LagrangianRelaxation
   #alpha = alpha_1, alpha_2,..., alpha_K
   lamda[0] = 0
   for k in range(K):
      (lamda[k], ub, opt) = LRRound(alpha[k], lamda[k-1])
      if opt:
      	return (lamda[k], ub, opt)

   return (lamda[-1:], ub, opt)


def BestPathScore(theta, t):



	return pi[1] + t





