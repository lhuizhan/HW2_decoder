#based on the thesis "Optimal Beam Search for Machine Translation"
#!/usr/bin/env python
import optparse
import sys
import models
import LR
from collections import namedtuple
import numpy as np 

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="kk", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.kk)
#tm = {('les', 'membres', 'de'): [phrase(english='the men and women in', logprob=-1.79239165783)]}
lm = models.LM(opts.lm)
#n-gram model
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):#find distinct word tuple
  # word is a single word 
  if (word,) not in tm: #match the first word
    #phrase = namedtuple("phrase", "english, logprob")
    tm[(word,)] = [models.phrase(word, 0.0)]


def BeamSearch(theta, tau, lb, beta):
	ubs = OUTSIDE(theta, tau)
	opt = True
	for (v, ) in tm:
	  for x in sig:
	  	pi[v, x] = float("-inf")
	for 


	return (lb, opt)

def OUTSIDE(theta, tau):
    #computes upper bounds on outside scores	


def Prune(pi, v, sig, beta):
	#prunes all but the top beta scoring hypotheses in this set
	#if prune is perform, return true, and set opt = False


def theta(e):
	#return weight on edge
	#e = (foreign phrase, target phrase)
    f_word = e[0].split()
    e_word = e[1].split()
    s_score = 0.0
    for i in range(len(e_word)-1):
    	s_score += lm.score(e_word[i], e_word[i+1])
    s_score += lm.score(f_word[-1:], e_word[0])
	return (omega(e[0], e[1]) + d_score)


def omega(q, r):
	#return the translation model score
	tm_score = 0.0
	for q in tm:
	  for phrase in tm[q]:
	  	if phrase == r:
	  		tm_score += phrase.logprob
	return tm_score


def sigma(u1, u2):
	#return language model score
    for (u1, u2) in lm:



def Check(sig):
	#ensures that each word has been translated 0 or 1 times
	for x in sig:
	  if 
	return (sig_i <= 1)


def SIGS(tail):
	#return the set of possible signature combinations

	return 	


def OptBeamStaged(alpha, beta):
	(lamda, ub, opt) = LR(alpha)
	if opt:
	  return ub
	thetaP = theta - AT*lamda
	tauP = tau + lamda*b
	lb[0] = float("-inf")
	for (k, lb_i) in enumerate(lb):
	  (lb[k], opt) = BeamSearch(thetaP, tauP, lb[k-1], beta[k]) 
	  if opt:
		return lb[k]
	return max(lb)

"""
def OptBeam(alpha, beta):
	lamda[0] = 0.0
	lb[0] = float("-inf")
	for k in enumerate()
"""
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: 
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
 		  A_M = np.zeros(shape=(len(f),))
          b = np.ones(shape=())

























  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    #h.predecessor = hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

