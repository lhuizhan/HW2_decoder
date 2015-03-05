#!/usr/bin/env python
import optparse
import sys
import models
import operator
import time
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=10000, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-r", "--iteration", dest="iteration", type="int", default=5,  help="LR iteration")
optparser.add_option("-e", "--improve_size", dest="e", default=0.3, type="float", help="threshold for examine the stop point of LR iteration")
opts = optparser.parse_args()[0]

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, start, end, bitstring")      
LR_state = namedtuple("LR_state", "lm_state, start, end, bitstring")                                                                 
  #n: the number of words that have been translated
  #l and m specify a contiguous span in the source sentence
  #r is the end position of the previous phrase
best_score = namedtuple("best_score", "iteration, score")

tm = models.TM(opts.tm, opts.k)
#tm = {('les', 'membres', 'de'): [phrase(english='the men and women in', logprob=-1.79239165783)]}
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):#find distinct word tuple
  # word is a single word 
  if (word,) not in tm: #match the first word
    #phrase = namedtuple("phrase", "english, logprob")
    tm[(word,)] = [models.phrase(word, 0.0)]


def extract_english(h): 
  #h.predecessor = hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def UpdateMultiplier(U, y_count, alpha):
  for i in xrange(len(U)):
    U[i] = U[i] - alpha*(y_count[i] - 1)
  if alpha > 0.01:
    alpha /= 1.2
  return U
  

def count_y(y_count, y_star):
  for i in range(y_star.start, y_star.end):
    y_count[i] += 1

  if y_star.predecessor is None:
    return y_count
  else:
    return count_y(y_count, y_star.predecessor)


def Check_Y_Count(f, y_star):
  opt_flag = True
  y_count = {i:0 for i in range(len(f))}
  y_count = count_y(y_count, y_star)
  for i in range(len(f)):
    if y_count[i] != 1:
      opt_flag = False  
  """
  sys.stderr.write(str(y_count))
  sys.stderr.write("\n")
  """
  return (y_count, opt_flag)

def distorsion(preEnd, nowStart):
  return abs(preEnd + 1 - nowStart)


def Find_Y_Star(f, C, U):  
  bc = tuple(0 for _ in range(len(f)))

  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0, 0, bc)
  y = [0 for _ in xrange(len(f))]
  stacks = [{} for _ in f] + [{}] 
  LR_stack = [{} for _ in f] + [{}]

  stacks[0][lm.begin()] = initial_hypothesis
  #[{('<s>',): hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for n in xrange(len(f)):
        for j in xrange(n+1,min(len(f)+1, n+1+len(f)-i)):#i start from 0
          if f[n:j] in tm: #tm = {('les', 'membres', 'de'): [phrase(english='the men and women in', logprob=-1.79239165783)]}                      
            for phrase in tm[f[n:j]]:
              logprob = h.logprob + phrase.logprob
              lm_state = h.lm_state

              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob         

              logprob += lm.end(lm_state) if j == len(f) else 0.0 #check                            
              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, n, j, None)                            
              (y_count, opt) = Check_Y_Count(f, new_hypothesis)
              
              valid = True
              bc_list = list(h.bitstring)
              
              for x in range(n, j):
                if x in C and (bc_list[x] + 1 > 1):
                  valid = False           
              
              if valid:    
                for x in range(n,j):            
                  bc_list[x] += 1
                  logprob += float(U[x])
                logprob -= distorsion(h.end, n)
              else:          
                logprob = float("-inf")


              bc = tuple(bc_list)

              new_LR_state = LR_state(lm_state, n, j, bc)
              new_hypothesis = new_hypothesis._replace(logprob=logprob, bitstring=bc)        
              
              
              if new_LR_state not in LR_stack[i+j-n] or LR_stack[i+j-n][new_LR_state].logprob < new_hypothesis.logprob:
                if new_hypothesis.logprob > float("-inf"):
                  stacks[i+j-n][lm_state] = new_hypothesis
                  LR_stack[i+j-n][new_LR_state] = new_hypothesis

              #if (lm_state) not in stacks[i+j-n] or stacks[i+j-n][lm_state].logprob < new_hypothesis.logprob: # second case is recombination
              #  stacks[i+j-n][lm_state] = new_hypothesis
              #  sys.stderr.write(str(new_hypothesis.bitstring))
              #  sys.stderr.write("\n")

  for key, value in sorted(LR_stack[-1].iteritems(), key=lambda (k,v): (v,k), reverse=True):    
    bc = key.bitstring
    win_flag = True
    for index in C:
      if bc[index] != 1:
        win_flag = False
    if win_flag:
      return (value, value.logprob)
    
  sys.stderr.write("No!!   No!!\n")  
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  return (winner, winner.logprob)
  

def Optimize(f, C, U, alpha):
  U = [0.0 for _ in xrange(len(f))]
  alpha = 2.0
  sys.stderr.write("Optimize\n")
  sys.stderr.write("C = %r\n" % str(C))

  improve = True
  iteration = 0
  best_score_1 = best_score(0, float("-inf"))
  best_score_2 = best_score(0, float("-inf"))

  while(improve):
    iteration += 1
    if iteration%10 == 0:
      sys.stderr.write("LR iteration: ")
      sys.stderr.write(str(iteration))
      sys.stderr.write("\n")

    (y_star, y_score) = Find_Y_Star(f, C, U)
    (y_count, opt) = Check_Y_Count(f, y_star)
    sys.stderr.write("C   = %r\n" % str(C))
    sys.stderr.write("Count %r\n\n" % str(y_count))
    if opt:
      return y_star
    else:
      U = UpdateMultiplier(U, y_count, alpha)
    
    if y_score > best_score_1.score:
      if iteration > 1:
        best_score_2 = best_score_1
      best_score_1 = best_score(iteration, y_score)
    elif y_score > best_score_2.score:
      best_score_2 = best_score(iteration, y_score)

    if (iteration > best_score_2.iteration):
      sys.stderr.write("Improve rate: ")
      sys.stderr.write(str((best_score_1.score - best_score_2.score)/float(iteration - best_score_2.iteration)))
      sys.stderr.write("\n")
      #if (best_score_1.score - best_score_2.score)/float(iteration - best_score_2.iteration) < opts.e:
      if ((best_score_1.score - best_score_2.score)/float(iteration - best_score_2.iteration) < opts.e) or ((best_score_1.iteration + 50 < iteration) and (best_score_2.iteration + 50 < iteration)):
        improve = False
  
  count = [0 for _ in xrange(len(f))] #initialize count
  
  sys.stderr.write("For loop in Optimize")
  sys.stderr.write("\n")
  for k in range(opts.iteration):
    #sys.stderr.write("k = ")
    #sys.stderr.write(str(k))
    #sys.stderr.write("\n")
    (y_star, y_score) = Find_Y_Star(f, C, U)
    (y_count, opt) = Check_Y_Count(f, y_star)
    if (opt):
      return y_star
    else:
      U = UpdateMultiplier(U, y_count, alpha)
      for i in xrange(len(f)):
        if y_count[i] > 1:
           count[i] += y_count[i]

  expand_C = []
  for c in C:
    expand_C.append(c)
    expand_C.append(c+1)
    expand_C.append(c-1)
  #sys.stderr.write("expand_C = ")
  #sys.stderr.write(str(expand_C))
  #sys.stderr.write("\n")

  #G = int(len(f) / (2*opts.iteration)) # add G hard constraints
  G = 1
  #sys.stderr.write("G = %r\n" % str(G))
  
  for key, value in sorted(y_count.items(), key=operator.itemgetter(1), reverse=True):
    #sys.stderr.write("K, v = %r, %r" %(key, value))
    if key not in C and value > 0 and G > 0 and len(C) < len(f)-1:   
      #sys.stderr.write("add c = %r" % key)   
      C.append(key)
      G -= 1
      
  return Optimize(f, C, U, alpha) 


"""
=============================================================================
"""
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # f = ('honorables', 's\xc3\xa9nateurs', ',', 'que', 'se', 'est', '-', 'il', 'pass\xc3\xa9', 'ici', ',', 'mardi', 'dernier', '?')

  alpha = 1.0
  U = [0.0 for _ in xrange(len(f))] #initialize multipliers
  C = [] #Hard constraint
  optResult = Optimize(f, C, U, alpha)
  print extract_english(optResult)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
