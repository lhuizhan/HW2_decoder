#!/usr/bin/env python
import optparse
import sys
import models
import operator
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=20, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-r", "--iteration", dest="iteration", type="int", default=10,  help="LR iteration")
optparser.add_option("-e", "--improve_size", dest="e", default=0.002, type="float", help="threshold for examine the stop point of LR iteration")
opts = optparser.parse_args()[0]

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, start, end")                                                                       
  #n: the number of words that have been translated
  #l and m specify a contiguous span in the source sentence
  #r is the end position of the previous phrase
best_score = namedtuple("best_score", "iteration, score")

tm = models.TM(opts.tm, opts.k)
#tm = {('les', 'membres', 'de'): [phrase(english='the men and women in', logprob=-1.79239165783)]}
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]#make french sentence into a tuple with words
#french = [[('honorables', 's\xc3\xa9nateurs', ',', 'que', 'se', 'est', '-', 'il', 'pass\xc3\xa9', 'ici', ',', 'mardi', 'dernier', '?'), ......]

# initial LR multiplier step size
alpha = 1.0

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):#find distinct word tuple
  # word is a single word 
  if (word,) not in tm: #match the first word
    #phrase = namedtuple("phrase", "english, logprob")
    tm[(word,)] = [models.phrase(word, 0.0)]


def extract_english(h): 
  #h.predecessor = hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def UpdateMultiplier(U, y_count):
  #1. calculate multiplier
  #2. check solution
  opt_flag = True
  for i in xrange(len(U)):
    if y_count[i] != 1:
      opt_flag = False
    U[i] = U[i] - alpha*(y_count[i] - 1)
    alpha /= 2.0
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
  y_count = [0 for _ in range(len(f))]
  y_count = count_y(y_count, y_star)
  for i in range(len(f)):
    if y_count[i] != 1:
      opt_flag = False  
  return (y_count, opt_flag)


def Find_Y_Star(f, C, U):  
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0, 0)
  y = [0 for _ in xrange(len(f))]
  stacks = [{} for _ in f] + [{}] #n dict 
  #[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

  stacks[0][lm.begin()] = initial_hypothesis
  #[{('<s>',): hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

  for i, stack in enumerate(stacks[:-1]):#ignore the last one element in stacks[]
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for n in xrange(len(f)-1):
        for j in xrange(n+1,n+len(f)-i):#i start from 0
          if f[n:j] in tm: #tm = {('les', 'membres', 'de'): [phrase(english='the men and women in', logprob=-1.79239165783)]}          
            for phrase in tm[f[n:j]]:
              logprob = h.logprob + phrase.logprob
              lm_state = h.lm_state
              #lm_state = ('motion', 'is')
              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob
              logprob += lm.end(lm_state) if j == len(f) else 0.0 #check
              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, n, j)
              (y_count, opt) = Check_Y_Count(f, new_hypothesis)
              sys.stderr.write(str(y_count))
              sys.stderr.write("\n")
              
              for x in range(len(f)):
                #sys.stderr.write(str(float(U[x])*float((y_count[x] - 1))))
                #sys.stderr.write("-")
                logprob += float(U[x])*float((y_count[x] - 1))
                #sys.stderr.write(str(logprob))
                #sys.stderr.write("\n")
               
              new_hypothesis2 = hypothesis(logprob, lm_state, h, phrase, n, j)
              """
              sys.stderr.write(str(i+j-n))
              sys.stderr.write("-")
              sys.stderr.write(str(lm_state))
              sys.stderr.write("\n")
              sys.stderr.write(str(stacks[i+j-n]))
              sys.stderr.write("---")              
              sys.stderr.write("\n")  
              
              if lm_state not in stacks[i+j-n]:
                sys.stderr.write("not in stacks[i+j-n]")
                sys.stderr.write("\n")
              else:
                sys.stderr.write(str(stacks[i+j-n][lm_state].logprob))  
                sys.stderr.write("\n")
              """
              if lm_state not in stacks[i+j-n+1] or stacks[i+j-n+1][lm_state].logprob < new_hypothesis2.logprob: # second case is recombination
                sys.stderr.write("In stack--i-n-j = ")
                sys.stderr.write(str(i))
                sys.stderr.write("-")
                sys.stderr.write(str(n))
                sys.stderr.write("-")
                sys.stderr.write(str(j))
                sys.stderr.write(" = ")              
                sys.stderr.write(str(i+j-n+1))
                sys.stderr.write("\n")
                stacks[i+j-n][lm_state] = new_hypothesis2 
  
  
  sys.stderr.write(str(len(f)))
  sys.stderr.write("\n")

  sys.stderr.write(str(stacks[-1]))
  sys.stderr.write("\n")
  sys.stderr.write(str(len(stacks)))
  
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  sys.stderr.write(str(winner))
  sys.stderr.write("\n")
  sys.stderr.write(str(extract_english(winner)))
  return winner


def Optimize(f, C, U):
  improve = True
  iteration = 0
  best_score_1 = best_score(0, 0.0)
  best_score_2 = best_score(0, 0.0)

  while(improve):
    iteration += 1
    (y_star, y_score) = Find_Y_Star(f, C, U)
    (y_count, opt) = Check_Y_Count(f, y_star)
    if opt:
      return y_star
    else:
      U = UpdateMultiplier(U, y_count)

    if y_score > best_score_1.score:
      best_score_2 = best_score_1
      best_score_1 = (iteration, y_score)
    elif y_score > best_score_2.score:
      best_score_2 = (iteration, y_score)

    if (best_score_1.score - best_score_2.score)/float(iteration - best_score_2.iteration) < opts.e:
      improve = False
  
  count = [0 for _ in xrange(len(f))] #initialize count
  
  for k in range(opts.iteration):
    (y_star, y_score) = Find_Y_Star(f, C, U)
    (y_count, opt) = Check_Y_Count(f, y_star)
    if (opt):
      return y_star
    else:
      U = UpdateMultiplier(U, y_count)
      for i in xrange(len(f)):
        if y_count[i] != 1:
           count[i] += y_count[i]

  expand_C = []
  for c in C:
    expand_C.append(c)
    expand_C.append(c+1)
    expand_C.append(c-1)
    
  G = int(len(f) / (2*opts.iteration)) # add G hard constraints
  for key, value in sorted(y_count.itervalues(),key=lambda (k, v): (v,k), reverse=True):
    if key not in expand_C and G > 0:      
      C.append(key)
      G -= 1
      
  return Optimize(f, C, U) 


"""
=============================================================================
"""
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # f = ('honorables', 's\xc3\xa9nateurs', ',', 'que', 'se', 'est', '-', 'il', 'pass\xc3\xa9', 'ici', ',', 'mardi', 'dernier', '?')

  U = [0.0 for _ in xrange(len(f))] #initialize multipliers
  C = [] #Hard constraint
  optResult = Optimize(f, C, U)
  print extract_english(optResult)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
