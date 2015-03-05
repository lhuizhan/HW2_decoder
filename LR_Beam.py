#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=20, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")

opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

#parameters
stepsize = 2.0
eta = -1.5
alpha_decrease_rate = 0.9
epsilon = 0.02
iteration_limit = 10
Constraints_limit = 2

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))

def beam(f, u, Constraints):
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  
  def distortion(end, newstart):
    return eta * abs(end + 1 - newstart)

  hypothesis = namedtuple("hypothesis", "logprob, LR_state, predecessor, phrase, y")
  LR_state = namedtuple("LR_state", "l, m, r, lm_state, bitstring")
  
  initial_y = [0 for i in range(len(f))]
  # initial_bitstring = '0' * len(Constraints)
  initial_bitstring = 0
  initial_lrstate = LR_state(0, 0, -1, lm.begin(), initial_bitstring)
  
  initial_hypothesis = hypothesis(0.0, initial_lrstate, None, None, initial_y)

  stacks = [{} for _ in f] + [{}]
  stacks[0][initial_lrstate] = initial_hypothesis
  
  for n, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      for s in xrange(0, len(f)):
        for t in xrange(s+1, len(f) + 1):

          if n + t - s > len(f):
            break

          if f[s:t] in tm:     
            #Update y vector
            new_y = h.y[:]
            for tmp in range(s, t):
              new_y[tmp] += 1

            #Compute the change of u[i] * y[i]            
            violation_change = sum(u[s: t])      
  
            if distortion(h.LR_state.r, s) > 6:
              continue


            for phrase in tm[f[s:t]]:

              current_lrstate = h.LR_state
              lm_state = current_lrstate.lm_state
              

              logprob = h.logprob + phrase.logprob + violation_change + distortion(h.LR_state.r, s)
              
              #Prune using constraint set, or update new constraint set
              new_bitstring = current_lrstate.bitstring
              flag = False

              # for (index, value) in enumerate(Constraints):
              #   if s<=value and value<t:
              #     if new_bitstring[index] == '1':
              #       flag = True
              #     else:
              #       new_bitstring[index] == '1'

              for (index, value) in enumerate(Constraints):
                if s<=value and value<t:
                  if (new_bitstring >> index) & 1 == 1:
                    flag = True
                  else:
                    new_bitstring = new_bitstring | (1 << index)

              if flag:
                continue

              # print lm_state
              # Update l, m, n, r, lm_state
              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob
              
              if s == current_lrstate.m+1:
                l = current_lrstate.l
                m = t -1
              else:
                if t == current_lrstate.l:
                  l = s
                  m = current_lrstate.m
                else:
                  l = s
                  m = t -1
              
              new_n = n + t - s

              r = t - 1
              
              new_lrstate = LR_state(l, m, r, lm_state, new_bitstring)
              # Is this still right?
              if new_n == len(f):
                logprob += lm.end(lm_state)
              




              new_hypothesis = hypothesis(logprob, new_lrstate, h, phrase, new_y)
              if new_lrstate not in stacks[new_n] or stacks[new_n][new_lrstate].logprob < logprob: # second case is recombination
                stacks[new_n][new_lrstate] = new_hypothesis
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

  final_state = 1 << len(Constraints)
  def isOK(bitstring):
    return bitstring == final_state -1
  flag = False
  for h in sorted(stacks[-1].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
    if isOK(h.LR_state.bitstring):
      winner = h
      flag = True
      break

  # if not flag:
  #   print "We got a problem!"
  # print winner
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


  # Test output
  # with open('test_output','w+') as test_output:
  #   print >> test_output, f[0:1]
  #   # print >> test_output, stacks[0]
  #   for h in sorted(stacks[0].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[1].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[2].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[3].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[4].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[5].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[6].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h
  #   print >> test_output, '==========================================================='
  #   for h in sorted(stacks[7].itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
  #     print >> test_output, h      
  # test_output.close()
  # sys.exit()
  # print extract_english(winner)
  ans = (winner.logprob, extract_english(winner), winner.y)
  return ans

def optimize(f, u, Constraints):
  violation_count = [0 for i in range(len(f))]
  previous = -1e10
  lstar = (-1, -1e10)
  converged = False
  

  def isoptimal(y):
    for yi in y:
      if yi != 1:
        return False
    return True

  def update_count(y):
    for (i,yi) in enumerate(y):
      if yi != 1:
        violation_count[i] += 1
  now = 0
  step = stepsize
  while not converged:
    result = beam(f, u, Constraints)
    y = result[2]
    # print 'English = ', result[1]
    # print 'u = ', u
    # print 'y = ', y
    if isoptimal(result[2]):
      return result
    #is converged?
    if lstar[1] < result[0]:
      if (result[0] - lstar[1])/(now - lstar[0]) < epsilon:
        converged = True
      else:
        lstar = (now, result[0])
    #Update u
    for (i, yi) in enumerate(y):
      u[i] -= step * (yi - 1)
    #Update stepsize
    if result[0] > previous:
      step = alpha_decrease_rate * step
    previous = result[0]
    now += 1
    if (now - lstar[0] > iteration_limit):
      break
    # print now, result[0], lstar
    # print '====================================================================='

  # print "!"
  K = iteration_limit
  G = Constraints_limit
  # Find the most violated element in y vector
  for iter_count in range(K):
    result = beam(f, u, Constraints)
    y = result[2]
    #update count
    if isoptimal(y):
      return result
    update_count(y)

    #Update u
    for (i, yi) in enumerate(y):
      u[i] -= step * (yi - 1)
    #Update stepsize
    if result[0] >previous:
      step = alpha_decrease_rate * step
    previous = result[0]

  new_Constraints = Constraints[:]

  #Add G constraints
  for g in range(G):
    maxcount = -1
    maxindex = -1
    for (i, vi) in enumerate(violation_count):
      if (i-1) not in new_Constraints and i not in new_Constraints and (i+1) not in new_Constraints:
        if maxcount < vi:
          maxcount = vi
          maxindex = i
    new_Constraints.append(maxindex)
    new_Constraints.sort()
  
  # print new_Constraints
  # print 'Constraint = ', new_Constraints    
  
  return optimize(f, u, new_Constraints)
  
def main():

  with open('lr_output_luckey150','w') as f:
    for fsen in french:
      # print 'French sentence =', fsen
      u = [0 for i in range(len(fsen))]
      result = optimize(fsen, u, [])

      # step = stepsize

      # for iter_count in range(iterations):
      #   result = beam(fsen, u, [])
      #   y = result[2]
      #   for (i, yi) in enumerate(y):
      #     u[i] -= step * (yi - 1)

      #   step = alpha_decrease_rate * step
        # print result
        # print u
      
      # output = fsen[0]
      # for i in range(1,len(fsen)):
      #   output = output + ' ' + fsen[i]
      # print output
      print result[1]
      print >>f, result[1]
      # break
  f.close()

if __name__ == '__main__':
  main()