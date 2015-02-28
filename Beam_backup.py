#!/usr/bin/env python
import optparse
import sys
import models
import math
from math import log
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

REORDERING_LIMITS = 5

def main():

	tm = models.TM(opts.tm, opts.k)
	lm = models.LM(opts.lm)
	french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

	# tm should translate unknown words as-is with probability 1
	for word in set(sum(french,())):
	  if (word,) not in tm:
	    tm[(word,)] = [models.phrase(word, 0.0)]

	sys.stderr.write("Decoding %s...\n" % (opts.input,))
	for f in french:
	  # The following code implements a monotone decoding
	  # algorithm (one that doesn't permute the target phrases).
	  # Hence all hypotheses in stacks[i] represent translations of 
	  # the first i words of the input sentence. You should generalize
	  # this so that they can represent translations of *any* i words.

	  # Including Swapping by introducing covered, a list of length len(f)
	  hypothesis = namedtuple("hypothesis", "covered, logprob, lm_state, predecessor, phrase")
	  initial_hypothesis = hypothesis([False for i in range(len(f))], 0.0, lm.begin(), None, None)
	  empty_cov = [False for i in range(len(f))]
	  print offset(empty_cov)
	  print getcov(3, len(f))
	  
	  stacks = [{} for _ in f] + [{}]
	  stacks = [[{} for tmp in range(2**len(f))] for _ in f] + [[{} for tmp in range(2**len(f))]]

	  stacks[0][0][lm.begin()] = initial_hypothesis

	  for i, stack in enumerate(stacks[:-1]):
	  	print stack
	  	for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:
	  		for j in xrange(i+1, len(f)+1):
	  			pass

	  def extract_english(h): 
	    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
	  print extract_english(winner)

	  if opts.verbose:
	    def extract_tm_logprob(h):
	      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
	    tm_logprob = extract_tm_logprob(winner)
	    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
	      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

def offset(cov):
  out = 0
  for bit in cov:
	out = (out << 1) | bit
  return out

def getcov(inp, length):
  out = []
  for i in range(length):
  	out.append(True if inp & 1 else False)
  	inp = inp >> 1
  return out
  	
if __name__ == '__main__':
	main()
