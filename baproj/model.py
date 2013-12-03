"""
	Bachelor AI project 2009. Andreas van Cranenburgh 0440949.
	Two-word stage dialogue simulator using a corpus of exemplars.

	Interactive usage:
	$ python model.py

	Other uses:
		- Re-enact dialogue in 01.cha (should be a text file with one utterance
		  per line, with each line preceeded by *MOT or *CHI, as in the CHILDES
		  corpus):

		  $ python
#	  	  >>> import model
#		  >>> model.evaluate(open('01.cha').read())
		  ...

		- Talk to self for 25 utterances:

#		  >>> model.selftalk(25)
		  ...

		- Demonstrate some sample utterances that work (but not from corpus):

#		  >>> model.test()
		  ...
"""

# (before continuing set tabstops to 4)
# update documentation by doing "pydoc -w model", then open "model.html"
# alternatively, use "epydoc model".

#random factoid:
#You can reference a list comprehension as it is being built by the symbol
#'_[1]'. For example, the following function unique-ifies a list of elements
#without changing their order by referencing its list comprehension.
#
#def unique(my_list):
#    return [x for x in my_list if x not in locals()['_[1]']]

try:
	from free_will import choice
except ImportError:
	from random import choice
from string import uppercase, lowercase
from collections import defaultdict
from pprint import pprint
#stands for pretty-print; library implementors have some strange standards
import math

def main():
	""" Interactive textual user interface. """
	print "Child dialogue simulator.",
	print "Enter parent utterance (or `quit' to quit)."
	print
	print "Utterances in corpus: "
	print getexemplars().keys()
	print
	lexicon = inferlexicon(getexemplars(), True)
	print 'Lexicon (distilled from corpus): ', lexicon
	#test(getexemplars(), lexicon)
	d = dialogue()
	d.send(None)
	while True:
		print "Parent: ",
		utt = raw_input()
		if utt == 'quit':
			break
		reply = d.send(utt)
		print "Child: ", reply

def dbg(*m):
	""" Print debug output (also see nodbg).
		Takes a variable number of arguments of any type.

		>>> dbg('choices', range(3))
		choices [0, 1, 2]
	"""
	print (' '.join(list(str(a).expandtabs() for a in m)))
def nodbg(*m): pass

def dialogue(exemplars={}, constructions={}, lexicon={}, debug=dbg):
	""" Co-routine for dialogue. Use send(utterance) to receive a reply.
		Initial input: exemplars
		Further input: one utterance at a time
		Output: one reply at a time"""
	if not exemplars:
		exemplars = getexemplars()
	if not lexicon:
		lexicon = inferlexicon(exemplars, False)
	if not constructions:
		constructions = inferconstructions(exemplars, lexicon)
	discourse, topic, reactionutt = [('','')], '', ''
	while True:
		utt = yield reactionutt
		newmeaning = interpret(utt, topic, exemplars, lexicon)
		if 'reinforcement' in newmeaning and len(discourse) > 1:
			reinforce(meaning, reaction, reactionutt, discourse, exemplars)
		meaning = newmeaning
		debug('	interpretation:', meaning)

		#todo: detect correction feedback, store. DISABLED
		if False and utt not in exemplars and meaning:
			if meaning == lexiconcheck(utt, meaning, lexicon):
				debug('added utterance to exemplars')
				exemplars[utt] = meaning
				constructions = inferconstructions(exemplars, lexicon)
			else:
				debug('generation check failed, not added to exemplars')

		reaction = response(meaning, exemplars)
		debug('	reaction:', reaction)
		reactionutt = express2(reaction, exemplars, constructions, lexicon)
		discourse.append((utt, meaning))
		discourse.append((reactionutt, reaction))
		# if the current topic is not present in the parent's utterance,
		# see if there is a new one. otherwise there is topic continuity
		if (topic not in meaning and topic) or not topic:
			topic = findtopic(discourse) or ''

def selftalk(u=25, e={}, debug=dbg):
	""" Talk to oneself, picking a random utterance from the exemplars when
		repetition is detected.
		Input: number of utterances to generate, exemplars
		Output: transcript of conversation
	"""
	if not e:
		e = getexemplars()
	a = choice(e.keys())
	speaking, listening = '*MOT', '*CHI'
	said = set([a])
	c = []

	d = dialogue(e)
	d.send(None)

	for x in range(u):
		c.append('%s: %s' % (speaking, a))
		debug(c[-1])
		speaking, listening = listening, speaking
		r = d.send(a)
		if a == r and speaking == '*MOT':
			debug('picking random utterance')
			# because sets are not subscriptable, it is not possible to apply
			# choice to sets, perhaps this indicates that the axiom of choice
			# is untenable after all?!
			a = choice(list(set(e.keys()) - said))
			said.add(a)
		else:
			a = r
	print '\n', '\n'.join(c)

def evaluate(conv, exemplars={}, n=1):
	""" When presented with a corpus fragment, feed parent utterances to model 
		and generate model responses in place of child utterances.
		n = number of iterations to perform
		Input: string with transcript of corpus
		Output: same transcript with replies from model instead of child
	"""
	# idea: only continue for loop if model answer matches child answer?
	# this allows reinforcement to select correct answers
	if not exemplars:
		exemplars = getexemplars()
	# start & initialize co-routine
	d = dialogue(exemplars)
	d.send(None)
	for a in range(n):
		modelconv = []
		for b in conv.split('\n'):
			try:
				who, utt = b.split(':', 1)
			except ValueError:
				# ignore unparsable lines:
				continue
			if 'MOT' in who:
				print b
				ans = '*CHI:  ' + d.send(utt)
				print ans
				for x in (b, ans):
					modelconv.append(x)
	print
	return '\n'.join((str(a) for a in modelconv))

def reinforce(meaning, reaction, reactionutt, discourse, exemplars):
	""" Strengthen connection between last two utterances by adding 
		a random identifier and adding or updating the resulting exemplar.
		Input: two utterances with meanings
		Side-effect: updated or added exemplars
	"""
	randomid = "".join(choice(lowercase) for a in range(5))
	if discourse[-1][0] in exemplars:
		print 'updated',
	else:
		print 'added',
	exemplars[discourse[-1][0]] = meaning + ' ' + randomid
	print 'reinforcement:', exemplars[discourse[-1][0]]
	if reactionutt in exemplars:
		print 'updated',
	else:
		print 'added',
	print 'with',
	exemplars[reactionutt] = reaction + ' ' + randomid
	print exemplars[reactionutt]

def findtopic(discourse, debug=dbg):
	""" Look for a recurring element in the last two utterances,
		if found, this becomes the new topic.
		Input: discourse (list of utterances and meanings)
		Output: clause, or None
	"""
	# prefer second clause, then rest.
	def minprefertwo(seq):
		seq = list(seq)
		try:
			return ((a,b) for a,b in seq if a==2).next()
		except StopIteration:
			return min(seq)

	# take intersection of last utterance with second to last utterance
	# (ignoring operators)
	m1, m2 = discourse[-1][1].split(), discourse[-2][1].split()
	common = intersect(m1[1:], m2)
	if common:
		# something went wrong with keeping this simple; apologies.
		topic = minprefertwo(
			(minprefertwo(
				((m1.index(a), a),
				(m2.index(a), a))) for a in common))[1]
		debug('	topic:', topic)
		return topic

def interpret(utterance, topic, exemplars, lexicon, debug=dbg):
	""" Return semantic representation for linguistic utterance.
		This function is in charge of backtracking over initial exemplars
		and picking the best result, the rest is done by interpretwith()
		Input: utterance
		Output: meaning representation
	"""
	def match(a, b):
		return len(intersect(a, b.split()))

	utterance = utterance.replace('[= ', '[=')
	words = utterance.split()
	initial = [(match(words, a), a, b) for a,b in exemplars.items()]

	# backtrack over all exemplars with any overlap at all:
	#initial = [a[1:] for a in sorted(initial, reverse=True) if a[0]]

	#only backtrack over exemplars with highest overlap (much faster):
	n = max(initial)[0]
	if n:
		initial = [a[1:] for a in initial if a[0] == n]
	else:
		# if none of the words are in corpus we can just forget about it:
		return ''

	i = []
	for exutt, exmeaning in initial:
		#print 'trying: %s' % a[1]
		w = [b for b in words if b not in exutt]
		n, meaning = interpretwith(w, exmeaning, exemplars, lexicon, nodbg)
		if meaning:
			# scoring (heuristics):
			#  1: distance to generation check (lexicon)
			#  2: nr of exemplars, 
			#  3: inverse of meaning length
			#  4: word lengths of matches in initial exemplar 
			#		(longer words are less likely to be 
			#		unimportant noise/function words).
			#  5: nr of substitutions, 
			# TODO heuristic: amount of pairwise overlap between exemplars used
			# this would require book-keeping during recursion..
			i.append((
				n,
				edit_dist(lexiconcheck(utterance, meaning, lexicon), meaning),
				1.0 / len(tokens(meaning)),
				#edit_dist(exmeaning, meaning),
				#1.0 / sum(len(a) for a in (b for b in words if b in exutt)),
				#end of heuristics. initial exemplar:
				(exutt, exmeaning)))
	#to debug choice of initial exemplar, enable this to see all candidates:
	#pprint(sorted(i))
	if i:
		# find lowest score:
		a = min(i)[-1]
		debug('initial exemplar:')
		debug('\t', a)

		# try to merge topic (but don't care if it fails, avoid substition):
		c = conciliate(topic, a[1], varclauses(b))
		if c:
			b = c
		else:
			b = a[1]

		# do interpretation again, this time with debug output:
		w = [c for c in words if c not in a[0].split()]
		debug('words left:', repr(' '.join(w)))
		if w:
			return interpretwith(w, b, exemplars, lexicon, debug)[1]
		#else: no words left to interpret
		return b
	else:
		return ''

def lexiconcheck(utt, meaning, lex):
	# generation check, filter interpretation 
	# by that which is present according to lexicon.
	# disabled because the lexicon is too incomplete for this.
	def elim(a):
		return a.replace('[=%s]' % a[2:-1], a[2:-1]).strip().lower()
	revlex = revdict(lex)
	# map utterance to lexicon items
	uttlex =  " ".join(lex[elim(b)] for b in utt.split() if elim(b) in lex)
	# reconstruct interpretation according to own knowledge
	# select those clauses that occur in reverse lexicon
	# or in utterance --> lexicon mapping
	# don't be picky with speech acts
	def check(a):
		return a not in revlex or a in uttlex or a[-1] == ':'
	return " ".join((a for a in meaning.split() if check(a)))


def interpretwith(words, partialmeaning, exemplars, lexicon, debug=dbg):
	""" Interpretation helper function called by interpret(), work out meaning
		of remaining words by stitching together the best matching fragments.
	"""
	l = len(words)

	# lexicon check is too permissive
	#if words[0] not in lexicon:
	# skip over words not in corpus
	if words and words[0][0] != '[':
		if words[0] not in ' '.join(exemplars.keys()).split():
			debug("skipping", repr(words[0]))
			words = words[1:]

	# handle demonstratives
	# eg. "this [=ball]" should add the meaning of "ball"
	if partialmeaning and [w for w in words if '[' in w]:
		word = [w for w in words if '[' in w][0][2:-1].strip().lower()
		# TODO multi-word, eg. ice cream ...
		if word in lexicon:
			pm = conciliate(lexicon[word], partialmeaning, None, debug)
			if pm:
				partialmeaning = pm
				debug("demonstrative dereferenced:", partialmeaning)
			else:
				# fail!
				return 0, ''
				partialmeaning[0] += ' ' + lexicon[word]
				debug("appended demonstrative:", partialmeaning)
		words = [a for a in words if a[0] != '[']

	for sub in substr_iter(words):
		substring = " ".join(sub)
		for b in exemplars:
			# check whether the substring occurs and match whole words only
			if substring in b and set(sub).issubset(b.split()):
				pm = partialmeaning
				# turn on substition for known words
				subst = [lexicon[word] for word in sub if word in lexicon]
				pm = conciliate(exemplars[b], partialmeaning, subst + [''], debug)
				if pm:
					# remove all words in exemplar from queue
					#if c in b.split()
					matches = [c for c in words if c in sub]
					words = [c for c in words if c not in sub]
					debug('	', repr(' '.join(matches)), 'in', repr(b))
					debug('	and', repr(exemplars[b]))
					debug('	matches', repr(pm))
					#recurse:
					m, p = interpretwith(words, pm, exemplars, lexicon, debug)
					return 1 + m, p
	# no match yet?
	if words:
		return 0, '' #partialmeaning[0]
	return 0, partialmeaning

def response(meaning, exemplars, debug=dbg):
	""" Transform a meaning into a response using adjacency pairs of speech
		acts.
		Input: meaning representation
		Output: meaning representation
	"""
	def vars(m):
		return len([a for a in tokens(m) if var(a)])
	if 'imperative' in meaning:
		# refusal doesn't occur in corpus, so disabled:
		#op = choice(('acknowledgement: ', 'refusal: '))
		return 'acknowledgement ' + meaning.split(':')[1]
	elif 'ynquestion' in meaning:
		op = choice(('agreement', 'denial'))
		return op
	elif 'whquestion' in meaning:
		if not ':' in meaning:
			return ''
		m = 'assertion:' + meaning.split(':', 1)[1]
		#d = [a for a in exemplars.values() if conciliate(a, [m])]
		# edit distance on clauses, plus a penalty for variables
		# (this penalty implements a bias for definite information):
		def score(m, a):
			return edit_dist(tokens(m), tokens(a)) + 0.5*vars(a)
		d = [(score(m, a), a) for a in exemplars.values()]
		#print sorted(d)
		if min(d)[0] < 1:
			#choices = [a[1] for a in d if a[0] == min(d)[0]]
			#max variance to be eligible is twice the minimal distance
			choices = [a[1] for a in d if a[0] < 1]
			debug("possible reactions:", sorted(choices))
			if choices:
				r = choice(choices)
			else:
				return ''
			m = conciliate(r, m, varclauses(m))
			if m:
				n = unifies(r, m)
				if n:
					if len(choices) > 1:
						debug('choice:', m)
					return n
				if len(choices) > 1:
					debug('choice:', m)
				return m
		else:
			return m #''
	#here we can decide to take spontaneous initiative (eg. "more juice!")
	elif 'assertion' in meaning:
		#re-phrase as two word utterance
		return meaning
		#return 'acknowledgement'

	# here we could express confusion
	return ''

def express(meaning, exemplars, lexicon):
	""" Express `meaning' by returning the most similar exemplar. """
	if meaning in exemplars.values():
		return choice(revdict(exemplars)[meaning])
	else:
		#find approximation
		# edit distance on clauses:
		def dist(m, a):
			return edit_dist(m.split(), a.split())
		d = [(dist(meaning, b), a) for a, b in exemplars.items()]
		if min(d)[0] < len(meaning):
			return choice([a for a in d if a[0] == min(d)[0]])[1]
		else:
			return '0'

def expressmulti(meaning, exemplars, constructions, lexicon):
	""" Express `meaning' by returning a matching exemplar or construction """
	# exemplar filtered according to constructions
	def check(a):
		return a in ' '.join(constructions.keys())
	if meaning in exemplars:
		return ' '.join(a for a in exemplars[meaning] if check(a))
	# try constructions
	if meaning in constructions.values():
		return choice(revdict(constructions)[meaning])
	# this sounds a little poetic:
	if any(meaning in clauses for clauses in constructions.values()):
		return choice([utt for utt, clauses in constructions.items()
			if meaning in clauses])
	if any(clauses in meaning for clauses in constructions.values()):
		return choice([utt for utt, clauses in constructions.items()
			if clauses in meaning])
	# fall back on two word stage
	return express2(meaning, exemplars, constructions, lexicon)

def express2(meaning, exemplars, constructions, lexicon, debug=dbg):
	""" Transform a meaning into a two word utterance using exemplars,
		constructions or the lexicon. Filter result according to lexical
		knowledge.
		Input: meaning representation
		Output: one or two word utterance
	"""
	# filter out words not in lexicon
	def reduce(utt):
		r=' '.join(a for a in utt.split() if a.lower() in lexicon)
		if r == None:
			return ''
		return r

	# first try to find a matching exemplar
	if meaning in exemplars.values():
		utt = choice(revdict(exemplars)[meaning])
		debug('reduced:', utt)
		return reduce(utt)

	# try constructions
	if meaning in constructions.values():
		utt = choice(revdict(constructions)[meaning])
		debug('reduced:', utt)
		return reduce(utt)
	# this sounds a little poetic:
	if any(meaning in clauses for clauses in constructions.values()):
		utt = choice([utt for utt, clauses in constructions.items()
			if meaning in clauses])
		debug('reduced:', utt)
		return reduce(utt)

	# if that fails:
	# use lexicon to fetch possible words, pick two and string together
	revlex = revdict(lexicon)
	if ':' in meaning:
		# might want to do something with operators as well?
		meaning = meaning.split()
	else:
		# operator without further clauses
		if meaning in revlex:
			return revlex[meaning]
		else:
			return ''
	if not arg(meaning[1]) or arg(meaning[1])[1] not in uppercase:
		clause = meaning[1]
	else:
		return ''
	debug("trying to express:", clause)

	# express first clause if in lexicon:
	for a in meaning[1:]:
		if a in revlex:
			return choice(revlex[a])
	if clause in revlex:
		return revlex[clause].pop()

	# compose from lexicon using predicate and argument
	topic = arg(meaning[1])
	if topic and topic[0] in uppercase:
		if len(meaning) > 2:
			topic = pred(meaning[2])
		else:
			topic = meaning[0]
	comment = pred(meaning[1])

	candidates = [a for a,b in lexicon.items() if comment in b]
	if candidates:
		utt = choice([a for a,b in lexicon.items() if comment in b])
		debug('comment:', comment, 'in', utt)
	else:
		utt = ''

	# decide whether to express topic. TODO: think of non-random way to decide
	if choice([0, 1]):
		return utt

	candidates = [a for a,b in lexicon.items() if topic in b and utt not in a]
	if candidates:
		a = choice(candidates)
		utt = a + ' ' + utt
		debug('topic:', topic, 'in', utt)
	else:
		debug('topic not expressible?', topic)
	return utt

def inferconstructions(exemplars, lexicon, constructions={}, debug=dbg):
	""" Build corpus of constructions from exemplars and lexicon.
		Input: exemplars & lexicon.
		Output: constructions, dictionary of pairings between substrings
		and clauses. """
	debug('distilling constructions')
	# take substr_iter of all sentences, chain these
	scores = defaultdict(int)
	n = len(constructions)
	for utt, meaning in exemplars.items():
		for a in substr_iter(utt.split()):
			# constructions consist of two or more words (lexicon otherwise):
			if len(a) == 1:
				break
			# filter out non-alphabetic characters
			words = ' '.join(b.lower() for b in a if all(c.lower() in lowercase for c in b))
			if ' ' not in words:
				continue
			# increment count in constructions if this one is unknown
			if not any(words in b for b in constructions):
				score = len([c for c in exemplars if words in c])
				if score:
					scores[words] = score
	# shouldn't this be a priority queue instead?
	# iterate over counts, take highest counts and pick longest constructions
	for score, form in sorted(revdict(scores).items()):
		form = form[0]
		if form not in scores:
			# skip this construction if it has been removed meanwhile
			continue
		for a in scores.keys():
			# remove shorter constructions
			if a in form and form != a:
				scores.pop(a)
			# longer constructions are OK
			# however: this means that the shorter one should abstract!
			# use lexicon to check presence of words.

		# replace score with meaning, intersect like inferlexicon
		meaning = findmeaning(form, exemplars, lexicon, dbg)
		if meaning:
			debug('construction (score %d)' % score, repr(form), repr(meaning))
			constructions[form] = meaning
		else:
			ms = [meaning for utt, meaning in exemplars.items() if form in utt]
			score = '(score %d)' % score
			candidate = 'not found in candidates %s', str(ms)
			debug('meaning of construction', repr(form), score, candidate)
		# if not found the next pass might find it by elimination

	#print sorted(revdict(scores).items())
	# base case (no change)
	if n == len(constructions):
		return constructions
	# recurse 
	return inferconstructions(exemplars, lexicon)

def findmeaning(form, exemplars, lexicon, debug=dbg):
	""" Given the words in a construction, find the most common meaning
		associated with it in the corpus of exemplars. """
	ms = [meaning for utt, meaning in exemplars.items() if form in utt]
	#print ms
	#ehh, why this whole intersection business?
	revlex = revdict(lexicon)
	if len(ms) > 1:
		m = [a.split() for a in ms]
		m = reduce(intersect, m[1:], m[0])
		#print m
		if m:
			n = []
			# if some word for a clause is not present in construction, 
			# abstract its meaning
			def abstract(a):
				if a in revlex and any(b in form for b in revlex[a]):
					return a
				else:
					if a[0] == '(':
						return '(X)'
				#abstract operators
				return a #''
			return ' '.join(abstract(a) for a in m)
		else:
			# prefer most frequent sub-interpretation (allow non-contiguous?)
			# --> combinations instead
			return max((ms.count(a), a) for a in ms)[1]
			scores = defaultdict(int)
			for meaning in ms:
				for a in substr_iter(meaning.split()):
					m = ' '.join(a)
					# increment count in constructions if this one is unknown
					scores[m] += 1
			return max(revdict(scores))[1][0]
		#elif len(m) > 1:
		#	# if multiple possibilities arise, prefer least frequent predicate
		#	def freq(a):
		#		return ' '.join(exemplars.values()).count(pred(a))
		#	return ' '.join(min((freq(a), a) for a in m)[1])
	if ms:
		return ms.pop()

# a mutable default argument is essentially a static variable (yay for arcana!)
def inferlexicon(exemplars, verbose=False, lexicon={}, debug=dbg):
	""" Infer lexicon from corpus of exemplars.
		Input: exemplars.
		Output: lexicon, dictionary of word-clause pairings. """
	exlcase = dict((a.lower(), b) for a,b in exemplars.items())
	n = len(lexicon)
	# look for word indices linking meanings to words:
	for utt,m in exemplars.items():
		if '[' in m:
			n = int(m[m.index('[')+1 : m.index(']')])
			clause = m[:m.index('[')].split()[-1]
			clause += m[m.index(']')+1:].split()[0]
			word = utt.split()[n].lower()
			exemplars[utt] = m[:m.index('[')] + m[m.index(']')+1:]
			if word not in lexicon:
				lexicon[word] = clause
	for word in set(" ".join(exemplars.keys()).split()) - set(lexicon.keys()):
		word = word.lower()
		utterances = [a.split() for a in exlcase.keys() if word in a.split()]
		meanings = [b.split() for a,b in exlcase.items() if word in a.split()]

		# reduce meanings by matching words with meanings:
		for a in list(reduce(intersect, meanings, meanings[0])): #[::-1]:
			if word in a:
				if len(a) == 1:
					lexicon[word] = a
				else:
					# select least frequent predicate (assumption is that 
					# this predicate is most specific to this word)
					def freq(b):
						return ' '.join(exemplars.values()).count(pred(b))
					candidates = ((freq(b), b) for b in a.split())
					lexicon[word] = min(candidates)[1]
				break
		# if this didn't succeed and the set of exemplars which contain the
		# word reduces to a single meaning, use this:
		#DISABLED
		if False and word not in lexicon:
			a = reduce(intersect, utterances, utterances[0])
			if len(a - set(lexicon.keys())) == 1:
				lexicon[word] = reduce(intersect, meanings, meanings[0])
				if len(lexicon[word]) == 0:
					lexicon[word] = ''
				else:
					lexicon[word] = lexicon[word].pop()
		# a single word utterance carrying a single meaning:
		if len(utterances) == len(meanings) == len(utterances[0]):
			if len(utterances) == len(meanings[0]) == 1:
				lexicon[word] = meanings[0][0]
	if verbose:
		notfound = set(' '.join(exlcase.keys()).split()) - set(lexicon.keys())
		debug('lexicon: not found:', notfound, '\n')
	# recurse as long as new meanings are inferred:
	if len(lexicon) == n:
		return inferlexicon(exemplars, verbose)
	return lexicon

def conciliate(meaning, partialmeaning, subst=None, debug=dbg):
	""" Test whether meaning can be made comptabible with partialmeaning,
		by looking for a family resemblance with partialmeanings, 
		if found, perform a substitution if necessary. Returns empty string on failure.

		>>> conciliate('question: animal(bunny) do(X)', 'assertion: point(dog) animal(dog)')
		substituted (bunny) for (dog)
		'assertion: point(bunny) animal(bunny)'
		>>> conciliate('assertion: do(hop) animal(bunny)', 'assertion: animal(bunny) do(X)')
		instantiated (X) with (hop)
		'assertion: animal(bunny) do(hop)'
		>>> conciliate('assertion: do(hop) animal(bunny)', 'assertion: point(bunny)')
		''
		"""
	pm = partialmeaning
	if subst == None:
		# None as in no restrictions
		subst = pm.split()
	if pm == '':
		return meaning
		partialmeaning = meaning
		return True
	success = False
	for part in meaning.split():
		if pred(part)+'(' in pm:
			# if there is an argument and it's not variable:
			# instantiate / substitute
			if arg(part) and not (var(arg(part)) or arg(part) in pm):
				oldarg = tokens(pm)[tokens(pm).index(pred(part)) + 1]
				# instantiate or substitute if allowed
				if True in [pred(part) in a for a in subst]:
					if oldarg[1] in uppercase:
						debug("instantiated", oldarg, "with", arg(part))
						partialmeaning = pm = pm.replace(oldarg, arg(part))
					else:
						debug("substituted", arg(part), "for", oldarg)
						partialmeaning = pm = pm.replace(oldarg, arg(part))
			success = True
			# disable this to match as much as possible:
			# return success
		#probe for variable predicate in first argument
		if pred(part) in uppercase:
			return partialmeaning
			return True
		# probe for variable predicate in second argument
		vars = [pred(a) for a in pm.split() if pred(a) in uppercase]
		# instantiate variable predicate if part is not an operator 
		# and this predicate is not already present
		if vars and ':' not in part and pred(part) not in pm:
			partialmeaning = pm.replace(vars[0], pred(part))
			debug("instantiated variable predicate with", pred(part))
			# recurse to replace further vars
			return conciliate(meaning, partialmeaning, subst, debug)
		"""
		#this code would allow to substitute predicates if arguments match:
		elif arg(part) in pm and arg(part):
			ar = arg(part)
			oldpred = pm[pm.index(ar):pm.index(ar)+len(ar)]
			print ar
			print oldpred
			exit()
			print "substituted", pred(part), "for", oldpred
			partialmeaning[0] = pm.replace(oldpred, pred(part))
			return True
		"""
	if success:
		return partialmeaning
	return ''
	return success #False

def unifies(meaning, partialmeaning, debug=dbg):
	""" succeed if everything in "partialmeaning" is compatible with "meaning",
		substituting along the way. Similar to conciliate() but operates on the
		whole meaning instead of looking at individual clauses, and does not
		perform substitution, merely instantiation.

		>>> unifies('assertion: do(hop) animal(bunny)', 'assertion: do(X) animal(bunny)')
		substituted (hop) for (X)
		'assertion: do(hop) animal(bunny)'
		>>> unifies('assertion: do(hop) animal(bunny)', 'assertion: animal(bunny) do(X)')
		''

	"""
	pm = partialmeaning
	if pm == '':
		return meaning
		partialmeaning = meaning
		return True

	for part1, part2 in zip(pm.split(), meaning.split()):
		if part1 == part2:
			continue
		elif var(arg(part1)):
			newarg = tokens(meaning)[tokens(meaning).index(pred(part2)) + 1]
			debug("substituted", newarg, "for", arg(part1))
			partialmeaning = pm = pm.replace(arg(part1), newarg)
		else:
			# meaning and partialmeaning are in conflict
			return '' #False
	return partialmeaning #True

def getexemplars():
	""" Obtain corpus, either by importing from module `exemplars,'
		or by falling back to a small sample corpus. """
	try:
		#self-referential import...
		from exemplars import getexemplars
	except ImportError:
		pass
	else:
		return getexemplars()
	#"""
	# sample corpus:
	return {'uh': 					'',
		'eh': 					'',
		'ah': 					'acknowledgement',
		'ok': 					'confirmation',
		'yes': 					'confirmation',
		'yeah': 				'confirmation',
		'no': 					'denial',
		'that\'s right !'		: 'reinforcement',
		'ball'					: 'assertion: point(ball) toy(ball)',
		'throw it to me'		: 'imperative: throw[0](X) toy(X)',
		'you kick the football !' : 'imperative: kick(football) toy(football)',
		'what is this ?'		: 'whquestion: point(X) Y(X)',
		#should be a yes/no question, but child responded with 'cow'. perhaps
		#assertion instead?
		'is that a cow ?'		: 'whquestion: point(cow) animal(cow)',
		'what does a bunny do ?': 	'whquestion: do(X) animal(bunny)',
		'what does a cow say ?'	: 'whquestion: do(X) animal(cow)',
		'what does a duckie say ?': 'whquestion: do(X) animal(duck[3])',
		'what animal does woof woof ?': 	'whquestion: animal(X) do(woof)',
		'what\'s a kitty say ?'	: 'whquestion: do(X) animal(cat[2])',
		'that\'s a kitty'		: 'assertion: point(cat) animal(cat[2])',
		'donkey'				: 'assertion: point(donkey) animal(donkey)',
		'that\'s a donkey'		: 'assertion: point(donkey) animal(donkey)',
		'meouw'					: 'assertion: do(meouw) animal(cat)',
		'quack'					: 'assertion: do(quack) animal(duck)',
		'moo'					: 'assertion: do(moo) animal(cow)',
		'bunny'					: 'assertion: animal(bunny) do(hop)',
		'dog'					: 'assertion: animal(dog) do(woof)',
		'hop'					: 'assertion: do(hop) animal(bunny)',
		'woof woof'				: 'assertion: do(woof) animal(dog)',
		'want some juice ?'		: 'ynquestion: want(juice) food(juice)',
		'chocolate'				: 'assertion: food(chocolate)',
		'what\'s inside ?' 		: 'whquestion: inside(X) toy(box)',
		#child guesses correctly
		'book' 					: 'assertion: inside(book) toy(box)',
		#child takes out book & mother confirms
		'book !' 				: 'assertion: point(book)'
		}

def test():
	""" Interpret some sample utterances. """
	test = [
		'what does a bunny do ?',
		'kitty do ?',  #ellipsis
		'throw the ball',
		'catch the ball',
		'where lives birdie ?',  #stumble on generalization
		"that's a ball",  #make topic
		'can you throw it ?',  #use topic
		'what does a duckie say ?',
		'what does this animal [=cow] do ?',
		'do you want to play with that [=ball] ?',
		'what does a ball say ?' # should fail
		# disabled: (chocolate / kick not in corpus)
		#'want some chocolate ?'
		#'kick the ball !',
		]
	#print evaluate('\n'.join('*MOT:  %s' % a for a in test))
	exemplars = getexemplars()
	lexicon = inferlexicon(exemplars)
	for a in test:
			print a
			b = interpret(a, '', exemplars, lexicon, nodbg)
			print 'interpretation:', b

def edit_dist(source, target):
	""" Edit distance of two sequences. Non-standard features are that all
		mutations have a cost of 1 (so as not to favor insertions or deletions
		over substitutions), as well as a decay based on index
		(mutations with higher indices weigh less). 
		
		>>> edit_dist('foo', 'bar')
		2.4890507991136213"""
	def substcost():
		if a == b:
			return 0
		# default is 2, but we should avoid a bias for insertion/deletions
		else: return 1
	#decay function that weighs higher positions lower
	def decay(n):
		# with a rate of 0.2 the third position in a sequence will
		# have a weight of around 0.5 (initial position weighs 1.0)
		return math.e ** (-0.2 * n)

	# initialize distance matrix
	# this looks better but doesn't work (tm)
	#distance = [[0] * (len(target) + 1)] * (len(source) + 1)
	distance = [[0] * (len(target) + 1) for b in range(len(source)+1)]
	for i, row in enumerate(distance):
		row[0] = i
	for j in range(len(distance[0])):
		distance[0][j] = j
	distance[0][0] = 0

	# populate distance matrix
	for i, a in enumerate(source):
		for j, b in enumerate(target):
			insc = distance[i][j+1] + decay(i)
			substc = distance[i][j] + decay(i) * substcost()
			delc = distance[i+1][j] + decay(i)
			distance[i+1][j+1] = min((substc, insc, delc))
	return distance[-1][-1]

def substr_iter(seq):
	""" Return all substrings of a sequence, in descending order of length.
		
		>>> list(substr_iter('abc'))
		['abc', 'ab', 'bc', 'a', 'b', 'c']"""
	for a in range(len(seq), -1, -1):
		for b in range(a, len(seq)):
			yield seq[b-a:b+1]

def revdict(d):
	""" Reverse dictionary, ie. swap values and keys; since a value may occur
		with multiple keys, return a list of all keys associated with a value.

		>>> revdict({0: 1, 1: 2})
		{1: [0], 2: [1]}"""
	# ignore multiple keys for a value:
	# return dict(a[::-1] for a in d.items())
	return dict((a, [b for b,c in d.items() if c==a ]) for a in d.values())

def varclauses(m):
	""" >>> varclauses(['foo(bar) zed(X)'])
		['zed(X)']"""
	return [a for a in m[0].split() if var(arg(a))]

def pred(token):
	""" >>> pred('foo(bar)')
		'foo'"""
	if '(' in token:
		return token[:token.index('(')]
	else: return token

def arg(token):
	"""	>>> arg('foo(bar)')
		'(bar)'"""
	if '(' in token:
		return token[token.index('('):]
	else: return ''

def var(arg):
	""" Test whether an argument string is variable:
		
		>>> var('(bar)')
		False
		>>> var('(X)')
		True
	"""
	return arg and (arg[0] in uppercase or arg[1] in uppercase)

def tokens(m):
	""" Turn meaning into a list of operator, predicates and arguments.
		
		>>> tokens('whquestion: do(X) animal(bunny)')
		['whquestion:', 'do', '(X)', 'animal', '(bunny)']
	"""
	return m.replace('(', ' (').split()

def intersect(a,b):
	""" >>> intersect([1,2,3], [2,3])
		set([2, 3])"""
	return set(a).intersection(set(b))

# this trick runs main when the script is executed, not when imported:
if __name__ == '__main__':
	import doctest
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = doctest.testmod(verbose=False,
		optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
	if attempted and not fail:
		print "%d doctests succeeded!" % attempted
	main()
