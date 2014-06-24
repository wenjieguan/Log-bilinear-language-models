import numpy as np
import heapq
import time
import h5py
import cPickle as pickle
from collections import namedtuple

'''
node: a inner node
word: a leaf node

count: the frequency of a word / the total frequency of left and right child of a inner node
index: ID for each node
left, right: pointing to its left and right child respectively
decisions: a list of 0/1 decisions from the root, 1 means visiting left child and 0 for right one
ancestors: a list of indices of all its ancestors
'''

node = namedtuple('node', ['count', 'index', 'left', 'right', 'decisions', 'ancestors'])
word = namedtuple('word', ['count', 'index', 'decisions', 'ancestors'])

class HLBL:
    def __init__(self, sentences = None, alpha = 0.025, min_alpha = 0.0237, dim = 100, context = 5, threshold = 5):
        '''
        vocab, for each word, stores its corresponding namedtuple word
        index2word records the index for each word
        total is the number of words in the training set
        alpha and min_alpha are the upper bound and lower bound for the learning rate
        dim is the dimension for each word embedding
        wordEm is a (vocabulary_size * dim) matrix, each row of which is a word embedding
        context is the size of history window
        words occur less than threshold times will be regarded as rare and will be mapped to a special token '<>'
        <_> is null padding, <s> denotes start of sentence, </s> means the end of sentence
        '''
        self.vocab = {}
        self.index2word = []
        self.total = -1
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.wordEm = self.contextW = self.nodeEm = None
        self.dim = dim
        self.context = context
        self.threshold = threshold
        self.l_pad = ['<_>'] * (self.context - 1) + ['<s>']
        self.r_pad = ['</s>']
        if sentences is not None:
            self.prepare_vocabulary(sentences)
            self.initialise()
            self.train(alpha = alpha, min_alpha = min_alpha, sentences)
            self.save()


    def save(self):
        print('Saving model...')
        pickle.dump(self.vocab, open('hlbl_tree.p', 'wb') )
        f = h5py.File('hlbl.hdf5', 'w')
        f.create_dataset('index2word', data = self.index2word)
        f.create_dataset('wordEm', data = self .wordEm)
        f.create_dataset('contextW', data = self.contextW)
        f.create_dataset('nodeEm', data = self.nodeEm)
        f.flush()
        f.close()
        print('Saved!')


    def load(self, name1 = 'hlbl_tree.p', name2 = 'hlbl.hdf5'):
        self.vocab = pickle.load(open(name1, 'rb') )
        f = h5py.File(name2, 'r')
        self.index2word = f['index2word'][:] #load all data in the memory
        self.wordEm = f['wordEm'][:]
        self.contextW = f['contextW'][:]
        self.nodeEm = f['nodeEm'][:]
        

    def top_10(self, given = 'computer'):
        '''
        this function prints out 10 most similar words for a given one, in terms of cosine similarity
        '''
        givenW = self.vocab.get(given.lower(), None)
        if givenW is None:
            print('Sorry the word {0} is not in the vocabulary!'.format(given) )
            return
        index = givenW.index
        if not hasattr(self, 'wordEm_norm'):
            self.wordEm_norm = (self.wordEm.T / np.linalg.norm(self.wordEm, axis=1) ).T
        givenEm = self.wordEm_norm[index]
        # the similarity between a word and itself is highest
        candidates = [(0, 0)] * 11
        for i, w in enumerate(self.wordEm_norm):
            score = np.dot(w, givenEm)
            heapq.heappush(candidates, (score, i) )
            heapq.heappop(candidates)
        results = []
        for i in range(10):
            results.append(heapq.heappop(candidates) )
        print('The top 10 words for {0} are:'.format(given) )
        for i in range(10):
            score, index = results[9-i]
            print('\t{0:2} {1:20} {2:<}'.format(i+1, self.index2word[index], score) )
    

    def initialise(self):
        print('Initialising weights...')
        self.nodeEm = (np.random.rand(len(self.vocab) - 1, self.dim) - 0.5) / self.dim 
        self.contextW = (np.random.rand(self.context, self.dim) - 0.5) / self.dim
        self.wordEm = (np.random.rand(len(self.vocab), self.dim) - 0.5) / self.dim
        

    def prepare_vocabulary(self, sentences):
        print('Building vocabulary...')
        total = 0
        vocab = {'<s>':0, '</s>':0}
        # each sentence is a list of strings, assume there is no empty sentence
        for sen_no, sentence in enumerate(sentences):
            vocab['<s>'] += 1
            vocab['</s>'] += 1
            for w in sentence:
                total += 1
                count = vocab.get(w, 0) + 1
                vocab[w] = count
        print('Visited {0} words and {1} sentences from the corpus'.format(total, sen_no + 1) )
        print('There are in total {0} different words'.format(len(vocab) ) )
        self.total = total

        # delete rare words and assign index to each word remaining
        self.vocab = {}
        self.index2word = []
        count_rare = 0
        for w, count in vocab.iteritems():
            if count >= self.threshold:
                index = len(self.vocab)
                self.vocab[w] = word(count, index, [], [])
                self.index2word.append(w)
            else:
                count_rare += 1
        index = len(self.vocab)
        self.vocab['<>'] = word(count_rare, index, [], [])
        self.index2word.append('<>')
        print('\nThe size of vocabulary is: {0}, with threshold being {1}\n'.format(len(self.vocab), self.threshold) )
        self.build_tree()


    def build_tree(self):
        print('Building a huffman tree...')
        heap = list(self.vocab.itervalues() )
        heapq.heapify(heap)
        # Since huffman tree is a full binary tree, there are len(self.vocab) - 1 inner nodes;
        for i in xrange(len(self.vocab), 2*len(self.vocab) - 1):
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = node(left.count + right.count, i, left, right, [], [])
            heapq.heappush(heap, parent)

        assert len(heap) == 1
        # breadth first traversal, traverse all inner nodes
        stack = [heap[0] ]
        while stack:
            parent = stack.pop()
            left = parent.left
            right = parent.right
            decisions = parent.decisions
            left.decisions.extend(decisions + [1])
            right.decisions.extend(decisions + [0])
            self_index = parent.index - len(self.vocab)
            ancestors = parent.ancestors + [self_index]
            left.ancestors.extend(ancestors)
            right.ancestors.extend(ancestors)
            # inner nodes have index greater than the size of the vocabulary
            if left.index >= len(self.vocab):
                stack.append(left)
            if right.index >= len(self.vocab):
                stack.append(right)


    def train(self, sentences, alpha = 0.025, min_alpha = 0.0237):
        print('Start training...')
        self.alpha = alpha
        self.min_alpha = min_alpha
        count = 0
        start = time.time()
        last_elapsed = 0
        # all OOV words will be mapped to RARE
        RARE = self.vocab['<>']
        for sentence in sentences:
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence) ):
                count += 1
                alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
                contextInd = [self.vocab.get(w, RARE).index for w in sentence[pos - self.context : pos] if w != '<_>']
                contextEm = self.wordEm[contextInd]
                contextW = self.contextW if len(contextInd) == self.context else self.contextW[-len(contextInd) : ]
                # r_hat is the prediction of word embedding for the next word
                r_hat = np.sum(contextEm * contextW, axis = 0)
                w = self.vocab.get(sentence[pos], RARE)
                ancestors = self.nodeEm[w.ancestors]
                # f is a list of probabilities of visiting left child for each ancestor of word w
                f = 1.0 / (1.0 + np.exp(-np.dot(r_hat, ancestors.T) ) )
                delta = -(np.asarray(w.decisions) - f)
                delta_q = np.outer(delta, r_hat)
                self.nodeEm[w.ancestors] -= (delta_q + 1e-5 * ancestors) * alpha
                delta_r_hat = np.dot(delta, ancestors)
                delta_r = contextW * delta_r_hat
                delta_c = contextEm * delta_r_hat
                self.contextW[-len(contextInd) : ] -= (delta_c + 1e-5 * contextW) * alpha
                self.wordEm[contextInd] -= (delta_r + 1e-4 * contextEm) * alpha
                elapsed = time.time() - start
                if elapsed - last_elapsed > 1:
                    print('visited {0} words, with {1:.2f} KWs/s, alpha: {2}.'.format(count, (count / elapsed) / 1000, alpha) )
                    last_elapsed = elapsed
        print('Training is finished!')


    def perplexity(self, sentences):
        print('Calculating perplexity...')
        RARE = self.vocab['<>']
        # _no_eos means no end of sentence tag </s>
        count_no_eos = count = 0
        logProbs_no_eos = logProbs = 0
        for sentence in sentences:
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence) ):
                count += 1
                count_no_eos += 1
                contextInd = [self.vocab.get(w, RARE).index for w in sentence[pos - self.context : pos] if w != '<_>']
                contextEm = self.wordEm[contextInd]
                contextW = self.contextW if len(contextInd) == self.context else self.contextW[-len(contextInd) : ]
                r_hat = np.sum(contextEm * contextW, axis = 0) # element wise multiplication
                w = self.vocab.get(sentence[pos], RARE)
                ancestors = self.nodeEm[w.ancestors]
                decisions = np.asarray(w.decisions)
                decisions = decisions - (decisions == 0)
                f = 1.0 / (1.0 + np.exp(-np.dot(r_hat, ancestors.T) * decisions) )
                prob = 1
                for i in f:
                    prob *= i
                res = np.log(prob)
                logProbs += res
                logProbs_no_eos += res
            logProbs_no_eos -= res
            count_no_eos -= 1
            # print results after each sentence
            ppl = np.exp(-logProbs / count)
            print('count: {0}'.format(count) )
            print('The perplexity is {0}'.format(ppl) )
        # the following displays the final perplexity
        ppl = np.exp(-logProbs / count)
        ppl_no_eos = np.exp(-logProbs_no_eos / count_no_eos)
        print('The perplexity with eos is {0}'.format(ppl) )
        print('               without eos is {0}'.format(ppl_no_eos) )
