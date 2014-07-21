import numpy as np
import time
import h5py

class LBL:
    def __init__(self, sentences = None, alpha = 0.025, min_alpha = 0.0237, dim = 100, context = 5, threshold = 3, batches = 1000):
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
        self.frequencies = []
        self.total = -1
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.wordEm = self.contextW = self.biases = None
        self.dim = dim
        self.context = context
        self.threshold = threshold
        self.l_pad = ['<_>'] * (self.context - 1) + ['<s>']
        self.r_pad = ['</s>']
        if sentences is not None:
            self.prepare_vocabulary(sentences)
            self.initialise()
            self.train(sentences, alpha = alpha, min_alpha = min_alpha, batches = batches)
            self.save()
            

    def save(self):
        print('Saving model...')
        f = h5py.File('lbl.hdf5', 'w')
        f.create_dataset('index2word', data = self.index2word)
        f.create_dataset('wordEm', data = self .wordEm)
        f.create_dataset('contextW', data = self.contextW)
        f.create_dataset('biases', data = self.biases)
        f.flush()
        f.close()
        print('Saved!')
        

    def load(self, name = 'lbl.hdf5'):
        f = h5py.File(name, 'r')
        self.wordEm = f['wordEm'][:]
        self.contextW = f['contextW'][:]
        self.biases = f['biases'][:]
        self.index2word = f['index2word'][:]
        self.vocab = dict(zip(self.index2word, range(len(self.index2word) ) ) )
        

    def initialise(self):
        print('Initialising weights...')
        self.contextW = [(np.random.rand(self.dim, self.dim) - 0.5) / self.dim for i in range(self.context) ]
        self.wordEm = (np.random.rand(len(self.vocab), self.dim) - 0.5) / self.dim
        self.biases = np.asarray(self.frequencies, np.float64) / np.sum(self.frequencies)
        

    def prepare_vocabulary(self, sentences):
        print('Building vocabulary...')
        total = 0
        vocab = {}
        for sen_no, sentence in enumerate(sentences):
            for w in sentence:
                total += 1
                count = vocab.get(w, 0) + 1
                vocab[w] = count
        print('Visited {0} words and {1} sentences from the corpus'.format(total, sen_no + 1) )
        print('There are in total {0} different words'.format(len(vocab) ) )
        self.total = total
        
        self.vocab = {}
        self.index2word = []
        self.frequencies = []
        index = 0
        count_oov = 0
        for w, count in vocab.iteritems():
            if count >= self.threshold:
                self.vocab[w] = index
                self.index2word.append(w)
                self.frequencies.append(count)
                index += 1
            else:
                count_oov += count
        self.vocab['<>'] = index
        index += 1
        self.vocab['<s>'] = index
        index += 1
        self.vocab['</s>'] = index
        self.index2word.extend(['<>', '<s>', '</s>'])
        self.frequencies.extend([count_OOV, sen_no, sen_no] )
        print('\nThe size of vocabulary is: {0}, with threshold being {1}\n'.format(len(self.vocab), self.threshold) )


    def train(self, sentences, alpha = 0.025, min_alpha = 0.0235, batches = 1000):
        print('Start training...')
        self.alpha = alpha
        self.min_alpha = min_alpha
        count = 0
        start = time.time()
        last_elapsed = 0
        RARE = self.vocab['<>']
        r_hat = np.zeros(self.dim)
        delta_c = [np.zeros((self.dim, self.dim) ) for i in range(self.context) ]
        delta_r = np.zeros((len(self.vocab), self.dim) )
        for sentence in sentences:
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence) ):
                count += 1
                r_hat.fill(0)
                contextEm = []
                contextW = []
                indices = []
                for i, r in enumerate(sentence[pos - self.context : pos]):
                    if r == '<_>':
                        continue
                    index = self.vocab.get(r, RARE)
                    indices.append(index)
                    ri = self.wordEm[index]
                    ci = self.contextW[i]
                    contextEm.append(ri)
                    contextW.append(ci)
                    r_hat += np.dot(ci, ri)
                energy = np.exp(np.dot(self.wordEm, r_hat) + self.biases)
                probs = energy / np.sum(energy)
                w_index = self.vocab.get(sentence[pos], RARE)
                w = self.wordEm[w_index]
                
                '''
                *
                * The original unvectorised way to get the gradient from each (context,target) sequence 
                *
                for i in range(len(contextEm) ):
                    delta_c[-len(contextEm) + i] -= np.outer(w, contextEm[i] )
                    index = indices[i]
                    delta_r[w_index] -= np.dot(self.wordEm[index], contextW[i].T)
                    delta_r[index] -= np.dot(self.wordEm[w_index], contextW[i] )
                for i in xrange(len(self.vocab) ):
                    for j in range(len(contextEm) ):
                        delta_c[-len(contextEm) + j] += probs[i] * np.outer(self.wordEm[i], contextEm[j])
                        index = indices[j]
                        delta_r[i] += probs[i] * np.dot(self.wordEm[index], contextW[j].T)
                        delta_r[index] += probs[i] * np.dot(self.wordEm[i], contextW[j] )
                '''
                
                '''
                *
                * The equivalent vectorised version, which is much faster
                *
                '''
                probs[w_index] -= 1
                temp = np.dot(probs, self.wordEm)
                for i in range(len(contextEm) ):
                    delta_c[self.context - len(contextEm) + i] += np.outer(temp, contextEm[i] )
                VRC = np.zeros(self.dim)
                for i in range(len(contextEm) ):
                    VRC += np.dot(contextEm[i], contextW[i].T)
                delta_r += np.outer(probs, VRC)
                for i in range(len(contextEm) ):
                    delta_r[indices[i] ] += np.dot(temp, contextW[i])

                # update after visiting batches sequences
                if count % batches == 0:
                    alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
                    for i in range(self.context):
                        self.contextW[i] -= (delta_c[i] + 1e-5 * self.contextW[i]) * alpha
                    self.wordEm -= (delta_r + 1e-4 * self.wordEm) * alpha
                    for i in range(self.context):
                        delta_c[i].fill(0)
                    delta_r.fill(0)
                elapsed = time.time() - start
                if elapsed - last_elapsed > 1:
                    print('visited {0} words, with {1:.2f} Ws/s, alpha: {2}.'.format(count, count / elapsed, alpha) )
                    last_elapsed = elapsed

        # add all remaining gradients
        if count % batches != 0:
            alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
            for i in range(self.context):
                self.contextW[i] -= (delta_c[i] + 1e-5 * self.contextW[i]) * alpha
            self.wordEm -= (delta_r + 1e-4 * self.wordEm) * alpha
        print('Training is finished!')

                
    def perplexity(self, sentences):
        print('Calculating perplexity...')
        RARE = self.vocab['<>']
        # _no_eos means no end of sentence tag </s>
        count_no_eos = count = 0
        logProbs_no_eos = logProbs = 0
        r_hat = np.zeros(self.dim)
        for sentence in sentences:
            sentence = self.l_pad + sentence + self.r_pad
            for pos in range(self.context, len(sentence) ):
                count += 1
                count_no_eos += 1
                r_hat.fill(0)
                for i, r in enumerate(sentence[pos - self.context : pos]):
                    if r == '<_>':
                        continue
                    index = self.vocab.get(r, RARE)
                    ri = self.wordEm[index]
                    ci = self.contextW[i]
                    r_hat += np.dot(ci, ri)
                w_index = self.vocab.get(sentence[pos], RARE)
                energy = np.exp(np.dot(self.wordEm, r_hat) + self.biases)
                res = np.log(energy[w_index] / np.sum(energy) )
                logProbs += res
                logProbs_no_eos += res
            logProbs_no_eos -= res
            count_no_eos -=1
            # print results after each sentence
            ppl = np.exp(-logProbs / count)
            print('count: {0}'.format(count) )
            print('The perplexity is {0}'.format(ppl) )
        # the following displays the final perplexity
        ppl = np.exp(-logProbs / count)
        ppl_no_eos = np.exp(-logProbs_no_eos / count_no_eos)
        print('The perplexity with eos is {0}'.format(ppl) )
        print('               without eos is {0}'.format(ppl_no_eos) )
