import numpy as np
import time
import h5py
import utils
from multiprocessing import Process, Lock, Queue

class LBL:
    def __init__(self, sentences = None, alpha = 0.025, min_alpha = 0.025, dim = 100, context = 5, threshold = 0, batches = 1000, workers = 4):
        '''
        sentences should be a generator
        vocab, for each word, stores its corresponding namedtuple word
        index2word records the index for each word
        total is the number of words in the training set
        alpha and min_alpha are the upper bound and lower bound for the learning rate
        dim is the dimension for each word embedding
        wordEm is a (vocabulary_size * dim) matrix, each row of which is a word embedding
        context is the size of history window
        words occur less than threshold times will be regarded as rare and will be mapped to a special token '<>'
        <_> is null padding, <s> denotes start of sentence, </s> means the end of sentence
        workers is the number of processes created for multitasking
        '''
        self.vocab = {}
        self.index2word = []
        self.total = 727507
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.wordEm = self.contextW = None
        self.dim = dim
        self.context = context
        self.threshold = threshold
        self.l_pad = ['<_>'] * (self.context - 1) + ['<s>']
        self.r_pad = ['</s>']
        if sentences is not None:
            self.prepare_vocabulary(sentences)
            self.initialise()
            self.train(sentences, alpha = alpha, min_alpha = min_alpha, batches = batches, workers = workers)
            self.save()
            

    def save(self):
        print('Saving model...')
        f = h5py.File('lbl.hdf5', 'w')
        f.create_dataset('index2word', data = self.index2word)
        f.create_dataset('wordEm', data = self .wordEm)
        f.create_dataset('contextW', data = self.contextW)
        f.flush()
        f.close()
        print('Saved!')
        

    def load(self, name = 'lbl.hdf5'):
        f = h5py.File(name, 'r')
        self.wordEm = f['wordEm'][:]
        self.contextW = f['contextW'][:]
        self.index2word = f['index2word'][:]
        self.vocab = dict(zip(self.index2word, range(len(self.index2word) ) ) )
        

    def initialise(self):
        print('Initialising weights...')
        # contextW_raw contains raw arrays which will be shared among processes
        self.contextW_raw = [utils.getSharedArray('d', self.dim * self.dim) for i in range(self.context) ]
        # contextW contains numpy wrappers which can be easily used by the parent process
        self.contextW = [utils.toNumpyArray(self.contextW_raw[i], np.float64, (self.dim, self.dim) ) for i in range(self.context) ]
        for i in range(self.context):
            self.contextW[i] += ((np.random.rand(self.dim, self.dim) - 0.5) / self.dim)
        self.wordEm_raw = utils.getSharedArray('d', len(self.vocab) * self.dim )
        self.wordEm = utils.toNumpyArray(self.wordEm_raw, np.float64, (len(self.vocab), self.dim) )
        self.wordEm += ((np.random.rand(len(self.vocab), self.dim) - 0.5) / self.dim)
            
        

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
        #self.total = total
        
        self.vocab = {}
        self.index2word = []
        index = 0
        for w, count in vocab.iteritems():
            if count >= self.threshold:
                self.vocab[w] = index
                self.index2word.append(w)
                index += 1
        #self.vocab['<>'] = index
        #index += 1
        self.vocab['<s>'] = index
        index += 1
        self.vocab['</s>'] = index
        self.index2word.extend(['<>', '<s>', '</s>'])
        print('\nThe size of vocabulary is: {0}, with threshold being {1}\n'.format(len(self.vocab), self.threshold) )


    def train(self, sentences, alpha = 0.025, min_alpha = 0.0235, batches = 1000, workers = 4):
        print('Start training...')
        self.alpha = alpha
        self.min_alpha = min_alpha
        count = 0
        # barrier is used to sync parent and all workers
        barrier = utils.getBarrier(workers + 1)
        lock = Lock()
        queue = Queue(workers)
        # delta_c_raw contains context weights for each position, they are shared, so each child process can 
        # add their delta on them. delta_c is a numpy wrapper which makes the parent process handle it easily
        delta_c_raw = [utils.getSharedArray('d', self.dim * self.dim) for i in range(self.context) ]
        delta_c = [utils.toNumpyArray(delta_c_raw[i], np.float64, (self.dim, self.dim) ) for i in range(self.context) ]
        delta_r_raw = utils.getSharedArray('d', len(self.vocab) * self.dim) 
        delta_r = utils.toNumpyArray(delta_r_raw, np.float64, (len(self.vocab), self.dim) )
        
        '''
        vocab: dictionary containing each word and its index, it's copied from the parent process
        self_wordEm, self_contextW, self_delta_c, self_delta_r point to data which is shared among parent and child processes
        '''
        def worker(vocab, self_wordEm, self_contextW,
                   self_delta_c, self_delta_r, dim, 
                   context, barrier, lock, queue):
            self_wordEm = utils.toNumpyArray(self_wordEm, np.float64, (len(vocab), dim) )
            self_contextW = [utils.toNumpyArray(self_contextW[i], np.float64, (dim, dim) ) for i in range(context) ]
            self_delta_r = utils.toNumpyArray(self_delta_r, np.float64, (len(vocab), dim) )
            self_delta_c = [utils.toNumpyArray(self_delta_c[i], np.float64, (dim, dim) ) for i in range(context) ]
            # delta_c and delta_r are local to a child process, deltas will be stored in them.
            # after finishing its task, a child process will add them to their counterparts in 
            # the parent process via self_delta_r and self_delta_c
            delta_c = [np.zeros((dim, dim) ) for i in range(context) ]
            delta_r = np.zeros((len(vocab), dim) )

            # the index of a rare word
            RARE = vocab['<>']
            while True:
                task = queue.get()
                if task is None:
                    break
                for sentence in task:
                    for pos in range(context, len(sentence) ):
                        r_hat = np.zeros(dim)
                        contextEm = []
                        contextW = []
                        indices = []

                        for i, r in enumerate(sentence[pos - context : pos]):
                            if r == '<_>':
                                continue
                            index = vocab.get(r, RARE)
                            indices.append(index)
                            ri = self_wordEm[index]
                            ci = self_contextW[i]
                            contextEm.append(ri)
                            contextW.append(ci)
                            r_hat += np.dot(ci, ri)

                        energy = np.exp(np.dot(self_wordEm, r_hat) )
                        probs = energy / np.sum(energy)
                        w_index = vocab.get(sentence[pos], RARE)
                        w = self_wordEm[w_index]                        
                        probs[w_index] -= 1
                        probs = probs.reshape(len(probs), 1)

                        temp = np.sum(probs * self_wordEm, axis = 0)
                        for i in range(len(contextEm) ):
                            delta_c[self.context - len(contextEm) + i] += np.outer(temp, contextEm[i] )
                        VRC = np.zeros(dim)
                        for i in range(len(contextEm) ):
                            VRC += np.dot(contextEm[i], contextW[i].T)
                        delta_r += probs * VRC
                        delta_r[indices] += [np.dot(temp, contextW[i]) for i in range(len(contextEm) ) ]
                
                lock.acquire()
                for i in range(context):
                    self_delta_c[i] += delta_c[i]
                lock.release()
                lock.acquire()
                self_delta_r += delta_r
                lock.release()
                barrier.sync()

                delta_c = [np.zeros((dim, dim) ) for i in range(context) ]
                delta_r = np.zeros((len(vocab), dim) )


        
        args = (self.vocab, self.wordEm_raw, self.contextW_raw, 
                delta_c_raw, delta_r_raw, self.dim, self.context, 
                barrier, lock, queue)
        pool = [Process(target = worker, args = args)  for i in range(workers) ]
        for p in pool:
            p.daemon = True
            p.start()
        
        distributor = utils.generateTasks(iter(sentences), self.l_pad, self.r_pad, workers, batches)
        start = time.time()
        for tasks in distributor:
            for i in range(workers):
                queue.put(tasks[i], block = False)                
            count += batches
            alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1 - 1.0 * count / self.total)
            barrier.sync()
            # this point, all child processes have finished their task and parent can update safely
            for i in range(self.context):
                self.contextW[i] -= (delta_c[i] + 1e-5 * self.contextW[i]) * alpha
            self.wordEm -= (delta_r + 1e-4 * self.wordEm) * alpha

            for i in range(self.context):
                delta_c[i].fill(0)
            delta_r.fill(0)
            elapsed = time.time() - start
            print('visited {0} words, with {1:.2f} Ws/s, alpha: {2}.'.format(count, count / elapsed, alpha) )
        # notify processes to exit
        for i in range(workers):
            queue.put(None)
        for p in pool:
            p.join()
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
                r_hat = np.zeros(self.dim)
                for i, r in enumerate(sentence[pos - self.context : pos]):
                    if r == '<_>':
                        continue
                    index = self.vocab.get(r, RARE)
                    ri = self.wordEm[index]
                    ci = self.contextW[i]
                    r_hat += np.dot(ci, ri)
                w_index = self.vocab.get(sentence[pos], RARE)
                energy = np.exp(np.dot(self.wordEm, r_hat) )
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
