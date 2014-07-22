import numpy as np
import time
import h5py
import utils
from multiprocessing import Process, Lock, Queue

class LBL:
    def __init__(self, sentences = None, alpha = 0.025, min_alpha = 0.0237, dim = 100, context = 5, threshold = 3, batches = 1000, workers = 4):
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
            self.train(sentences, alpha = alpha, min_alpha = min_alpha, batches = batches, workers = workers)
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
        vocab_size, dim = f['wordEm'].shape
        context = (f['contextW'].shape)[0]
        self.wordEm_raw = utils.getSharedArray('f', vocab_size * dim )
        self.wordEm = utils.toNumpyArray(self.wordEm_raw, np.float32, (vocab_size, dim) )
        self.wordEm += f['wordEm'][:] 
        self.contextW_raw = [utils.getSharedArray('f', dim * dim) for i in range(context) ]
        self.contextW = [utils.toNumpyArray(self.contextW_raw[i], np.float32, (dim, dim) ) for i in range(context) ]
        for i in range(context):
            self.contextW[i] += f['contextW'][i][:]
        self.biases_raw= utils.getSharedArray('f', vocab_size)
        self.biases = utils.toNumpyArray(self.biases_raw, np.float32, vocab_size)
        self.biases += f['biases'][:]
        self.index2word = f['index2word'][:]
        self.vocab = dict(zip(self.index2word, range(len(self.index2word) ) ) )
        

    def initialise(self):
        print('Initialising weights...')
        # contextW_raw contains raw arrays which will be shared among processes
        self.contextW_raw = [utils.getSharedArray('f', self.dim * self.dim) for i in range(self.context) ]
        # contextW contains numpy wrappers which can be easily used by the parent process
        self.contextW = [utils.toNumpyArray(self.contextW_raw[i], np.float32, (self.dim, self.dim) ) for i in range(self.context) ]
        for i in range(self.context):
            self.contextW[i] += ((np.random.rand(self.dim, self.dim).astype(np.float32) - 0.5) / self.dim)
        self.wordEm_raw = utils.getSharedArray('f', len(self.vocab) * self.dim )
        self.wordEm = utils.toNumpyArray(self.wordEm_raw, np.float32, (len(self.vocab), self.dim) )
        self.wordEm += ((np.random.rand(len(self.vocab), self.dim).astype(np.float32) - 0.5) / self.dim)
        self.biases_raw= utils.getSharedArray('f', len(self.vocab) )
        self.biases = utils.toNumpyArray(self.biases_raw, np.float32, len(self.vocab) )
        self.biases += (np.asarray(self.frequencies, np.float32) / np.sum(self.frequencies) )
            
        
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
        self.frequencies.extend([count_oov, sen_no, sen_no] )
        print('\nThe size of vocabulary is: {0}, with threshold being {1}\n'.format(len(self.vocab), self.threshold) )


    def train(self, sentences, alpha = 0.025, min_alpha = 0.0235, batches = 1000, workers = 4):
        print('Start training...')
        self.alpha = alpha
        self.min_alpha = min_alpha
        count = 0
        # barrier is used to sync parent and all workers
        barrier = utils.getBarrier(workers + 1)
        lock1 = Lock()
        lock2 = Lock()
        queue = Queue(workers)
        # delta_c_raw contains context weights for each position, they are shared, so each child process can 
        # add their delta on them. delta_c is a numpy wrapper which makes the parent process handle it easily
        delta_c_raw = [utils.getSharedArray('f', self.dim * self.dim) for i in range(self.context) ]
        delta_c = [utils.toNumpyArray(delta_c_raw[i], np.float32, (self.dim, self.dim) ) for i in range(self.context) ]
        delta_r_raw = utils.getSharedArray('f', len(self.vocab) * self.dim) 
        delta_r = utils.toNumpyArray(delta_r_raw, np.float32, (len(self.vocab), self.dim) )
        


        '''
        vocab: dictionary containing each word and its index, it's copied from the parent process
        self_wordEm, self_contextW, self_biases, self_delta_c, self_delta_r point to data which is shared among parent and child processes
        '''
        def worker(model, self_delta_c, self_delta_r, barrier, lock1, lock2, queue):
            self_wordEm = utils.toNumpyArray(model.wordEm_raw, np.float32, (len(model.vocab), model.dim) )
            self_contextW = [utils.toNumpyArray(model.contextW_raw[i], np.float32, (model.dim, model.dim) ) for i in range(model.context) ]
            self_biases = utils.toNumpyArray(model.biases_raw, np.float32, len(model.vocab) )
            self_delta_r = utils.toNumpyArray(self_delta_r, np.float32, (len(model.vocab), model.dim) )
            self_delta_c = [utils.toNumpyArray(self_delta_c[i], np.float32, (model.dim, model.dim) ) for i in range(model.context) ]
            # delta_c and delta_r are local to a child process, deltas will be stored in them.
            # after finishing its task, a child process will add them to their counterparts in 
            # the parent process via self_delta_r and self_delta_c
            delta_c = [np.zeros((model.dim, model.dim), np.float32) for i in range(model.context) ]
            delta_r = np.zeros((len(model.vocab), model.dim), np.float32)

            # the index of a rare word
            RARE = model.vocab['<>']
            r_hat = np.zeros(model.dim, np.float32)
            VRC = np.zeros(model.dim, np.float32)
            while True:
                task = queue.get()
                if task is None:
                    break
                for sentence in task:
                    for pos in range(model.context, len(sentence) ):
                        r_hat.fill(0)
                        contextEm = []
                        contextW = []
                        indices = []

                        for i, r in enumerate(sentence[pos - model.context : pos]):
                            if r == '<_>':
                                continue
                            index = model.vocab.get(r, RARE)
                            indices.append(index)
                            ri = self_wordEm[index]
                            ci = self_contextW[i]
                            contextEm.append(ri)
                            contextW.append(ci)
                            r_hat += np.dot(ci, ri)

                        energy = np.exp(np.dot(self_wordEm, r_hat) + self_biases)
                        probs = energy / np.sum(energy)
                        w_index = model.vocab.get(sentence[pos], RARE)
                        probs[w_index] -= 1

                        temp = np.dot(probs, self_wordEm)
                        for i in range(len(contextEm) ):
                            delta_c[model.context - len(contextEm) + i] += np.outer(temp, contextEm[i] )
                        VRC.fill(0)
                        for i in range(len(contextEm) ):
                            VRC += np.dot(contextEm[i], contextW[i].T)
                        delta_r += np.outer(probs, VRC)
                        for i in range(len(contextEm) ):
                            delta_r[indices[i] ] += np.dot(temp, contextW[i])
                
                lock1.acquire()
                for i in range(model.context):
                    self_delta_c[i] += delta_c[i]
                lock1.release()
                lock2.acquire()
                self_delta_r += delta_r
                lock2.release()
                barrier.sync()

                for i in range(model.context):
                    delta_c[i].fill(0)
                delta_r.fill(0)


        
        args = (self, delta_c_raw, delta_r_raw, barrier, lock1, lock2, queue)
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
        r_hat = np.zeros(self.dim, np.float32)
        # _no_eos means no end of sentence tag </s>
        count_no_eos = count = 0
        logProbs_no_eos = logProbs = 0
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
