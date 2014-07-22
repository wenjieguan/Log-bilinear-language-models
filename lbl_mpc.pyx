import cython
from cpython.cobject cimport PyCObject_AsVoidPtr
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset
from scipy.linalg.blas import fblas

INT32 = np.int32
ctypedef np.int32_t INT32_t
SINGLE = np.float32
ctypedef np.float32_t SINGLE_t

ctypedef SINGLE_t *SPTR

ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY)
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY)
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX)

cdef saxpy_ptr saxpy = <saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)    # y += alpha * x
cdef sdot_ptr sdot = <sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)        # float = dot(x, y)
cdef sscal_ptr sscal = <sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)    # x = alpha * x

cdef int ONE = 1
cdef SINGLE_t ONEF = <SINGLE_t>1.0



cdef inline void update(const int start, const int end, INT32_t *sentence,
                        const int vocab_size, const int context, const int dim,
                        SINGLE_t *wordEm, SINGLE_t **contextW, SINGLE_t *biases, 
                        SINGLE_t *delta_r, SINGLE_t **delta_c, 
                        SINGLE_t *work_d, SINGLE_t *work_v):

    cdef int i, j
    cdef SINGLE_t sum = <SINGLE_t>0.0
    cdef SINGLE_t alpha  
    
    #energy = np.exp(np.dot(self_wordEm, r_hat) + self_biases)
    for i in range(vocab_size):
        work_v[i] = <SINGLE_t>exp(<SINGLE_t>sdot(&dim, &wordEm[i * dim], &ONE, work_d, &ONE) + biases[i] )
        sum += work_v[i]
    
    #probs = energy / np.sum(energy)
    alpha = ONEF / sum
    sscal(&vocab_size, &alpha, work_v, &ONE)
    
    #probs[w_index] -= 1
    work_v[sentence[end] ] -= ONEF
    
    #VRC = np.zeros(dim, np.float32)
    #for i in range(len(contextEm) ):
    #    VRC += np.dot(contextEm[i], contextW[i].T)
    memset(work_d, 0, dim * cython.sizeof(SINGLE_t) )
    for i in range(start, end):
        for j in range(dim):
            work_d[j] += <SINGLE_t>sdot(&dim, &wordEm[sentence[i] * dim], &ONE, &contextW[context - end + i][j * dim], &ONE)
    
    #delta_r += np.outer(probs, VRC)
    for i in range(vocab_size):
        saxpy(&dim, &work_v[i], work_d, &ONE, &delta_r[i * dim], &ONE)
    
    #temp = np.dot(probs, self_wordEm)
    for i in range(dim):
        work_d[i] = <SINGLE_t>sdot(&vocab_size, work_v, &ONE, &wordEm[i], &dim)
    
    #delta_r[indices] += [np.dot(temp, contextW[i]) for i in range(len(contextEm) ) ]
    for i in range(start, end):
        for j in range(dim):
            delta_r[sentence[i] * dim + j] += <SINGLE_t>sdot(&dim, work_d, &ONE, &contextW[context - end + i][j], &dim)
    
    #for i in range(len(contextEm) ):
    #    delta_c[context - len(contextEm) + i] += np.outer(temp, contextEm[i] )
    for i in range(start, end):
        for j in range(dim):
            saxpy(&dim, &work_d[j], &wordEm[sentence[i] * dim], &ONE, &delta_c[context - end + i][j * dim], &ONE)
    


cdef int NULLPAD_IND = -1

def train_sentence_fast(model, _sentence, _delta_c, _delta_r, _work_d, _work_v): 
    cdef int pos, w, i, j, start, end
    cdef int sen_len = <int>len(_sentence)
    cdef int context = <int>(model.context)
    cdef int dim = <int>(model.dim )
    cdef int vocab_size = <int>len(model.vocab)
    cdef SINGLE_t *r
    cdef SINGLE_t *c
     
    cdef SINGLE_t *wordEm = <SINGLE_t *>(np.PyArray_DATA(model.wordEm) )
    cdef SINGLE_t *biases = <SINGLE_t *>(np.PyArray_DATA(model.biases) )
    cdef SINGLE_t **contextW = <SINGLE_t **>PyMem_Malloc(context * cython.sizeof(SPTR) )
    
    if contextW is NULL:
        raise MemoryError()
    for i in range(context):
        contextW[i] = <SINGLE_t *>(np.PyArray_DATA(model.contextW[i]) )   
    
    cdef SINGLE_t *delta_r = <SINGLE_t *>(np.PyArray_DATA(_delta_r) )
    cdef SINGLE_t **delta_c = <SINGLE_t **>PyMem_Malloc(context * cython.sizeof(SPTR) )   
    
    if delta_c is NULL:
        raise MemoryError() 
    for i in range(context):
        delta_c[i] = <SINGLE_t *>(np.PyArray_DATA(_delta_c[i]) )
    
    cdef SINGLE_t *work_d = <SINGLE_t *>(np.PyArray_DATA(_work_d) )
    cdef SINGLE_t *work_v = <SINGLE_t *>(np.PyArray_DATA(_work_v) )
    # sentence should be a numpy array with int32
    cdef INT32_t *sentence = <INT32_t *>(np.PyArray_DATA(_sentence) )   
    
    for pos in range(context, sen_len):
        start = pos - context
        end = pos
        memset(work_d, 0, dim * cython.sizeof(SINGLE_t) )
        i = 0
        for w in range(pos - context, pos):
            i += 1
            if sentence[w] == NULLPAD_IND:
                start += 1
            else:
                r = &wordEm[sentence[w] * dim]
                c = contextW[i - 1]
                # r_hat += dot(c,r)
                for j in range(dim):
                    work_d[j] += <SINGLE_t>sdot(&dim, &c[j * dim], &ONE, r, &ONE)
                
        # now work_d actually is r_hat
        update(start, end, sentence, 
               vocab_size, context, dim, 
               wordEm, contextW, biases, 
               delta_r, delta_c, work_d, work_v)
              
    PyMem_Free(contextW)
    PyMem_Free(delta_c)
        
        
