from multiprocessing import Semaphore
from multiprocessing.managers import BaseManager
from multiprocessing.sharedctypes import RawArray
import numpy

'''
Barrier is used to synchronise n > 1 processes, each process should
call sync on barrier. Once all 'total' number of processes have called
sync, all processes can continue otherwise wait.
'''
class Barrier(object):
    def __init__(self, total = 2):
        self.waiting = 0
        self.total = total
        self.waitSem = Semaphore()
        self.waitSem.acquire()
        self.mutex = Semaphore()
    
    def sync(self):
        self.mutex.acquire()
        if self.waiting == self.total - 1:
            self.waitSem.release()
        else:
            self.waiting += 1
            self.mutex.release()
            self.waitSem.acquire()
            self.waiting -= 1
            if self.waiting ==0:
                self.mutex.release()
            else:
                self.waitSem.release()


class MyManager(BaseManager):
    pass



'''
the returned Barrier proxy can be shared among processes, each of
which can call sync() to sync with the others
'''
def getBarrier(total):
    MyManager.register('Barrier', Barrier)
    manager = MyManager()
    manager.start()
    return manager.Barrier(total)



'''
this function is used to group sentences into tasks where each contains
(batches / task_num) words, it returns a generator.
sentences should be a generator
'''
def generateTasks(sentences, l_pad, r_pad, task_num, batches):
    task_num = int(task_num)
    batches = int(batches)
    tasks = [[] for i in range(task_num) ]
    task_size = batches / task_num
    window = len(l_pad) # the size of the context
    unfinished = False # indicate whether a task has been full
    leftover = None # part of a sentence which should be put into other tasks later
    
    while True:
        i = 0
        while i < task_num:
            if not unfinished:
                count = 0
            if not leftover:
                for sentence in sentences:
                    count += (len(sentence) + 1) # eos is counted as a word
                    sentence = l_pad + sentence + r_pad
                    if count <= task_size:
                        tasks[i].append(sentence)
                        if count == task_size:
                            unfinished = False
                            break
                    else:
                        separator = len(sentence) - count + task_size
                        tasks[i].append(sentence[ : separator])
                        leftover = sentence[separator - window : ] # include previous window words as context
                        unfinished = False
                        break
                if count == task_size or leftover:
                    i += 1
                    continue
                yield tasks # returns tasks once there is no sentence left
                return # indicates the generator is exhausted
            else:
                while leftover:
                    count += (len(leftover) - window) # the previous window words are context
                    if count <= task_size:
                        tasks[i].append(leftover)
                        leftover = None
                        if count < task_size:
                            unfinished = True # indicates the task is not full which should read from sentences
                    else:
                        separator = len(leftover) - count + task_size
                        tasks[i].append(leftover[ : separator])
                        leftover = leftover[separator - window : ]
                        break
                if not unfinished:
                    i += 1
        yield tasks
        tasks = [[] for i in range(task_num) ]
    


'''
this function is used to create ctype arrays which can be shared
among processes, the array will be zero initialised
'''
def getSharedArray(dtype, size):
    return RawArray(dtype, size)



'''
this function is used to convert the ctype arrays created by
getSharedArray to a numpy array with corresponding shape
'''
def toNumpyArray(buffer, dtype, shape):
    array = numpy.frombuffer(buffer, dtype)
    return array.reshape(shape)



class Text:
    # returns a generator
    def __iter__(self):
        with open(self.name) as f:
            for line in f:
                yield line.split()
                
    def __init__(self,name):
        self.name = name
