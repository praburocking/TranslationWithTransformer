import torch
import gc
import sys
import logging
import pickle
file_name='R:/studies/transformers/translation/simple transformer/code/app2.log'
logging.basicConfig(filename=file_name, format='%(asctime)s %(message)s', filemode='w')

#Let us Create an object
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

def print_mem_use():
    total_tensor_obj_size=0
    total_obj_size = 0
    bytes_in_mb=1000000

    for obj in gc.get_objects():
        total_obj_size += sys.getsizeof(obj)
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # logger.debug(str(type(obj))+"----------"+str(sys.getsizeof(obj/bytes_in_mb)))
                total_tensor_obj_size+=sys.getsizeof(obj)
        except:
            pass
    logger.debug('########## total python object size -----'+str(total_obj_size/bytes_in_mb)+'total tensor object size -----'+str(total_tensor_obj_size/bytes_in_mb))


def save_obj(obj,file_name):
    with open(file_name, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as inp:
      return pickle.load(inp)