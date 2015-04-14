
#coding:utf-8
'''
Created on 2015��4��11��
 
@author: Richard
'''
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
# rng = numpy.random.RandomState(123)
# theano_rng = RandomStreams(rng.randint(2 ** 30))

def get_corrupted_input_binomial(theano_rng, input, corruption_level):
    """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
    return theano_rng.binomial(size = input.shape, n = 1, 
                               p=1 - corruption_level, 
                               dtype=theano.config.floatX) * input




def get_corrupted_input_gaussian(theano_rng, input):
    """This function adds Gaussian noise to the input.
    """
    return 0.9*theano_rng.normal(size = input.shape, avg = 0.0,
                              std = 0.1, dtype = theano.config.floatX) + 0.1*theano_rng.normal(size=input.shape, 
                              avg=4, std=0.1, dtype=theano.config.floatX) + input



def test(dataset = 'mnist.pkl.gz', output_folder = 'plots'):
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    input = T.dmatrix('input')
    
    output = get_corrupted_input_gaussian(theano_rng = theano_rng, input = input)
    
    corrupt = theano.function([input], output)
    
    mnist_noise = corrupt(train_set_x.get_value(borrow = True))
    mnist_noise = theano.shared(value=mnist_noise, name='mnist_noise', borrow = True)
#     print train_set_x.get_value(borrow=True)[0]
#     print mnist_noise.get_value(borrow=True)[0]
    
    image_clean = Image.fromarray(tile_raster_images(X = train_set_x.get_value(borrow = True),
                                               img_shape=(28, 28), tile_shape=(10, 10),
                                               tile_spacing=(1,1)))
    image_clean.save('clean.png')
    
    image_noise = Image.fromarray(tile_raster_images(X = mnist_noise.get_value(borrow = True),
                                               img_shape=(28, 28), tile_shape=(10, 10),
                                               tile_spacing=(1,1)))
    image_noise.save('noise.png')
    
    print 'Done!'


if __name__ == '__main__':
    test()





