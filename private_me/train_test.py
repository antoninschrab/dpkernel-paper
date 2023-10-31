# API for testing regime
from __future__ import division, print_function
import argparse
import os
import sys
import pickle as pkl
from functools import partial
from math import sqrt

import numpy as np

import private_me.data as datasets
import private_me.tst as tst
from private_me.util import NumpySeedContext, med_sq_distance, ContextTimer 
from private_me.private_mechanism import gauss_mech
from private_me.kernel import KGauss

__author__ = 'leon'

# function that puts help in each option with the corresponding defaults
def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault('help', "Default %(default)s.")
        return g.add_argument(*args, **kwargs)
    return f

def valid_reg(s):
    if s == 'auto':
        return s
    else:
        try:
            s = float(s)
        except:
            msg = 'Must be either a positive float or auto'
            raise argparse.ArgumentTypeError(msg)
        return s

# parameters for test, optimisation and file
def _add_args(subparser):
    me = subparser.add_argument_group("test parameters")
    m = get_adder(me)
    m('--test-type', choices=['ME', 'SCF'], default='ME')
    m('--privacy-type', choices=['None','local_meanCov', 'meanCov', 'statistic'], 
                        default='None',
                        help='Default: None, \
                              0. None: no privacy, epsilon and delta are ignored.\
                              1. local_meanCov (NTE): locally private with mean, Cov noise added,\
                              median heuristic computed on train (can be made private), \
                              require paired two sample test here. \
                              2. meanCov (TCMC): curator private with mean, Cov noise added,\
                              optimisation allowed.\
                              3. statistic (TCS): curator private with statistic noise added,\
                              optimisation allowed.'
                              )
    m('--reg', type=valid_reg, default='auto', help='Default auto, i.e. float otherwise, \
                                                     only used for statistic privacy,\
                                                     auto means it is set to 0.0001')
    m('--n-test-locs', type=int, default=5)
    m('--alpha', type=float, default=0.01)
    m('--epsilon', type=float, default=2.5)
    m('--delta', type=float, default=1e-5)
    m('--kernel', choices=['rbf'], default='rbf')
    m('--null', choices=['private', 'true', 'asymptotic'], default = 'private', 
                help='Default: private, if private use private finite null,\
                                        if true use non-private true null,\
                                        if asymptotic use asymptotic null.')
    m('--MC-noise-mechanism', choices=['analyse_gauss'], 
                              default='analyse_gauss',
                              help='Default: analyse_gauss, used for covariance pertubation')
    m('--gauss-mechanism', choices=['Normal', 'Improved'], default='Improved',
                              help='Default: Improved, used for any gauss-mechanism')
                              # https://arxiv.org/pdf/1805.06530.pdf
    m('--MC-mean-noise-prop', type=float, default=0.5, help='Default: 0.5, used for proportion of \
                                                               noise (epsilon and delta) on mean vs \
                                                               covariance in meanCov')
    optim = subparser.add_argument_group("optimisation/randomisation parameters")
    o = get_adder(optim)
    o('--tr-proportion', type=float, default=0.2,
                           help='Default: 0.2, if no optimisation, will ignore testing proportion.')
    o('--optim-locs', action='store_true', default=False,
                      help='Default: False, will randomise locations or freqs if False')
    o('--optim-bw', action='store_true', default=False,
                     help='Default: False, will use median heuristic for bw or private median heuristic')
    o('--random-locs', choices=['subset_XY', 'subset_even', 'fit_gauss'], default='subset_XY',
                       help='Default: subset_XY, subset_even useful when training is small, for ME only')
    o('--max-iter', type=int, default=200)
    o('--locs-step-size', type=float, default=0.1)
    o('--gwidth-step-size', type=float, default=0.1)
    o('--tol-fun', type=float, default=1e-3)
    o('--batch-proportion', type=float, default=1.0, help='Default: 1.0, 1.0 would mean do gradient descent')
    
    io = subparser.add_argument_group("I/O parameters")
    i = get_adder(io)
    io.add_argument('results_filename')

# datasets parameters
def make_parser(rest_of_args=_add_args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="The dataset to run on")
    # Subparser chosen by the first argument of your parser
    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group('Data parameters')
        rest_of_args(subparser)
        return data, get_adder(data)

    def toy_extra_args(g):
        g('--sample-size', type=int, default=6250)
        g('--sample', choices=['simulate'])
        g('--seed', type=int, default=np.random.randint(2**32),
          help="Seed for the simulate process (default: random).")

    blobs, b = add_subparser('blobs_data', 
        description = 
        'Mixture of 2d Gaussians arranged in a 2d grid. This dataset is used \
         in Chwialkovski et al., 2015 as well as Gretton et al., 2012. \
         Part of the code taken from Dino Sejdinovic and Kacper Chwialkovski code.')
    toy_extra_args(b)

    sg, s = add_subparser('same_gaussian', 
        description = 
        'Two same standard Gaussians for P, Q. The null hypothesis \
         H0: P=Q is true.')
    s('--dimension', type=int, default=50)
    toy_extra_args(s)

    gvd, gv = add_subparser('gaussian_var_diff',
        description = 
        'Toy dataset two in Chwialkovski et al., 2015. \
         P = N(0, I), Q = N(0, diag((2, 1, 1, ...))). Only the \
         variances of the first dimension differ.')
    gv('--dimension', type=int, default=50)
    toy_extra_args(gv)

    gmd, gm = add_subparser('gaussian_mean_diff',
        description = 
        'Toy dataset one in Chwialkovski et al., 2015. \
         P = N(0, I), Q = N( (mean-y,0,0, 000), I). Only the \
         first dimension of the means differ.')
    gm('--dimension', type=int, default=100)
    gm('--mean-y', type=float, default=1.0)
    toy_extra_args(gm)

    return parser

def check_output_file(file_name, parser):
    if os.path.exists(file_name):
        parser.error(("Output file {} exists, in case I am overwriting, change the name or delete it.")
                      .format(file_name))

def parse_args(rest_of_args=_add_args):
    parser = make_parser(rest_of_args)
    args = parser.parse_args()
    check_output_file(args.results_filename, parser)
    return args

def generate_data(args):
    if args.dataset == 'same_gaussian':
        sample_source = datasets.SSSameGauss(d=args.dimension)
    elif args.dataset == 'gaussian_var_diff':
        sample_source = datasets.SSGaussVarDiff(d=args.dimension)
    elif args.dataset == 'gaussian_mean_diff':
        sample_source = datasets.SSGaussMeanDiff(d=args.dimension, my=args.mean_y)
    elif args.dataset == 'blobs_data':
        sample_source = datasets.SSBlobs()
    else:
        raise ValueError("unknown dataset {}".format(args.dataset))
    tst_data = sample_source.sample(args.sample_size, seed=args.seed)
    tr, te = tst_data.split_tr_te(tr_proportion=args.tr_proportion, seed=args.seed+20)
    return tr, te

def construct_kernel(privacy_type, k_type, k_para):
    if k_type == 'rbf':
        kernel = KGauss(k_para) #gwidth2
    else:
        raise ValueError('Kernel type {} not recognised'.format(k_type))
    return kernel

def construct_test(test_type, privacy_type, test_locs, kernel, gwidth2, reg, alpha=0.05):
    if privacy_type == 'None' and test_type == 'ME':
        test = tst.METest(test_locs, kernel, alpha)
    elif privacy_type == 'None' and test_type == 'SCF':
        test = tst.SmoothCFTest(test_locs, gwidth2, alpha)
    elif privacy_type == 'meanCov': 
        test = tst.PrivateMC_Test(test_type, test_locs, kernel, gwidth2, alpha)
    elif privacy_type == 'statistic':
        test = tst.PrivateSt_Test(test_type, test_locs, kernel, gwidth2, reg, alpha)
    elif privacy_type == 'local_meanCov':
        test = tst.PrivateLocal_PairTest(test_type, test_locs, kernel, gwidth2, alpha)
    else:
        raise ValueError('Test type {}, Privacy type {} not recognised'.format(test_type, privacy_type))
    return test

def test_configs(train, test, args):
    seed = args.seed + 92856
    dict_config = {'tst_data': train, 'alpha':args.alpha, 
                   'n_test_locs':args.n_test_locs,
                   'max_iter': args.max_iter, 
                   'locs_step_size': args.locs_step_size,
                   'gwidth_step_size': args.gwidth_step_size,
                   'batch_proportion': args.batch_proportion,
                   'tol_fun': args.tol_fun, 'seed': seed}

    if args.privacy_type == 'local_meanCov':
        args.optim_locs = False
        args.optim_bw = False
    if args.test_type == 'ME':
        if not args.optim_locs:
            print('Sampling test locations using method: {}'.format(args.random_locs))
            if args.random_locs == 'subset_XY': #sample from whole data
                test_locs = tst.MeanEmbeddingTest.init_locs_subset(train, args.n_test_locs, seed=seed+1)
            elif args.random_locs == 'subset_even':
                # TODO: Implement stratify!
                raise NotImplementedError
            elif args.random_locs =='fit_gauss':
                test_locs = tst.MeanEmbeddingTest.init_locs_2randn(train, args.n_test_locs, seed=seed)
            else:
                raise ValueError('random_locs must be either subset_XY, subset_even or fit_gauss')
        if args.optim_locs and args.optim_bw: #optimise landmarks and bw
            print('Optimise bandwidth and test locations')
            test_locs, gwidth2, info = tst.MeanEmbeddingTest.optimize_locs_width(**dict_config)
        elif args.optim_locs and not args.optim_bw:
            raise NotImplementedError
        elif args.optim_bw: 
            raise NotImplementedError
        else: # we do not optimise either bw or landmarks 
            if args.privacy_type == 'local_meanCov':
                print('Caculate bandwidth heuristic under local private setting using full training data')
                gwidth2 = med_sq_distance(train.stack_xy(), n_sub=1000)
            elif args.privacy_type in ['None', 'meanCov', 'statistic']:
                print('Caculate bandwidth heuristic using test data')
                gwidth2 = med_sq_distance(test.stack_xy(), n_sub=1000)
            else:
                raise ValueError('Choose one of the allowed privacy types.')
    elif args.test_type == 'SCF':
        if not args.optim_locs:
            print('Sampling test frequencies using method using train set')
            d = train.dim()
            test_locs = np.random.randn(args.n_test_locs, d)
        if args.optim_locs and args.optim_bw:
            test_locs, gwidth2, info = tst.SmoothCFTest.optimize_freqs_width(**dict_config)
        elif args.optim_locs and not args.optim_bw:
            raise NotImplementedError
        elif args.optim_bw: #optim_locs: False here
            raise NotImplementedError
        else:
            if args.privacy_type == 'local_meanCov':
                gwidth2 = train.mean_std()*train.dim()**0.5
            elif args.privacy_type in ['None', 'meanCov', 'statistic']:
                print('Caculate gwidth2 heuristic using test data')
                gwidth2 = test.mean_std()*test.dim()**0.5
            else:
                raise ValueError('Choose one of the allowed privacy types.')
    else:
        raise ValueError('Choose one of the test type')
    return test_locs, gwidth2


def main():
    args = parse_args()
    print("Loading data...")
    tr, te = generate_data(args)
    with ContextTimer() as t:
        print("Choosing locations and parameters...")
        test_locs, gwidth2 = test_configs(tr, te, args)
        if args.privacy_type == 'None':
            args.epsilon = args.delta = None
        print("Construct Private {} Test under epsilon: {}, delta: {}, \
               Privacy Type: {} under null: {}".format(args.test_type, args.epsilon, args.delta, 
                                                       args.privacy_type, args.null))
        kernel = construct_kernel(args.privacy_type, args.kernel, gwidth2)
        test = construct_test(args.test_type, args.privacy_type, test_locs, kernel, gwidth2, args.reg, args.alpha)
        if args.privacy_type in ['local_meanCov', 'meanCov']:
            test_result = test.perform_test(te, epsilon=args.epsilon, delta=args.delta, 
                                            null=args.null, noise=args.MC_noise_mechanism,
                                            gauss_noise=args.gauss_mechanism,
                                            mean_noise_prop=args.MC_mean_noise_prop, seed=args.seed)
        else:
            test_result = test.perform_test(te, epsilon=args.epsilon, delta=args.delta, 
                                            gauss_noise=args.gauss_mechanism,
                                            null=args.null, seed=args.seed)
    print('Test completed, time taken: {}s'.format(t.secs))
    save_results = {'args': args, 'results': test_result}
    with open( args.results_filename, "wb" ) as file:
        pkl.dump(save_results, file)
    print(save_results)
    print('Saved result to {}'.format(args.results_filename))
    
if __name__ == '__main__':
    main()
