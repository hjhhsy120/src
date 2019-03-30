from __future__ import print_function

import vctrainer, deepwalk, app, combine, rw2vc, simrank, node2vec
import generalwalk
import dumpwalk
import fixedpair

import vctrainer_ops

def getmodel(model, g, args):
    if model == 'deepwalk':
        return deepwalk.DeepWalk(graph=g, batch_size=args.batch_size, fac=args.epoch_fac, window=args.window_size,
                                 degree_bound=args.degree_bound, degree_power=args.degree_power)
    if model == 'app':
        return app.APP(graph=g, batch_size=args.batch_size, stop_factor=args.app_jump_factor, sample=args.app_sample, step=args.app_step)

    if model == 'deepwalk,app':
        return combine.combine(g, args)

    if model == 'generalwalk':
        return generalwalk.GeneralWalk(g, batch_size=args.batch_size,  fac=args.epoch_fac, window=args.window_size,
                                       degree_bound=args.degree_bound, degree_power=args.degree_power)

    if model == 'dumpwalk':
        return dumpwalk.dumpwalk(g, fac=args.epoch_fac, window=args.window_size,
                                       degree_bound=args.degree_bound, degree_power=args.degree_power)

    if model == 'fixedpair':
        return fixedpair.fixedpair(g, pair_file=args.pair_file)

    if model == 'rw2vc':
        return rw2vc.rw2vc(graph=g, rw_file=args.rw_file, emb_file=args.output,
                           window=args.window_size, emb_model=args.emb_model, rep_size=args.representation_size,
                           epoch=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.lr, negative_ratio=args.negative_ratio)

    if model == 'simrank':
        return simrank.SimRank(graph=g, fac=args.epoch_fac,
                            maxIteration=args.simrank_maxiter, damp=args.simrank_damp)

    if model == 'node2vec':
        return node2vec.Node2vec(graph=g, fac=args.epoch_fac, window=args.window_size,
                                 degree_bound=args.degree_bound, degree_power=args.degree_power,
                                 p=args.node2vec_p, q=args.node2vec_q)


    model_list = ['app', 'deepwalk', 'deepwalk,app', 'rw2vc', 'generalwalk', 'dumpwalk', 'fixedpair',
                'simrank', 'node2vec']
    print ("The sampling method {} does not exist!", model)
    print ("Please choose from the following:")
    for m in model_list:
        print(m)
    exit()

def getmodels(g, args):

    model_v = getmodel(args.model_v, g, args)

    if args.model_v == 'rw2vc':
        return model_v

    # if not args.model_c:
    #     model_c = model_v
    # elif args.model_c == args.model_v:
    #     model_c = model_v
    # else:
    #     model_c = getmodel(args.model_c, g, args)

    if not args.emb_model:
        arg_emb_model = "asym"
    else:
        arg_emb_model = args.emb_model

    # trainer = vctrainer.vctrainer(g, model_v, emb_model=arg_emb_model, rep_size=args.representation_size,
    #                               epoch=args.epochs, batch_size=args.batch_size,
    #                               learning_rate=args.lr, negative_ratio=args.negative_ratio, emb_file=args.output)
    trainer = vctrainer_ops.vctrainer(g, model_v, emb_model=arg_emb_model, rep_size=args.representation_size,
                                  epoch=args.epochs, batch_size=args.batch_size,
                                  learning_rate=args.lr, negative_ratio=args.negative_ratio, emb_file=args.output,
                                  thread_num=args.thread_num)
    return trainer
