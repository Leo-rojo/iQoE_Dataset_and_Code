from each_user_function import each_user
import os
import numpy as np
if __name__ == "__main__":
    import time
    from multiprocessing import Pool
    #if folder mq does not exist, create it
    if not os.path.exists('../output_data/mq'):
        os.makedirs('../output_data/mq')
    if not os.path.exists('../output_data/time_over'):
        os.makedirs('../output_data/time_over')

    # params
    comb_of_par = []
    reg='SVR'#['RF','XGboost','SVR']
    n_queries = 250
    for nr_chunk in [7]:  # [2,4,8,
        for rs in [42]:
            #us=np.random.randint(0,32,3)
            for u in range(32):
                #ms = np.random.randint(0, 8, 3)
                for m in range(8):
                    model_name=['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas'][m]
                    main_path = '../output_data/'+reg + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunk)+'_'+str(1)
                    comb_of_par.append((nr_chunk, rs, u, m, reg, n_queries))

    print('param missing: '+ str(len(comb_of_par)))
                #else:
                    #print(main_path + '/' + model_name + '/user_' + str(u) + '/rmses')

    with Pool() as p:
        #p.map(each_user, [u for u in range(32)])
        p.starmap(each_user, comb_of_par)
    p.close()

    mod_names = ['bit', 'logbit', 'psnr', 'ssim', 'vmaf', 'FTW', 'SDNdash', 'videoAtlas']
    bymod_mean = []
    bymod_std = []
    allu = []
    for mod in mod_names:
        for u in range(32):
            eachu = []
            eachu.append(os.path.getsize("../output_data/mq/" + str(u) + mod + 'm_q' + 'initial' + '.pkl') / 1024)
            for nq in range(250):
                eachu.append(os.path.getsize("../output_data/mq/" + str(u) + mod + 'm_q' + str(nq) + '.pkl') / 1024)
            allu.append(eachu)

    bymod_mean.append(np.mean(allu, axis=0))
    bymod_std.append(np.std(allu, axis=0))
    np.save('../output_data/space_ave', bymod_mean)
    np.save('../output_data/space_std', bymod_std)

