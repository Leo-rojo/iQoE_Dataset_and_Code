from each_user_function import each_user
import os
if __name__ == "__main__":
    import time
    from multiprocessing import Pool
    #if folder mq does not exist, create it
    if not os.path.exists('mq'):
        os.makedirs('mq')

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
                    main_path = reg + '_results_qn_' + str(n_queries) + '_nr_ch_' + str(nr_chunk)+'_'+str(1)
                    comb_of_par.append((nr_chunk, rs, u, m, reg, n_queries))

    print('param missing: '+ str(len(comb_of_par)))
                #else:
                    #print(main_path + '/' + model_name + '/user_' + str(u) + '/rmses')

    with Pool() as p:
        #p.map(each_user, [u for u in range(32)])
        p.starmap(each_user, comb_of_par)
    p.close()

