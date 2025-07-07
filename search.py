# from Feb23_func import *
from Mar05_func import *
from Feb25_input import *

# 修改了stop criteria, 不提前停下
print('----------------------------------- MCMC settings --------------------------------------------')
# good_discard = [] # record the good discard sites [[m1,l1,m2,l2]]

groupNum = mcmcparas['groupNum']
iters = mcmcparas['iters']
subiter1 = mcmcparas['subiter1']; subiter2 = mcmcparas['subiter2']
subiter3 = mcmcparas['subiter3']; subiter4 = mcmcparas['subiter4']; subiter5 = mcmcparas['subiter5']
subiters = mcmcparas['subiters'] 

rep = mcmcparas['rep']; rep_in = mcmcparas['rep_in']
threadnum = mcmcparas['threadnum']
# 
jump_iters = mcmcparas['jump_iters']
sitenum = mcmcparas['sitenum']
stopsite = mcmcparas['stopsite']

test_iters = mcmcparas['test_iters']
test_subiter1 = mcmcparas['test_subiter1']; test_subiter2 = mcmcparas['test_subiter2']
test_subiter3 = mcmcparas['test_subiter3']; test_subiter4 = mcmcparas['test_subiter4']; test_subiter5 = mcmcparas['test_subiter5']
test_subiters = mcmcparas['test_subiters']

discard_iters = mcmcparas['discard_iters']
discard_subiter1 = mcmcparas['discard_subiter1']; discard_subiter2 = mcmcparas['discard_subiter2']
discard_subiter3 = mcmcparas['discard_subiter3']; discard_subiter4 = mcmcparas['discard_subiter4']; discard_subiter5 = mcmcparas['discard_subiter5']
discard_subiters = mcmcparas['discard_subiters']

# ChromSep(settings['filedir'],settings)
# part_num = Cut_chr_part(settings)

# part_num = 15
part_num = 1


def Updating(ziplist):


    datas,groups_result,blocks_class,chains,t,comparesite,knum = ziplist

    pairnum = int(knum*(knum-1)/2)
    Each_ms = chains['m_list'][t]; Each_ls = chains['l_list'][t] # 2D (knum or knum+1,subiters*iters+1)
    Each_ps = chains['p_list'][t]
    Each_posts = chains['post'][t] #1D 

    wethercheck = True

    # print("range of k: ",comparesite,min(comparesite+jump_iters,iters))
    # for k in range(comparesite,min(comparesite+jump_iters,iters)):
    for k in tqdm(range(comparesite, min(comparesite + jump_iters, iters)), desc="Processing"):
        start = time.time()
        for k1 in range(1,subiter1+1):
            kk = k*subiters+ (k1-1)+1
            # print('-------------',kth,'--------------')
            outcomeAll = update_group_each(datas,settings,knum,Each_ms[:,kk-1],Each_ls[:,kk-1],Each_ps[:,kk-1],Each_posts[kk-1],groups_result,wethercheck)
            Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk] = outcomeAll[0:3]
            Each_posts[kk] = outcomeAll[3]
            
            test_groups_result = copy.deepcopy(groups_result)
            test_block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
            test_groups_result.append(test_block_group)
            overlapflag = check_result_overlap(test_groups_result,True)
            if overlapflag == True:
                print('update_all overlap')
                print(k, t, kk,Each_ms[:,kk-1],Each_ls[:,kk-1],'UPDATE:',outcomeAll[0:2])
                sys.exit()

            X = countX_list(datas,knum, test_block_group)
            posterior_test = logposterior(datas,settings,knum,test_block_group,weightFlag=True)[0]
            if abs(posterior_test-Each_posts[kk])>200:
                with open(DebugName, 'a') as f:
                    f.write(f"update_all t={t}, kth={kk}, blocks: {test_block_group}, posterior={Each_posts[kk]}, prob={Each_ps[:,kk]}, {posterior_test}, Case1: {X[:-1]}\n")
            # print('update_all t=',t,', kth=',kk,', blocks: ',test_block_group,', posterior=',Each_posts[kk],posterior_test,', Case1: ',X[:-1])
            
        for k2 in range(1,subiter2+1):
            kk = k*subiters+subiter1+(k2-1)+1
            pth = random.choice(range(pairnum))
            outcomeL = update_L(datas,settings,knum,Each_ms[:,kk-1],Each_ls[:,kk-1],Each_ps[:,kk-1],Each_posts[kk-1],groups_result,wethercheck,pth)
            Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk] = outcomeL[0:3]
            Each_posts[kk] = outcomeL[3]
            test_groups_result = copy.deepcopy(groups_result)
            test_block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
            test_groups_result.append(test_block_group)
            overlapflag = check_result_overlap(test_groups_result,True)
            if overlapflag == True:
                print('update_L overlap')
                print(k, t, kk,Each_ms[:,kk-1],Each_ls[:,kk-1],'UPDATE:',outcomeL[0:2])
                sys.exit()
                # print('update_L: ', kk)
                # print('update_L t=',t,', k=',k,', kk=',kk,', m1=',Each_ms[:,kk],', l1=',Each_ls[:,kk],', m2=',Each_m2s[:,kk],', l2=',Each_l2s[:,kk],', p1=',Each_ps[:,kk],', p234=',Each_p234s[:,kk],', posterior=',Each_posts[kk])
            X = countX_list(datas,knum, test_block_group)
            posterior_test = logposterior(datas,settings,knum,test_block_group,weightFlag=True)[0]

            if abs(posterior_test-Each_posts[kk])>200:
                with open(DebugName, 'a') as f:
                    f.write(f"update_L t={t}, kth={kk}, blocks: {test_block_group}, posterior={Each_posts[kk]}, prob={Each_ps[:,kk]}, {posterior_test}, Case1: {X[:-1]}\n")
        for k3 in range(1,subiter3+1):
            kk = k*subiters+subiter1+subiter2+(k3-1)+1
            outcomeSK = update_expand_shrink_list(datas,settings,knum,Each_ms[:,kk-1],Each_ls[:,kk-1],Each_ps[:,kk-1],Each_posts[kk-1],groups_result,wethercheck)
            Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk]= outcomeSK[0:3]
            Each_posts[kk] = outcomeSK[3]
            # print('update_shrink: ', kk)
            # print('update_shrink t=',t,', k=',k,', kk=',kk,', m1=',Each_ms[:,kk],', l1=',Each_ls[:,kk],', m2=',Each_m2s[:,kk],', l2=',Each_l2s[:,kk],', p1=',Each_ps[:,kk],', p234=',Each_p234s[:,kk],', posterior=',Each_posts[kk])
            test_groups_result = copy.deepcopy(groups_result)
            test_block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
            test_groups_result.append(test_block_group)
            overlapflag = check_result_overlap(test_groups_result,True)
            if overlapflag == True:
                print('update_shrink overlap')
                print(k, t, kk,Each_ms[:,kk-1],Each_ls[:,kk-1],'UPDATE:',outcomeSK[0:2])
                sys.exit()

            X = countX_list(datas,knum, test_block_group)
            posterior_test = logposterior(datas,settings,knum,test_block_group,weightFlag=True)[0]

            if abs(posterior_test-Each_posts[kk])>200:
                with open(DebugName, 'a') as f:
                    f.write(f"update_expand_shrink_list t={t}, kth={kk}, blocks: {test_block_group}, posterior={Each_posts[kk]}, prob={Each_ps[:,kk]}, {posterior_test}, Case1: {X[:-1]}\n")
        for k4 in range(1,subiter4+1):
            kk = k*subiters+subiter1+subiter2+subiter3+(k4-1)+1
            Each_ms[:,kk] = Each_ms[:,kk-1].copy(); Each_ls[:,kk] = Each_ls[:,kk-1].copy()
            Each_ps[:,kk] = Each_ps[:,kk-1].copy(); Each_posts[kk] = Each_posts[kk-1].copy()
            for repp in range(2):
                for kth in range(knum):
                    outcomeBL = update_kth_block(datas,settings,knum,Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk],Each_posts[kk],groups_result,wethercheck,kth)
                    Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk]= outcomeBL[0:3]
                    Each_posts[kk] = outcomeBL[3]
                    test_groups_result = copy.deepcopy(groups_result)
                    test_block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
                    test_groups_result.append(test_block_group)
                    overlapflag = check_result_overlap(test_groups_result,True)
                    if overlapflag == True:
                        print('update_kth_block overlap')
                        print(k, t, kk,Each_ms[:,kk-1],Each_ls[:,kk-1],'UPDATE:',outcomeBL[0:2])
                        sys.exit()
            
            X = countX_list(datas,knum, test_block_group)
            posterior_test = logposterior(datas,settings,knum,test_block_group,weightFlag=True)[0]

            if abs(posterior_test-Each_posts[kk])>200:
                with open(DebugName, 'a') as f:
                    f.write(f"update_kth_block t={t}, kth={kk}, blocks: {test_block_group}, posterior={Each_posts[kk]}, prob={Each_ps[:,kk]}, {posterior_test}, Case1: {X[:-1]}\n")
            # print('update_kth_block: ', kk)
            # print('update_kth_block t=',t,', k=',k,', kk=',kk,', m1=',Each_ms[:,kk],', l1=',Each_ls[:,kk],', m2=',Each_m2s[:,kk],', l2=',Each_l2s[:,kk],', p1=',Each_ps[:,kk],', p234=',Each_p234s[:,kk],', posterior=',Each_posts[kk])
        for k5 in range(1,subiter5+1):
            kk = k*subiters+subiter1+subiter2+subiter3+subiter4+(k5-1)+1

            outcomeP = update_P_list(datas,settings,knum,Each_ms[:,kk-1],Each_ls[:,kk-1],Each_ps[:,kk-1],Each_posts[kk-1],groups_result)
            Each_ms[:,kk],Each_ls[:,kk],Each_ps[:,kk] = outcomeP[0:3]
            Each_posts[kk] = outcomeP[3]
            # print('update_P: ', kk)
            # print('update_P t=',t,', k=',k,', kk=',kk,', m1=',Each_ms[:,kk],', l1=',Each_ls[:,kk],', m2=',Each_m2s[:,kk],', l2=',Each_l2s[:,kk],', p1=',Each_ps[:,kk],', p234=',Each_p234s[:,kk],', posterior=',Each_posts[kk])
            test_groups_result = copy.deepcopy(groups_result)
            test_block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
            test_groups_result.append(test_block_group)
            overlapflag = check_result_overlap(test_groups_result,True)
            if overlapflag == True:
                print('update_P overlap')
                print(k, t, kk,Each_ms[:,kk-1],Each_ls[:,kk-1],'UPDATE:',outcomeP[0:2])
                sys.exit()
            X = countX_list(datas,knum, test_block_group)
            posterior_test = logposterior(datas,settings,knum,test_block_group,weightFlag=True)[0]
            if abs(posterior_test-Each_posts[kk])>200:
                with open(DebugName, 'a') as f:
                    f.write(f"update_P_list t={t}, kth={kk}, blocks: {test_block_group}, posterior={Each_posts[kk]}, prob={Each_ps[:,kk]}, {posterior_test}, Case1: {X[:-1]}\n")
        end = time.time()
        # print('time: ', end-start,kk)
        block_group = list_to_group(Each_ms[:,kk],Each_ls[:,kk])
        x_list = countX_list(datas,knum,block_group)
        # print(k,'test t: ',t,', paras:',Each_ms[:,kk],', ',Each_ls[:,kk],', ',Each_posts[kk],', Case1: ',x_list,overlapflag)
        # print('time: ', end-start)
        # print(k,t,x_list)
    Each_chains = {}
    Each_chains['m_list'] = Each_ms; Each_chains['l_list'] = Each_ls #2D (knum,subiters*iters+1)
    Each_chains['p_list'] = Each_ps #2D (knum+1,subiters*iters+1)
    Each_chains['post'] = Each_posts #1D (subiters*iters+1)

    # print('Updating---------------t: ',t_in,', comparesite',comparesite, Each_chains['post'])

    return Each_chains

def find_best_chain_in(input_chains,comparesite,printFlag,t):
    
    next_iter = min(subiters*iters,(comparesite+jump_iters)*subiters)
    max_coordinates = np.where(np.array(input_chains['post'])[:,(comparesite)*subiters:next_iter+1] == np.max(np.array(input_chains['post'])[:,(comparesite)*subiters:next_iter+1]))
    # if len(max_coordinates[0]) == 0:
    # np.savetxt('/home/project16/Blocks_Intra/A230404_IntraChrom/input_chains.txt', np.array(input_chains['post'])[:,(comparesite)*subiters:next_iter+1])
        
    # print('max_coordinates: ',t,max_coordinates)
    min_index = max_coordinates[0][np.argmin(max_coordinates[1])]

    best_chains_in = {}
    best_chains_in['m_list'] = input_chains['m_list'][min_index]
    best_chains_in['l_list'] = input_chains['l_list'][min_index]
    best_chains_in['p_list'] = input_chains['p_list'][min_index]
    best_chains_in['post'] = input_chains['post'][min_index]
    best_chains_in['startsite'] = input_chains['startsite'][min_index]
    best_chains_in['min_index'] = min_index

    # print(input_chains['post'].shape)
    # print(np.max(np.array(input_chains['post']),axis=0),min_index)
    return best_chains_in

def Get_new_chains_in(datas,settings,knum,groups_result,input_chains,comparesite,each_oldFlag,t):
 
    output_chains = input_chains.copy()
    best_chains_in = find_best_chain_in(input_chains,comparesite,False,t)

    min_index = best_chains_in['min_index']
    output_chains['min_index'] = min_index
    other_index = np.delete(np.arange(rep_in),min_index)  # 在一个rep中，有rep_in 条链（去进行mcmc jump）,除了最好的那条链的index:min_index，其他的链的index

    next_iter = min(subiters*iters,(comparesite+jump_iters)*subiters)

    # print('out1: ',output_chains['post'])
    for key in ['m_list','l_list','p_list']:
        output_chains[key] = np.array(output_chains[key]).copy()
        if each_oldFlag != 0.5:
            output_chains[key] = np.array(output_chains[key]).copy()
            output_chains[key][:,:,:]= np.array(output_chains[key])[int(each_oldFlag),:,:].copy()
            output_chains[key][min_index,:,next_iter] = best_chains_in[key][:,pd.Series(best_chains_in['m_list'][0]).dropna().index[-1]].copy()

        else:
            output_chains[key][min_index,:,next_iter] = best_chains_in[key][:,pd.Series(best_chains_in['m_list'][0]).dropna().index[-1]].copy()

    for key in ['post']:
        output_chains[key] = np.array(output_chains[key]).copy()
        if each_oldFlag != 0.5:
            output_chains[key] = np.array(output_chains[key]).copy()
            output_chains[key][:,:]= np.array(output_chains[key])[int(each_oldFlag),:].copy()
            output_chains[key][min_index,next_iter] = best_chains_in[key][pd.Series(best_chains_in['m_list'][0]).dropna().index[-1]].copy()

        else:
            output_chains[key][min_index,next_iter] = best_chains_in[key][pd.Series(best_chains_in['m_list'][0]).dropna().index[-1]].copy()


    output_chains['discard_chains'] = [] 
    for t_in in range(rep_in):

        discard_m_list,discard_l_list = np.array(input_chains['m_list'])[t_in,:,0:next_iter-1],np.array(input_chains['l_list'])[t_in,:,0:next_iter-1]
        discard_p_list = np.array(input_chains['p_list'])[t_in,:,0:next_iter-1]
        discard_post = np.array(input_chains['post'])[t_in,0:next_iter-1]

        discard_blocks = list_to_group(discard_m_list[:,-1],discard_l_list[:,-1])
        X_discard = countX_list(datas,knum,discard_blocks)
        if check_elements(X_discard,'all')==True:
            output_chains['discard_chains'].append([discard_m_list,discard_l_list,discard_p_list,discard_post])      


    for t_in in other_index:

        stopFlag, initial_group = generate_random_sample(datas,settings,knum,groups_result,extendFlag=True) 

        if stopFlag == True:
            output_chains['m_list'][t_in,:,next_iter] = output_chains['m_list'][min_index,:,next_iter-1]
            output_chains['l_list'][t_in,:,next_iter] = output_chains['l_list'][min_index,:,next_iter-1]
            output_chains['p_list'][t_in,:,next_iter] = output_chains['p_list'][min_index,:,next_iter-1]
            output_chains['post'][t_in][next_iter] = output_chains['post'][min_index][next_iter-1]
            output_chains['startsite'][t_in] = next_iter
        else:
            init_post,init_p_list = logposterior(datas,settings,knum,initial_group,weightFlag=True)

            init_m_list = [int((x[0]+x[1])/2 )for x in initial_group]
            init_l_list = [x[1]-x[0]+1 for x in initial_group]

            output_chains['m_list'][t_in,:,next_iter] = init_m_list; output_chains['l_list'][t_in,:,next_iter] = init_l_list
            output_chains['p_list'][t_in,:,next_iter] = init_p_list
            
            output_chains['post'][t_in][next_iter] = init_post
            output_chains['startsite'][t_in] = next_iter
            with open(DebugName, 'a') as f:
                f.write(f"=======generate_random_sample t and t_in={t,t_in}, blocks: {initial_group}, posterior={init_post}, prob={init_p_list},startsite={next_iter}\n")

    # print('out3: ',output_chains['post'])

    return output_chains

def testUpdateing(ziplist):

    datas,groups_result,candi_block_pair,t = ziplist
    pairnum = 1
    wethercheck = False

    test_chains = {}
    test_chains['m_list'] = np.nan * np.zeros((rep * rep_in, 2, test_subiters * test_iters + 1))
    test_chains['l_list'] = np.nan * np.zeros((rep * rep_in, 2, test_subiters * test_iters + 1))
    test_chains['p_list'] = np.nan * np.zeros((rep * rep_in, 1 + 1, test_subiters * test_iters + 1))
    test_chains['post'] = np.full((rep * rep_in, test_subiters * test_iters + 1), np.NINF)

    init_m_list = [int((x[0] + x[1]) / 2) for x in candi_block_pair]
    init_l_list = [x[1] - x[0] + 1 for x in candi_block_pair]
    init_post, init_p_list = logposterior(datas,settings,2,candi_block_pair,weightFlag=True)

    test_chains['m_list'][t, :, 0] = init_m_list
    test_chains['l_list'][t, :, 0] = init_l_list
    test_chains['p_list'][t, :, 0] = init_p_list
    test_chains['post'][t, 0] = init_post
    test_init_X = countX_list(datas,2,candi_block_pair)
    # print('test candidate for t = ', t, ' : m1=', init_m_list, ',l1=', init_l_list, ',post=', '%.3f' % init_post, ', Case1 = ', ['%.f' % x for x in test_init_X[:-1]], ', iters=', test_iters)

    test_Each_ms = test_chains['m_list'][t]; test_Each_ls = test_chains['l_list'][t]
    test_Each_ps = test_chains['p_list'][t]
    test_Each_posts = test_chains['post'][t] 

    for k in tqdm(range(test_iters), desc="Test Processing"):
        start = time.time()
        if test_subiter1 != 0:
            print(f'test_subiter1 for update_group_each must be 0, but now is {test_subiter1}')
        for k1 in range(1,test_subiter1+1):
            kk = k*test_subiters+ (k1-1)+1
            outcomeAll = update_group_each(datas,settings,2,test_Each_ms[:,kk-1],test_Each_ls[:,kk-1],test_Each_ps[:,kk-1],test_Each_posts[kk-1],groups_result,wethercheck)
            test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk] = outcomeAll[0:3]
            test_Each_posts[kk] = outcomeAll[3]
        for k2 in range(1,test_subiter2+1):
            kk = k*test_subiters+test_subiter1+(k2-1)+1
            pth = random.choice(range(pairnum))
            outcomeL = update_L(datas,settings,2,test_Each_ms[:,kk-1],test_Each_ls[:,kk-1],test_Each_ps[:,kk-1],test_Each_posts[kk-1],groups_result,wethercheck,pth)
            test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk] = outcomeL[0:3]
            test_Each_posts[kk] = outcomeL[3]
        for k3 in range(1,test_subiter3+1):
            kk = k*test_subiters+test_subiter1+test_subiter2+(k3-1)+1
            outcomeSK = update_expand_shrink_list(datas,settings,2,test_Each_ms[:,kk-1],test_Each_ls[:,kk-1],test_Each_ps[:,kk-1],test_Each_posts[kk-1],groups_result,wethercheck)
            test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk]= outcomeSK[0:3]
            test_Each_posts[kk] = outcomeSK[3]

        for k4 in range(1,test_subiter4+1):
            kk = k*test_subiters+test_subiter1+test_subiter2+test_subiter3+(k4-1)+1
            test_Each_ms[:,kk] = test_Each_ms[:,kk-1].copy(); test_Each_ls[:,kk] = test_Each_ls[:,kk-1].copy()
            test_Each_ps[:,kk] = test_Each_ps[:,kk-1].copy(); test_Each_posts[kk] = test_Each_posts[kk-1].copy()
            for repp in range(1):
                for kth in range(2):
                    outcomeBL = update_kth_block(datas,settings,2,test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk],test_Each_posts[kk],groups_result,wethercheck,kth)
                    test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk]= outcomeBL[0:3]
                    test_Each_posts[kk] = outcomeBL[3]
        for k5 in range(1,test_subiter5+1):
            kk = k*test_subiters+test_subiter1+test_subiter2+test_subiter3+test_subiter4+(k5-1)+1

            outcomeP = update_P_list(datas,settings,2,test_Each_ms[:,kk-1],test_Each_ls[:,kk-1],test_Each_ps[:,kk-1],test_Each_posts[kk-1],groups_result)
            test_Each_ms[:,kk],test_Each_ls[:,kk],test_Each_ps[:,kk] = outcomeP[0:3]
            test_Each_posts[kk] = outcomeP[3]

        # update_test_block_group = list_to_group(test_Each_ms[:,kk],test_Each_ls[:,kk])
        # x_list = countX_list(datas,2,update_test_block_group)
        
    test_Each_chains = {}
    test_Each_chains['m_list'] = test_Each_ms; test_Each_chains['l_list'] = test_Each_ls 
    test_Each_chains['p_list'] = test_Each_ps 
    test_Each_chains['post'] = test_Each_posts 

    return test_Each_chains

def testMulti(datas,groups_result,candi_block_pair):

    start_test = time.time()

    test_Multi = mp.Pool(threadnum,maxtasksperchild=500)
    test_ziplist = zip([datas]*rep*rep_in,[groups_result]*rep*rep_in,[candi_block_pair]*rep*rep_in,range(rep*rep_in))
    test_result = test_Multi.map(testUpdateing, test_ziplist)
    test_Multi.terminate() 

    test_chains = {}

    test_chains['m_list'] = [x['m_list'] for x in test_result]
    test_chains['l_list'] = [x['l_list'] for x in test_result]
    test_chains['p_list'] = [x['p_list'] for x in test_result]
    test_chains['post'] = [x['post'] for x in test_result] 
    # best_paras = {'m_list':np.zeros((rep,2)),'l_list':np.zeros((rep,2)),'p_list':np.zeros((rep,1+1)),'post':np.nan*(np.zeros(rep))}
    # x_est = np.zeros((rep,1+1))

    test_max_post = np.zeros(rep); test_max_m_list = np.zeros((rep,2)); test_max_l_list = np.zeros((rep,2)); test_max_p_list = np.zeros((rep,1+1))

    for t in range(rep):


        input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in test_chains.items() if isinstance(v, list)}

        max_coordinates = np.where(np.array(input_chains['post']) == np.max(np.array(input_chains['post'])))
        min_index = max_coordinates[0][np.argmin(max_coordinates[1])]

        best_chains_in = {}
        best_chains_in['m_list'] = input_chains['m_list'][min_index]
        best_chains_in['l_list'] = input_chains['l_list'][min_index]
        best_chains_in['p_list'] = input_chains['p_list'][min_index]
        best_chains_in['post'] = input_chains['post'][min_index]
        best_chains_in['min_index'] = min_index

    # test_s_t,test_e_t= np.zeros((rep,2)),np.zeros((rep,2))
    test_X_t = np.zeros((rep,1+1))
    for t in range(rep):
        input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in test_chains.items() if isinstance(v, list)}

        
        test_max_post[t] = max(best_chains_in['post'])
        maxindex_in = list(best_chains_in['post']).index(test_max_post[t])
        test_max_m_list[t] = best_chains_in['m_list'][:,maxindex_in]; test_max_l_list[t] = best_chains_in['l_list'][:,maxindex_in]
        test_max_p_list[t] = best_chains_in['p_list'][:,maxindex_in]

        # test_s_t[t] = [test_max_m_list[t,i] - math.floor((test_max_l_list[t,i] - 1) / 2) for i in range(len(test_max_m_list[t]))]
        # test_e_t[t] = [test_max_m_list[t,i] + math.ceil((test_max_l_list[t,i] - 1) / 2) for i in range(len(test_max_m_list[t]))]

        test_block_group= list_to_group(test_max_m_list[t],test_max_l_list[t])
        test_X_t[t] = countX_list(datas,2,test_block_group)
        print('Each candidate best, t: ',t,', m: ',test_max_m_list[t],', l: ',test_max_l_list[t],', p1: ',['%.3e' % x for x in test_max_p_list[t][:-1]],', p234: ','%.3e'%test_max_p_list[t][-1],', post: ','%.3f'%test_max_post[t],', Case1: ',test_X_t[t][:-1])
    end_test = time.time()
    timeuse = end_test - start_test
    test_max_post_allrep = max(test_max_post)
    maxindex = list(test_max_post).index(test_max_post_allrep )

    m1,l1,m2,l2 = test_max_m_list[maxindex][0],test_max_l_list[maxindex][0],test_max_m_list[maxindex][1],test_max_l_list[maxindex][1]
    locs = FindLocation(m1,l1,m2,l2)
    readsnum1_shrink,readsnum2_shrink = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=True,expandFlag=True)
    readsnum1,readsnum2 = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=False,expandFlag=True)

    # chrom1 = 'chr21'; chrom2 = 'chr21'
    # chrom1 = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]; chrom2 = chrom1
    IAB = test_X_t[maxindex][0]
    pvalue_shrink = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1_shrink,readsnum2_shrink,IAB)
    pvalue = getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1,readsnum2,IAB)
    # with open(settings['outdir'] + f'/Final_Loops_test_{knum}.bedpe','a') as outfile:
    #     outfile.write(f"{chrom1}\t{int(locs[0])}\t{int(locs[1])}\t{chrom2}\t{int(locs[2])}\t{int(locs[3])}\t{int(IAB)}\t{int(readsnum1)}\t{int(readsnum2)}\t{pvalue:.2e}\t{int(readsnum1_shrink)}\t{int(readsnum2_shrink)}\t{pvalue_shrink:.2e}\t{test_max_post_allrep}\t{testname}\t{(pvalue<0.05)and(IAB>=2)}\t{datas['filename']}\t{timeuse}s")
    #     outfile.write('\n')

    return locs,IAB,readsnum1,readsnum2,pvalue,readsnum1_shrink,readsnum2_shrink,pvalue_shrink,test_max_post_allrep,timeuse

def recordCandidate(datas,settings,knum,groups_result,blocks_class,blocks_class_candicate,candidate_loops_knum2,p,maxindex,s_t,e_t,max_post_allrep,strongPvalueFlag,foundLinksNum,timeuse,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post):
    testNum = len(candidate_loops_knum2)
    if testNum==0:
        write2Final_Loops(datas,settings,knum,'Final_Loops',p,maxindex,s_t[maxindex],e_t[maxindex],max_post_allrep,foundLinksNum,timeuse)
        blocks_class = blocks_class_candicate.copy()
        savetxt(knum,mcmcparas,settings['outdir'],p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,foundLinksNum)

        print('No candidate loops, continue to find next loop!')
    else:
        print('candidate_loops_knum2:',candidate_loops_knum2,' testNum:',testNum)

        test_locs = np.zeros((testNum,4))
        test_IAB = np.zeros(testNum)
        test_readsnum1 = np.zeros(testNum)
        test_readsnum2 = np.zeros(testNum)
        test_pvalues = np.zeros(testNum)
        test_readsnum1_shrink = np.zeros(testNum)
        test_readsnum2_shrink = np.zeros(testNum)
        test_pvalues_shrink = np.zeros(testNum)
        test_max_post_allrep = np.zeros(testNum)
        test_timeuse = np.zeros(testNum)
        test_name = np.zeros(testNum)

        for idx,candi_block_pair in enumerate(candidate_loops_knum2):
            print('verify: ',idx,candi_block_pair)
            test_name[idx] = f'{p}_{idx}'
            test_locs[idx],test_IAB[idx],test_readsnum1[idx],test_readsnum2[idx],test_pvalues[idx],test_readsnum1_shrink[idx],test_readsnum2_shrink[idx],test_pvalues_shrink[idx],test_max_post_allrep[idx],test_timeuse[idx] = testMulti(datas,groups_result,candi_block_pair)                

        pvalue_mean = harmonic_mean_p(test_pvalues)
        if pvalue_mean < 0.05 or strongPvalueFlag:
            write2Final_Loops(datas,settings,knum,'Final_Loops',p,maxindex,s_t[maxindex],e_t[maxindex],max_post_allrep,foundLinksNum,timeuse)
            blocks_class = blocks_class_candicate.copy()
            print('pvalue_mean: ',pvalue_mean,'test_pvalues ',test_pvalues,'so write into Final_Loops')
            savetxt(knum,mcmcparas,settings['outdir'],p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,foundLinksNum)

            chrom1 = 'chr21'; chrom2 = 'chr21'
            for idx in range(testNum):
                locs = test_locs[idx]
                IAB = test_IAB[idx]
                readsnum1 = test_readsnum1[idx]
                readsnum2 = test_readsnum2[idx]
                pvalue = test_pvalues[idx]
                readsnum1_shrink = test_readsnum1_shrink[idx]
                readsnum2_shrink = test_readsnum2_shrink[idx]
                pvalue_shrink = test_pvalues_shrink[idx]
                test_max_post_allrep = test_max_post_allrep[idx]
                timeuse = test_timeuse[idx]
                testname = test_name[idx]
                with open(settings['outdir'] + f'/Final_Loops_test_{knum}.bedpe','a') as outfile:
                    outfile.write(f"{chrom1}\t{int(locs[0])}\t{int(locs[1])}\t{chrom2}\t{int(locs[2])}\t{int(locs[3])}\t{int(IAB)}\t{int(readsnum1)}\t{int(readsnum2)}\t{pvalue:.2e}\t{int(readsnum1_shrink)}\t{int(readsnum2_shrink)}\t{pvalue_shrink:.2e}\t{test_max_post_allrep}\t{testname}\t{(pvalue<0.05)and(IAB>=2)}\t{datas['filename']}\t{timeuse}s")
                    outfile.write('\n')
        else:
            write2Final_Loops(datas,settings,knum,'Final_Loops_ambiguous',p,maxindex,s_t[maxindex],e_t[maxindex],max_post_allrep,foundLinksNum,timeuse)
            print('pvalue_mean: ',pvalue_mean,'test_pvalues ',test_pvalues,'so write into Final_Loops_ambiguous')

    return blocks_class
        



def DiscardUpdateing(ziplist):

    datas,groups_result,discard_chain_one,t = ziplist

    init_m_list = discard_chain_one[0][:,-1]
    init_l_list = discard_chain_one[1][:,-1]
    init_p_list = discard_chain_one[2][:,-1]
    init_post = discard_chain_one[3][-1]

    # print('DiscardUpdateing: ',init_m_list,init_l_list,init_p_list,init_post)

    knum = len(init_m_list)
    pairnum = int(knum*(knum-1)/2)
    wethercheck = True


    discard_chains = {}
    discard_chains['m_list'] = np.nan * np.zeros((rep * rep_in, knum, discard_subiters * discard_iters + 1))
    discard_chains['l_list'] = np.nan * np.zeros((rep * rep_in, knum, discard_subiters * discard_iters + 1))
    discard_chains['p_list'] = np.nan * np.zeros((rep * rep_in, pairnum + 1, discard_subiters * discard_iters + 1))
    discard_chains['post'] = np.full((rep * rep_in, discard_subiters * discard_iters + 1), np.NINF)

    discard_chains['m_list'][t, :, 0] = init_m_list
    discard_chains['l_list'][t, :, 0] = init_l_list
    discard_chains['p_list'][t, :, 0] = init_p_list
    discard_chains['post'][t, 0] = init_post

    # discard_block_pair = list_to_group(init_m_list,init_l_list)
    # discard_init_X = countX_list(datas,knum,discard_block_pair)
    # print('test candidate for t = ', t, ' : m1=', init_m_list, ',l1=', init_l_list, ',post=', '%.3f' % init_post, ', Case1 = ', ['%.f' % x for x in discard_init_X[:-1]], ', iters=', discard_iters)

    discard_Each_ms = discard_chains['m_list'][t]; discard_Each_ls = discard_chains['l_list'][t]
    discard_Each_ps = discard_chains['p_list'][t]
    discard_Each_posts = discard_chains['post'][t] 

    for k in tqdm(range(discard_iters), desc="Discarding Processing"):
        start = time.time()
        for k1 in range(1,discard_subiter1+1):
            kk = k*discard_subiters+ (k1-1)+1
            outcomeAll = update_group_each(datas,settings,knum,discard_Each_ms[:,kk-1],discard_Each_ls[:,kk-1],discard_Each_ps[:,kk-1],discard_Each_posts[kk-1],groups_result,wethercheck)
            discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk] = outcomeAll[0:3]
            discard_Each_posts[kk] = outcomeAll[3]
        for k2 in range(1,discard_subiter2+1):
            kk = k*discard_subiters+discard_subiter1+(k2-1)+1
            pth = random.choice(range(pairnum))
            outcomeL = update_L(datas,settings,knum,discard_Each_ms[:,kk-1],discard_Each_ls[:,kk-1],discard_Each_ps[:,kk-1],discard_Each_posts[kk-1],groups_result,wethercheck,pth)
            discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk] = outcomeL[0:3]
            discard_Each_posts[kk] = outcomeL[3]
        for k3 in range(1,discard_subiter3+1):
            kk = k*discard_subiters+discard_subiter1+discard_subiter2+(k3-1)+1
            outcomeSK = update_expand_shrink_list(datas,settings,knum,discard_Each_ms[:,kk-1],discard_Each_ls[:,kk-1],discard_Each_ps[:,kk-1],discard_Each_posts[kk-1],groups_result,wethercheck)
            discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk]= outcomeSK[0:3]
            discard_Each_posts[kk] = outcomeSK[3]

        for k4 in range(1,discard_subiter4+1):
            kk = k*discard_subiters+discard_subiter1+discard_subiter2+discard_subiter3+(k4-1)+1
            discard_Each_ms[:,kk] = discard_Each_ms[:,kk-1].copy(); discard_Each_ls[:,kk] = discard_Each_ls[:,kk-1].copy()
            discard_Each_ps[:,kk] = discard_Each_ps[:,kk-1].copy(); discard_Each_posts[kk] = discard_Each_posts[kk-1].copy()
            for repp in range(1):
                for kth in range(knum):
                    outcomeBL = update_kth_block(datas,settings,knum,discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk],discard_Each_posts[kk],groups_result,wethercheck,kth)
                    discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk]= outcomeBL[0:3]
                    discard_Each_posts[kk] = outcomeBL[3]
        for k5 in range(1,discard_subiter5+1):
            kk = k*discard_subiters+discard_subiter1+discard_subiter2+discard_subiter3+discard_subiter4+(k5-1)+1

            outcomeP = update_P_list(datas,settings,knum,discard_Each_ms[:,kk-1],discard_Each_ls[:,kk-1],discard_Each_ps[:,kk-1],discard_Each_posts[kk-1],groups_result)
            discard_Each_ms[:,kk],discard_Each_ls[:,kk],discard_Each_ps[:,kk] = outcomeP[0:3]
            discard_Each_posts[kk] = outcomeP[3]

        # update_test_block_group = list_to_group(discard_Each_ms[:,kk],discard_Each_ls[:,kk])
        # x_list = countX_list(datas,knum,update_test_block_group)
        
    test_Each_chains = {}
    test_Each_chains['m_list'] = discard_Each_ms; test_Each_chains['l_list'] = discard_Each_ls 
    test_Each_chains['p_list'] = discard_Each_ps 
    test_Each_chains['post'] = discard_Each_posts 

    return test_Each_chains


def testDiscard(datas,groups_result,discard_chain_one,knum,p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post):

    discard_Multi = mp.Pool(threadnum,maxtasksperchild=500)
    discard_ziplist = zip([datas]*rep*rep_in,[groups_result]*rep*rep_in,[discard_chain_one]*rep*rep_in,range(rep*rep_in))
    discard_result = discard_Multi.map(DiscardUpdateing, discard_ziplist)
    discard_Multi.terminate() 

    discard_chains = {}

    discard_chains['m_list'] = [x['m_list'] for x in discard_result]
    discard_chains['l_list'] = [x['l_list'] for x in discard_result]
    discard_chains['p_list'] = [x['p_list'] for x in discard_result]
    discard_chains['post'] = [x['post'] for x in discard_result] 
    # best_paras = {'m_list':np.zeros((rep,2)),'l_list':np.zeros((rep,2)),'p_list':np.zeros((rep,1+1)),'post':np.nan*(np.zeros(rep))}
    # x_est = np.zeros((rep,1+1))

    # discard_max_post = np.zeros(rep); discard_max_m_list = np.zeros((rep,knum)); discard_max_l_list = np.zeros((rep,knum)); discard_max_p_list = np.zeros((rep,pairnum+1))

    for t in range(rep):


        input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in discard_chains.items() if isinstance(v, list)}

        max_coordinates = np.where(np.array(input_chains['post']) == np.max(np.array(input_chains['post'])))
        min_index = max_coordinates[0][np.argmin(max_coordinates[1])]

        best_chains_in = {}
        best_chains_in['m_list'] = input_chains['m_list'][min_index]
        best_chains_in['l_list'] = input_chains['l_list'][min_index]
        best_chains_in['p_list'] = input_chains['p_list'][min_index]
        best_chains_in['post'] = input_chains['post'][min_index]
        best_chains_in['min_index'] = min_index

    s_t,e_t= np.zeros((rep,knum)),np.zeros((rep,knum))
    for t in range(rep):
        input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in discard_chains.items() if isinstance(v, list)}
        m_list[p+1][t][:, :best_chains_in['m_list'].shape[1]] = best_chains_in['m_list']
        l_list[p+1][t][:, :best_chains_in['l_list'].shape[1]] = best_chains_in['l_list']
        p_list[p+1][t][:, :best_chains_in['p_list'].shape[1]] = best_chains_in['p_list']
        posterior[p+1][t][:best_chains_in['post'].shape[0]] = best_chains_in['post']
        
        max_post[p+1][t] = max(best_chains_in['post'])
        maxindex_in = list(best_chains_in['post']).index(max_post[p+1][t])
        max_m_list[p+1][t] = best_chains_in['m_list'][:,maxindex_in]; max_l_list[p+1][t] = best_chains_in['l_list'][:,maxindex_in]
        max_p_list[p+1][t] = best_chains_in['p_list'][:,maxindex_in]

        s_t[t] = [max_m_list[p+1,t,i] - math.floor((max_l_list[p+1,t,i] - 1) / 2) for i in range(len(max_m_list[p+1][t]))]
        e_t[t] = [max_m_list[p+1,t,i] + math.ceil((max_l_list[p+1,t,i] - 1) / 2) for i in range(len(max_m_list[p+1][t]))]


    return m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,s_t,e_t
    

















def MultiprocessP():

    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write('\n')
        f.write('-----------start to find loops------------' + '\n')
        f.write('\n')

    part_flag = np.full(part_num,True)
    randnum = 0
    # files = os.listdir(settings['outdir']+'/ChromSep.part')    # 这里simulation 不太需要
    # file_path = os.path.join(settings['outdir'],'ChromSep.part', files[randnum])
    file_path = './input/simu4_multi.bedpe'


    startfiletime = time.time()
    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write('----------------initial file_path: '+ file_path + '---------------'+'\n')
    print('initial file_path: ',file_path)
    
    datas = ConstructDatas(file_path, settings); groups_result = []; blocks_class = []

    # groups_result = [[[25711000,25716000],[25950000,25955000],[26050000,26056000]],
    #                  [[26600000,26609000],[26821584,26827000],[26890000,26894000]]
    #                 ]
    # blocks_class = [[[25711000,25716000],[25950000,25955000],[26050000,26056000]],
    #                  [[26600000,26609000],[26821584,26827000],[26890000,26894000]]
    #                 ]


    knum_list = [3,2]

    for knum_idex,current_knum in enumerate(knum_list):
        inittxt(settings['outdir'],current_knum)

    for knum_idex,current_knum in enumerate(knum_list):
        knum = current_knum
        pairnum = int(knum * (knum - 1) / 2)
        posterior,m_list,l_list,max_post,max_m_list,max_l_list,max_p_list,p_list,foundLinksNum = ConstructOutput(knum,mcmcparas)

        part_flag = np.full(part_num,True)
        p = 0
        weak_zero_flag = 0
        while p < groupNum:
            print('part_flag: ',part_flag)
            if part_flag[randnum] == False:
                file_loop_num = len(groups_result)*pairnum
                endfiletime = time.time()
                with open(settings['outdir']+'/worklog.txt', 'a') as f:
                    f.write('knum={}, number of loops: {}\n'.format(knum,file_loop_num))
                    f.write(f'time of this file: {endfiletime - startfiletime:.2f} seconds\n')
                print(f'knum={knum}, number of loops={file_loop_num}')
                print(f'time of this file: ,{endfiletime-startfiletime}')
                # os.remove(settings['outdir']+'/ChromSep.part/'+ files[randnum])
                print('--------------------------------------------All links of this dataset are found---------------------------------------------')


                if randnum < len(part_flag)-1:
                    randnum = randnum+1
                    # file_path = os.path.join(settings['outdir'],'ChromSep.part', files[randnum])
                    # file_path = '/home/project16/Blocks_simu/Bern_simu_test/simuMulti5/ChromSep.part/simuMulti5_chr21_part1.bedpe'
                    print('wrong')
                    startfiletime = time.time()
                    with open(settings['outdir']+'/worklog.txt', 'a') as f:
                        f.write('-----------------next file_path: '+ file_path + '-----------------' + '\n')
                    print('next file_path: ',file_path)

                    datas = ConstructDatas(file_path,settings); groups_result = []
                    stopFlag, chains = Initial2Chains(datas,settings,knum,mcmcparas,groups_result)
                    if stopFlag == True:
                        break
                    print('---------initial done--------------')
                else:
                    print('--------------------------------------------totally Found ',NumLoops,' Loops---------------------------------------------')
                    print('--------------------------------------------all files done----------------------------------------------------------')
                    break
            else:
                stopFlag, chains = Initial2Chains(datas,settings,knum,mcmcparas,groups_result)
                print('Initial2Chains done, stopFlag is:',stopFlag)

            if stopFlag == False:
                count_jump = np.zeros(rep)
                old_chain_Flag = np.repeat(0.5,rep)
                comparesite = 0

                with open(DebugName, 'a') as f:
                    f.write(f'=============start p={p} ===========\n')

                all_equalFlag1 = 0; all_similarFlag2 = 0

                start_in = time.time()
                # for site in tqdm(range(sitenum)):
                for site in range(sitenum):
                    Multi = mp.Pool(threadnum,maxtasksperchild=500)
                    ziplist = zip([datas]*rep*rep_in,[groups_result]*rep*rep_in,[blocks_class]*rep*rep_in,[chains]*rep*rep_in,range(rep*rep_in),[comparesite]*rep*rep_in,[knum]*rep*rep_in)
                    result = Multi.map(Updating, ziplist)
                    Multi.terminate() 

                    chains['m_list'] = [x['m_list'] for x in result]
                    chains['l_list'] = [x['l_list'] for x in result]
                    chains['p_list'] = [x['p_list'] for x in result]
                    chains['post'] = [x['post'] for x in result]
                    new_chains = chains.copy()

                    best_paras = {'m_list':np.zeros((rep,knum)),'l_list':np.zeros((rep,knum)),'p_list':np.zeros((rep,pairnum+1)),'post':np.nan*(np.zeros(rep))}
                    x_est = np.zeros((rep,pairnum+1))

                    for t in range(rep):


                        input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in chains.items() if isinstance(v, list)}
                        input_chains['startsite'] = chains['startsite'][t*rep_in:(t+1)*rep_in]

                        output_chains= Get_new_chains_in(datas,settings,knum,groups_result,input_chains,comparesite,old_chain_Flag[t],t)
                        for key, value in new_chains.items():
                            new_chains[key][t*rep_in:(t+1)*rep_in]= output_chains[key]
                        new_chains['min_index'][t] = output_chains['min_index']

                        [datas['discard_chains'].append(x) for x in output_chains['discard_chains']]
                        new_chain_Flag = int(new_chains['min_index'][t])
                        if new_chain_Flag != old_chain_Flag[t]:
                            count_jump[t] += 1
                            old_chain_Flag[t] = new_chain_Flag
                        # if want to see temp result: 
                        best_chains_in = find_best_chain_in(input_chains,comparesite,False,t)
                        posterior[p][t] = best_chains_in['post']
                        m_list[p][t] = best_chains_in['m_list']
                        l_list[p][t] = best_chains_in['l_list']


                        best_index = list(best_chains_in['post']).index(max(best_chains_in['post']))
                        best_paras['m_list'][t] = best_chains_in['m_list'][:,best_index]
                        best_paras['l_list'][t] = best_chains_in['l_list'][:,best_index]
                        best_paras['p_list'][t] = best_chains_in['p_list'][:,best_index]
                        best_paras['post'][t] = best_chains_in['post'][best_index]
                        block_group= list_to_group(best_paras['m_list'][t],best_paras['l_list'][t])
                        x_est[t] = countX_list(datas,knum,block_group)
                        
                        last_paras_index = len([x for x in best_chains_in['post'] if x != -float('inf')])-1
                        last_paras = {}
                        last_paras['m_list'] = best_chains_in['m_list'][:,last_paras_index]
                        last_paras['l_list'] = best_chains_in['l_list'][:,last_paras_index]
                        last_paras['p_list'] = best_chains_in['p_list'][:,last_paras_index]
                        last_paras['post'] = best_chains_in['post'][last_paras_index]
                        block_group= list_to_group(last_paras['m_list'],last_paras['l_list'])
                        x_est[t] = countX_list(datas,knum,block_group)
                        # 这是记录parallel mcmc每个可能跳跃的地方的结果
                        print('site: ',site, 'temp best, t: ',t,', paras:',last_paras['m_list'],', ',last_paras['l_list'],', ',last_paras['post'],', Case1: ',x_est[t][:-1])
                        # print(len(datas['indexList']),len(datas['indexList_list']),stopFlag)
                    
                    chains = new_chains
                    comparesite += jump_iters

                    # rhat
                    rhats = np.full(knum,np.inf)
                    for i in range(knum):
                        rhats[i] = R_hat(m_list[p][:,i,:],10)
                    print('R hat: ',rhats)
                    if np.all(rhats < 1.1):
                        print('MCMC can stop at site: ',site,', because of R hat.')
                        stopsite[p] = comparesite*subiters 
                        break
                    
                    # rhat part
                    if site >= int(sitenum*2):
                        Rhat_part_Flags = np.full(knum,np.inf)
                        for i in range(knum):
                            Rhat_part_Flags[i] = R_hat_part_flag(m_list[p][:,i,:],10)
                        # print('R hat part: ',Rhat_part_Flags)
                        if np.all(Rhat_part_Flags):
                            print('MCMC can stop at site: ',site,', because of R hat part.')
                            stopsite[p] = comparesite*subiters 
                            break

                    similarFlag2 = (all_similar(lst=best_paras['post'],err=2))&(np.all(x_est[:,:-1] >= settings['minLinks']))
                    if similarFlag2:
                        all_similarFlag2 += 1
                    else:
                        all_similarFlag2 = 0

                    if all_similarFlag2 == 3:
                        print('MCMC can stop at site: ',site,', because of similar likelihood.')
                        stopsite[p] = comparesite*subiters 
                        break

                        
                s_t,e_t= np.zeros((rep,knum)),np.zeros((rep,knum))
                X_t = np.zeros((rep,pairnum+1))
                for t in range(rep):
                    input_chains = {k: v[t*rep_in:(t+1)*rep_in] for k, v in chains.items() if isinstance(v, list)}
                    input_chains['startsite'] = chains['startsite'][t*rep_in:(t+1)*rep_in]
                    best_chains_in = find_best_chain_in(input_chains,comparesite,False,t)
                    m_list[p][t] = best_chains_in['m_list']
                    l_list[p][t] = best_chains_in['l_list']
                    p_list[p][t] = best_chains_in['p_list']
                    posterior[p][t] = best_chains_in['post']
                    Each_startsite = best_chains_in['startsite']
                    
                    max_post[p][t] = max(best_chains_in['post'])
                    maxindex_in = list(best_chains_in['post']).index(max_post[p][t])
                    max_m_list[p][t] = best_chains_in['m_list'][:,maxindex_in]; max_l_list[p][t] = best_chains_in['l_list'][:,maxindex_in]
                    max_p_list[p][t] = best_chains_in['p_list'][:,maxindex_in]

                    s_t[t] = [max_m_list[p,t,i] - math.floor((max_l_list[p,t,i] - 1) / 2) for i in range(len(max_m_list[p][t]))]
                    e_t[t] = [max_m_list[p,t,i] + math.ceil((max_l_list[p,t,i] - 1) / 2) for i in range(len(max_m_list[p][t]))]

                    block_group= list_to_group(max_m_list[p][t],max_l_list[p][t])
                    X_t[t] = countX_list(datas,knum,block_group)
                    print('Eachbest: ', 'search: ',p, ', rep: ', t,', m: ',max_m_list[p][t],', l: ',max_l_list[p][t],', p1: ',['%.3e' % x for x in max_p_list[p][t][:-1]],', p234: ','%.3e'%max_p_list[p][t][-1],', post: ','%.3f'%max_post[p][t],', Case1: ',X_t[t][:-1],', jump_rate: ',(count_jump[t]-1)/site)
                    foundLinksNum[p][t] = X_t[t][:-1]
                
                end_in = time.time()
                timeuse = end_in - start_in

            
                max_post_allrep = max(max_post[p])
                maxindex = list(max_post[p]).index(max_post_allrep)
                        

                pvalues_maxindex = []; pvalue_pair_index=0
                for i in range(knum-1):
                    for j in range(i+1,knum):
                        m1,l1,m2,l2 = max_m_list[p,maxindex,i],max_l_list[p,maxindex,i],max_m_list[p,maxindex,j],max_l_list[p,maxindex,j]
                        readsnum1,readsnum2 = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=False,expandFlag=True)
                        IAB = X_t[maxindex][pvalue_pair_index]
                        pvalue_pair_index += 1
                        pvalues_maxindex.append(getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1,readsnum2,IAB))

                strongPvalueFlag = np.all(np.array(pvalues_maxindex) < 1e-4)

                write_flag = True
                if knum >= 3:
                    X_maxindex = X_t[maxindex][:-1]
                    if np.count_nonzero(np.array(X_maxindex)<=2)>=2:
                        write_flag = False
                        weak_zero_flag += 1
                    #     write_flag = False
                    #     weak_zero_flag += 1

                    print(f'========== weak_zero_flag: {weak_zero_flag} ==========')

                
                if (np.max(foundLinksNum[p][0:-1])<=settings['minLinks']-1) & (np.max(foundLinksNum[max(0,p-1)][0:-1])<=settings['minLinks']-1):
                    part_flag[randnum] = False
                    continue

                elif weak_zero_flag >= 1:
                    part_flag[randnum] = False
                    continue

                elif knum == 2:
                    datas['permuTestPost'].append(max_post_allrep)
                    TestStopflag,keep_value = PermutationTest_last(datas['permuTestPost'],listLen = 3,expandNum=5,permuNum=10000)
                    if TestStopflag:
                        part_flag[randnum] = False
                        with open(f"{settings['outdir']}/permutationTestPost.txt", 'a') as f:
                            f.write(f"{datas['filename']}\t{keep_value}\n")
                        continue


                # 开始一些该loop 的后续处理,比如 寻找 candidate_loops_knum2， 根据candidate_loops_knum2 的结果判断是否要更新 blocks_class, (不管如何都要更新全部的groups_result,以防后续寻找)
                # 对于knum=3, 根据candidate_loops_knum2 的结果判断是写进 Final_Loops 还是写进 Final_Loops_3 还是 写进 Final_Loops_3_ambiguous

                new_block_group = []
                for i in range(len(s_t[maxindex])):
                    new_block_group.append((s_t[maxindex][i],e_t[maxindex][i]))
                print(f'p: {p}, new_block_group: {new_block_group},blocks_class: {blocks_class}')


                blocks_class_candicate,candidate_loops_knum2,candidate_loops_knum3 = UpdateNewClass(new_block_group, blocks_class)

                groups_result.append(new_block_group)
                if knum==3:
                    groups_result = groups_result+candidate_loops_knum3
                    candidateLen = len(candidate_loops_knum3)
                elif knum==2:
                    groups_result = groups_result+candidate_loops_knum2
                    candidateLen = len(candidate_loops_knum2)
                else:
                    pass

                with open(settings['outdir'] + f'/Groups_{p}_{knum}.txt','a') as outfile:
                    for group in groups_result:
                        outfile.write(str(group)+'\n')
                    outfile.write('\n')

                if write_flag == False:
                    blocks_class_candicate = blocks_class.copy()
                    candidate_loops_knum2,candidate_loops_knum3 = [],[]

                print('-------------------start verify candidate loops,  candidate loops num: ',len(candidate_loops_knum2),'----------------')

                blocks_class = recordCandidate(datas,settings,knum,groups_result,blocks_class,blocks_class_candicate,candidate_loops_knum2,p,maxindex,s_t,e_t,max_post_allrep,strongPvalueFlag,foundLinksNum,timeuse,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post)

                with open(settings['outdir'] + f'/Class_{p}_{knum}.txt','a') as outfile:
                    for group in blocks_class:
                        outfile.write(str(group)+'\n')
                    outfile.write('\n')


                # if (np.all(rhats < 1.1) < 1.1)|(all_similarFlag2 == 3): 
                    
                #     groups_result,blocks_class,candidate_loops_knum2,foundLinksNum = resultprocess(datas,settings,knum,file_path,p,maxindex,s_t[maxindex],e_t[maxindex],max_post_allrep,foundLinksNum,groups_result,blocks_class,timeuse,write_flag)
                #     savetxt(knum,mcmcparas,settings['outdir'],p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,foundLinksNum)
    
                # else:
                #     groups_result,blocks_class,candidate_loops_knum2,foundLinksNum = resultprocess(datas,settings,knum,file_path,p,maxindex,s_t[maxindex],e_t[maxindex],max_post_allrep,foundLinksNum,groups_result,blocks_class,timeuse,write_flag)
                #     savetxt(knum,mcmcparas,settings['outdir'],p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,foundLinksNum)                    
                
                intervalcount = 1 + candidateLen
                discard_chains = datas['discard_chains'].copy()
                print('-------------------start verify discard loops,  discard loops num: ',len(discard_chains),'----------------')

                while len(discard_chains)>0:
                # for i in range(discardnum):
                    i = 0

                    discard_m_list,discard_l_list = discard_chains[i][0][:,-1],discard_chains[i][1][:,-1]
                    # print('discard_m_list,discard_l_list: ',discard_m_list,discard_l_list)  
                    part_groups_result = groups_result[-intervalcount:].copy()
                    part_groups_result.append(list_to_group(discard_m_list,discard_l_list))

                    overlap_flag = check_result_overlap(part_groups_result,extendFlag=True)
                    if overlap_flag:
                        print('discard loop overlap with current groups, decide to discard this loop!')
                        del discard_chains[i]
                        continue
                    else:
                        print('discard loop do not overlap with current groups, decide to test this loop!')
                        print('discard_m_list,discard_l_list: ',discard_m_list,discard_l_list)  
                    
                    startDiscard_time = time.time()
                    m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post,s_t,e_t = testDiscard(datas,groups_result,discard_chains[i],knum,p,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post)
                    endDiscard_time = time.time()
                    timeuse = endDiscard_time - startDiscard_time    

                    X_t = np.zeros((rep,pairnum+1))
                    for t in range(rep):
                        block_group= list_to_group(max_m_list[p+1][t],max_l_list[p+1][t])
                        X_t[t] = countX_list(datas,knum,block_group)
                        print('Discard test Eachbest: ', 'candidate search: ',p+1, ', rep: ', t,', m: ',max_m_list[p+1][t],', l: ',max_l_list[p+1][t],', p1: ',['%.3e' % x for x in max_p_list[p+1][t][:-1]],', p234: ','%.3e'%max_p_list[p+1][t][-1],', post: ','%.3f'%max_post[p+1][t],', Case1: ',X_t[t][:-1])
                        foundLinksNum[p+1][t] = X_t[t][:-1]

                    if (all_similar(lst=max_post[p+1],err=2))&(np.all(X_t[:,:-1] >= settings['minLinks'])):
                        print('decide to keep this loop!')

                        max_post_allrep = max(max_post[p+1])
                        maxindex = list(max_post[p+1]).index(max_post_allrep)

                        pvalues_maxindex = []; pvalue_pair_index=0
                        for i in range(knum-1):
                            for j in range(i+1,knum):
                                m1,l1,m2,l2 = max_m_list[p+1,maxindex,i],max_l_list[p+1,maxindex,i],max_m_list[p+1,maxindex,j],max_l_list[p+1,maxindex,j]
                                readsnum1,readsnum2 = GetReadsNum(datas,settings,m1,l1,m2,l2,shrinkFlag=False,expandFlag=True)
                                IAB = X_t[maxindex][pvalue_pair_index]
                                pvalue_pair_index += 1
                                pvalues_maxindex.append(getPvalue_hypergeom(len(datas['data_bedpe']),readsnum1,readsnum2,IAB))

                        strongPvalueFlag = np.all(np.array(pvalues_maxindex) < 1e-4)

                        write_flag = True
                        if knum >= 3:
                            X_maxindex = X_t[maxindex][:-1]
                            if np.count_nonzero(np.array(X_maxindex)<=2)>=2:
                                write_flag = False
                                weak_zero_flag += 1

                            print(f'========== weak_zero_flag for this accepted discard loop: {weak_zero_flag} ==========')

                        new_block_group = []
                        for i in range(len(s_t[maxindex])):
                            new_block_group.append((s_t[maxindex][i],e_t[maxindex][i]))
                        print(f'p+1: {p+1}, new_block_group: {new_block_group},blocks_class: {blocks_class}')

                        blocks_class_candicate,candidate_loops_knum2,candidate_loops_knum3 = UpdateNewClass(new_block_group, blocks_class)

                        groups_result.append(new_block_group)
                        if knum==3:
                            groups_result = groups_result+candidate_loops_knum3
                            candidateLen = len(candidate_loops_knum3)
                        elif knum==2:
                            groups_result = groups_result+candidate_loops_knum2
                            candidateLen = len(candidate_loops_knum2)
                        else:
                            pass

                        with open(settings['outdir'] + f'/Groups_{p+1}_{knum}.txt','a') as outfile:
                            for group in groups_result:
                                outfile.write(str(group)+'\n')
                            outfile.write('\n')

                        if write_flag == False:
                            blocks_class_candicate = blocks_class.copy()
                            candidate_loops_knum2,candidate_loops_knum3 = [],[]

                        print('-------------------start verify candidate loops for accepted discard loops,  candidate loops num: ',len(candidate_loops_knum2),'----------------')

                        blocks_class = recordCandidate(datas,settings,knum,groups_result,blocks_class, blocks_class_candicate,candidate_loops_knum2,p+1,maxindex,s_t,e_t,max_post_allrep,strongPvalueFlag,foundLinksNum,timeuse,m_list,l_list,p_list,posterior,max_m_list,max_l_list,max_p_list,max_post)

                        with open(settings['outdir'] + f'/Class_{p+1}_{knum}.txt','a') as outfile:
                            for group in blocks_class:
                                outfile.write(str(group)+'\n')
                            outfile.write('\n')

                        p = p+1
                        intervalcount = intervalcount + 1
                        intervalcount = intervalcount + candidateLen
                        del discard_chains[i]

                    else:
                        print('reject! Decide to discard this loop!')
                        m_list[p+1] = np.nan*np.zeros((rep,knum,subiters*iters+1))
                        l_list[p+1] = np.nan*np.zeros((rep,knum,subiters*iters+1))
                        p_list[p+1] = np.nan*np.zeros((rep,pairnum+1,subiters*iters+1))
                        posterior[p+1] = np.full((rep,subiters*iters+1), np.NINF)

                        max_m_list[p+1] = np.nan*np.zeros((rep,knum))
                        max_l_list[p+1] = np.nan*np.zeros((rep,knum))
                        max_p_list[p+1] = np.nan*np.zeros((rep,pairnum+1))
                        max_post[p+1] = np.full(rep, np.NINF)

                        del discard_chains[i]
                    
                datas['discard_chains'] = discard_chains.copy()
 
                print('--------------------------------------------continue finding, Loops=',p+1,'---------------------------------------------')
            else:
                part_flag[randnum] = False
                # break
                print('--------------------------------------------No initial point for this dataset can be found!---------------------------------------------')
            NumLoops = p
            print('stopFlag=',stopFlag)
            p = p+1

        globals()[f'groups_result_{knum_list[knum_idex]}'] = copy.deepcopy(groups_result)

        if knum==3 and knum_list[knum_idex+1]==2:
            groups_result_2 = []
            for i in range(len(groups_result)):
                combinations_2_class = list(itertools.combinations(groups_result[i], 2))
                combinations_2_class = [list(pair) for pair in combinations_2_class]
                groups_result_2 = groups_result_2 + combinations_2_class
            groups_result_2 = list(map(lambda x: [list(i) for i in x], set(map(lambda x: tuple(map(tuple, x)), groups_result_2))))
            groups_result_2.sort(key=lambda x: x[0][0])
            groups_result = merge_overlapping_groups(groups_result_2)
        else:
            print('All knum done!')


        # save results
            
if __name__ == '__main__':
    print('----------------------------------------start MCMC---------------------------------------------')
    start = time.time()
    MultiprocessP()
    end = time.time()
    print('----------------------------------Total Time used: ',end-start,'s ----------------------------------')
    with open(settings['outdir']+'/worklog.txt', 'a') as f:
        f.write('----------------------------------Total Time used: '+ str(end-start)+'s ----------------------------------' + '\n')