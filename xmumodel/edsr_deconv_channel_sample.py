# -*- coding: utf-8 -*-

from xmumodel.model import Model
import tensorlayer.layers as tl
import tensorflow as tf
from xmuutil import utils
from xmuutil.scalelayer import ScaleLayer
from xmuutil.relulayer import ReluLayer
from xmuutil.transposed_conv2d_layer import TransposedConv2dLayer
from tqdm import tqdm
import shutil
import os
import numpy as np
"""
EDSR upsample-layer
subpixel -> deconv 
"""

lasso_alpha=1e-3

class EDSR_P(Model):
    def buildModel(self):
        print("Building EDSR...")
        self.norm_input = utils.normalize_color_tf(self.input)
        self.norm_target = utils.normalize_color_tf(self.target)

        #input layer
        x = tl.InputLayer(self.norm_input, name='inputlayer')

        # One convolution before res blocks and to convert to required feature depth
        #第一层的剪枝将会作用在这里，filter数量减少
        #x = tl.Conv2d(x, self.feature_size-self.prunedlist[0], [3, 3], name='c')
        x = tl.Conv2d(x, self.feature_size, [3, 3], name='c')

        # Store the output of the first convolution to add later
        conv_1 = x

        scaling_factor = 0.1
        # Add the residual blocks to the model
        for i in range(self.num_layers):
            x = self.__resBlock(x, self.feature_size, scale=scaling_factor,layer=i)
        # One more convolution, and then we add the output of our first conv layer
        x = tl.Conv2d(x, self.feature_size, [3, 3], act = None, name = 'm1')
        x = tl.ElementwiseLayer([conv_1,x],tf.add, name='res_add')

        x = TransposedConv2dLayer(x, self.feature_size, [5,5], [2,2], name='deconv_1')
        x = tl.Conv2d(x, self.feature_size, [3, 3], act=tf.nn.relu, name='deconv_conv_1')
        x = TransposedConv2dLayer(x, self.feature_size, [5, 5], [2, 2], name='deconv_2')
        if self.scale==8:
            x = tl.Conv2d(x, self.feature_size, [3, 3], act=tf.nn.relu, name='deconv_conv_2')
            x = TransposedConv2dLayer(x, self.feature_size, [5, 5], [2, 2], name='deconv_3')

        # One final convolution on the upsampling output
        output = tl.Conv2d(x,self.output_channels,[1,1],act=tf.nn.relu, name='lastLayer')
        # output = tl.Conv2d(x, self.output_channels, [1, 1], act=None, name='lastLayer')
        self.output = output.outputs
        self.output = tf.clip_by_value(output.outputs, 0.0, 1.0)

        self.cacuLoss()

        # Tensorflow graph setup... session, saver, etc.
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        session_conf.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=session_conf)
        self.saver = tf.train.Saver(var_list = tf.trainable_variables(), max_to_keep=100)
        print("Done building!")

    def __resBlock(self, x, channels = 64, kernel_size = (3, 3), scale = 1.0,layer = 0):
        nn = ReluLayer(x, name='res%d/ru1'%(layer))
        nn = tl.Conv2d(nn, channels-self.prunedlist[layer], kernel_size, act=tf.nn.relu, name='res%d/c1'%(layer))
        nn = tl.Conv2d(nn, channels, kernel_size, act=None, name='res%d/c2'%(layer))
        nn = ScaleLayer(nn,scale, name='res%d/scale'%(layer))
        n = tl.ElementwiseLayer([x,nn],tf.add, name='res%d/res_add'%(layer))
        #这里调用dictionary(X, W2, Y)函数，注意X就是x,即本层的输入，
        #W2仅仅是此block里第一层卷积的权值，Y=n-Y'这里的Y'是上一层剪枝后的输出 
        #即在残差模块中不断调整，弥补shortcut中的无法影响的部分
        
        #把本层的beta记录起来（append到model的beta数组里），减掉的通道数也要记录起来，影响新模型上一层的filter数量
        
        
        #所以上面是算完旧的，这里还要再算一次新的
        return n

    def cacuLoss(self):
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.norm_target, self.output))
        PSNR = utils.psnr_tf(self.norm_target, self.output, is_norm=True)
        
        

        # Scalar to keep track for loss
        summary_loss = tf.summary.scalar("loss", self.loss)
        summary_psnr = tf.summary.scalar("PSNR", PSNR)

        streaming_loss, self.streaming_loss_update = tf.contrib.metrics.streaming_mean(self.loss)
        streaming_loss_scalar = tf.summary.scalar('loss',streaming_loss)

        streaming_psnr, self.streaming_psnr_update = tf.contrib.metrics.streaming_mean(PSNR)
        streaming_psnr_scalar = tf.summary.scalar('PSNR',streaming_psnr)

        self.train_merge = tf.summary.merge([summary_loss,summary_psnr])
        self.test_merge = tf.summary.merge([streaming_loss_scalar,streaming_psnr_scalar])

    def train(self,batch_size= 10, iterations=1000, lr_init=1e-4, lr_decay=0.5, decay_every=2e5,
              save_dir="saved_models",reuse=False,reuse_dir=None,reuse_step=None, log_dir="log"):
        
        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        # Using adam optimizer as mentioned in the paper
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_v)
        # optimizer = tf.train.RMSPropOptimizer(lr_v, decay=0.95, momentum=0.9, epsilon=1e-8)
        # This is the train operation for our objective
        self.train_op = optimizer.compute_gradients(self.loss)
        #self.train_op = optimizer.apply_gradients(gradient)
	#self.train_op = optimizer.minimize(self.loss)


        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir, global_step=reuse_step)

            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input:test_x,
                            self.target:test_y
                    })
                else:
                    break

            sess.run(tf.assign(lr_v, lr_init))
            #This is our training loop
            grad_sum = np.zeros(shape=(16,3,3,self.feature_size, self.feature_size-self.prunedlist[0]))
            aim = np.zeros(shape=(16,self.feature_size-self.prunedlist[0]))
            candidate = []
            for i in tqdm(range(iterations)):
                if i != 0 and (i % decay_every == 0):
                    new_lr_decay = lr_decay ** (i // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,gradients = sess.run([self.train_merge,self.train_op],feed)
                for index,element in enumerate(gradients):
                    real_index = int((index-2)/4)
                    
                    if(index>=2 and index%4==2 and index<63):
                        grad_sum[real_index] = grad_sum[real_index]+np.fabs(np.array(element[0])*np.array(element[1]))
                

                #Write train summary for this step
                train_writer.add_summary(summary,i)

                # Test every 500 iterations; Save our trained model
                if (i!=0 and i % 500 == 0) or (i+1 == iterations):
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        # sess.run([self.streaming_loss_update],feed_dict=test_feed[j])
                        sess.run([self.streaming_loss_update, self.streaming_psnr_update], feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    # Write test summary
                    test_writer.add_summary(streaming_summ, i)

                    #self.save(save_dir, i)
            for index in range(16):
                aim[index] = np.sum(np.sum(np.sum(grad_sum[index],0),0),0)
                candidate.append(aim[index].argsort()[0:self.prunesize])
                candidate[index] = sorted(candidate[index])
            print("start_pruning")
            utils.prune_network(self, sess, candidate)
            """
            for index,element in enumerate(candidate):
                var_tensor = tl.get_variables_with_name("res"+str(index)+"/"+"c1")
                w = var_tensor[0]
                b = var_tensor[1]
                w_next = tl.get_variables_with_name("res"+str(index)+"/"+"c2")[0]
                
                w_np = w.eval()
                b_np = b.eval()
                w_next_np = w_next.eval()
                
                w_np = np.delete(w_np, element, -1)
                b_np = np.delete(b_np, element, 0)
                w_next_np = np.delete(w_next_np, element, -2)

                w_new = tf.convert_to_tensor(w_np)
                #print(w_new.shape)
                b_new = tf.convert_to_tensor(b_np)
                #print(b_new.shape)
                w_next_new = tf.convert_to_tensor(w_next_np)
                #print(w_next_new.shape)
                
                sess.run(tf.assign(w, w_new, False))
                sess.run(tf.assign(b, b_new, False))
                sess.run(tf.assign(w_next, w_next_new, False))
            #tl.print_all_variables() 
            """
            self.save(save_dir, i)
            save_prunedlist = []
            for i,e in enumerate(self.prunedlist):
                save_prunedlist.append(int(e+len(candidate[i])))

            np.savetxt(save_dir+"/prunedlist",np.array(save_prunedlist),fmt="%d",delimiter=",")
            print("end_pruning")   
            
            test_writer.close()
            train_writer.close()



    #待完成，目标是返回本层的beta即可
    def dictionary(X, W2, Y,alpha=1e-4, rank=None, DEBUG=0, B2=None, rank_tol=.1, verbose=0):
        verbose=0
        if verbose:
            timer = Timer()
            timer.tic()
        #rank是本层需要留下的通道数，即c‘，rank_tol应该就是所有的通道数
        #if 0 and rank_tol != dcfgs.dic.rank_tol:
        #    print("rank_tol", dcfgs.dic.rank_tol)
        #rank_tol = dcfgs.dic.rank_tol
        # X: N c h w,  W2: n c h w
        norank=False  #False
        if norank:
           rank = None
        #TODO remove this
        N = X.shape[0]
        c = X.shape[1]   #输入fea map的维数
        h = X.shape[2]
        w=h
        n = W2.shape[0]
        # TODO I forgot this
        # TODO support grp lasso
    
        #暂时不支持组lasso
        grp_lasso = False
    
        if grp_lasso:#False
            reX = X.reshape((N, -1))
            ally = Y.reshape((N,-1))
            samples = np.random.choice(N, N//10, replace=False)
            Z = reX[samples].copy()
            reY = ally[samples].copy()
    
        else:
            samples = np.random.randint(0,N, min(400, N//20))
            #samples = np.random.randint(0,N, min(400, N//20))
            # c N hw
            reX = np.rollaxis(X.reshape((N, c, -1))[samples], 1, 0)
            #c hw n
            reW2 = np.transpose(W2.reshape((n, c, -1)), [1,2,0])
            if 0:
                W2_std = np.linalg.norm(reW2.reshape(c, -1), axis=1)
            # c Nn
            Z = np.matmul(reX, reW2).reshape((c, -1)).T
            #'''lasso函数是给定Y和X，求解最优的W，本文是求解reY和X*W之间的最优beta。所以这里将X*W得到Z'''
    
            # Nn
            reY = Y[samples].reshape(-1)
        
        #'''alpha是L1损失的系数'''
        _solver = Lasso(alpha=alpha, warm_start=True,selection='random' )
        
    
    
        def solve(alpha):
            
            _solver.alpha=alpha
            _solver.fit(Z, reY) 
            #'''z已经是X*W, 所以要拟合的参数是beta'''
    
            #_solver.fit(Z, reY)
            if grp_lasso and 0:
                idxs = _solver.coef_[0] != 0.
            else:     
                idxs = _solver.coef_ != 0. #'''idxs是Lasso之后beta不为0对应的index'''
            tmp = sum(idxs) #'''idxs是Lasso之后beta不为0的索引，tmp是个数'''
            return idxs, tmp
    
        '''
        def updateW2(idxs):
            nonlocal Z
            tmp_r = sum(idxs)
            reW2, _ = fc_kernel((X[:,idxs, :,:]).reshape(N, tmp_r*h*w), Y)
            reW2 = reW2.T.reshape(tmp_r, h*w, n)
            nowstd=np.linalg.norm(reW2.reshape(tmp_r, -1), axis=1)
    
            reW2 = (W2_std[idxs] / nowstd)[:,np.newaxis,np.newaxis] * reW2
            newshape = list(reW2.shape)
            newshape[0] = c
            newreW2 = np.zeros(newshape, dtype=reW2.dtype)
            newreW2[idxs, ...] = reW2
            Z = np.matmul(reX, newreW2).reshape((c, -1)).T
            if 0:
                print(_solver.coef_)
            return reW2
        '''
    
        if rank == c:  #rank为留下来的通道数    c为实际的通道数
            idxs = np.array([True] * rank)
        elif not norank:  #norank为False   则true
            left=0
            right=lasso_alpha 
            lbound = rank# - rank_tol * c
            if rank_tol>=1:
                rbound = rank + rank_tol
            else:
                rbound = rank + rank_tol * rank
                #rbound = rank + rank_tol * c
                if rank_tol == .2:
                    print("TODO: remove this")
                    lbound = rank + 0.1 * rank
                    rbound = rank + 0.2 * rank
            while True: #一直lasso拟合，直到不为0的个数tmp满足rank限制
                _, tmp = solve(right) #tmp是Lasso之后beta不为0的个数
                if False:
                    if tmp > rank:
                        break
                    else:
                        right/=2
                        if verbose:print("relax right to",right)
                else:
                    if tmp < rank:#如果不为0的个数小于rank则break
                        break
                    else:#如果未达到rank，则继续solve，right是L1范数的系数，增大会使得模型更多为0
                        right*=2
                        if verbose:print("relax right to",right)
            while True:
                alpha = (left+right) / 2
                idxs, tmp = solve(alpha)
                if verbose:print(tmp, alpha, left, right)
                if tmp > rbound:
                    left=alpha
                elif tmp < lbound:
                    right=alpha
                else:
                    break
                
            if 0:
                if rbound == lbound:
                    rbound +=1
                orig_step = left/100 + 0.1 # right / 10
                step = orig_step
    
                def waitstable(a):
                    tmp = -1
                    cnt = 0
                    for i in range(10):
                        tmp_rank = tmp
                        idxs, tmp = solve(a)
                        if tmp == 0:
                            break
                        updateW2(idxs)
                        if tmp_rank == tmp:
                            cnt+=1
                        else:
                            cnt=0
                        if cnt == 2:
                            break
                        if 1: 
                            if verbose:print(tmp, "Z", Z.mean(), "c", _solver.coef_.mean())
                    return idxs, tmp
    
                previous_Z = Z.copy()
                state = 0
                statecnt = 0
                inc = 10
                while True:
                    Z = previous_Z.copy()
                    idxs, tmp = waitstable(alpha)
                    if tmp > rbound:
                        if state == 1:
                            state = 0
                            step/=2
                            statecnt=0
                        else:
                            statecnt+=1
                        if statecnt >=2:
                            step*=inc
                        alpha += step
                    elif tmp < lbound:
                        if state == 0:
                            state = 1
                            step /= 2
                            statecnt=0
                        else:
                            statecnt+=1
                        if statecnt >=2:
                            step*=inc
                        alpha -= step
                    else:
                        break
                    if verbose:print(tmp, alpha, 'step', step)
            rank=tmp
        else:
            print("start lasso kernel")
            idxs, rank = solve(alpha)
            print("end lasso kernel")
    
    
        if verbose:
            timer.toc(show='lasso')
            timer.tic()
        if grp_lasso:#false
            inW, inB = fc_kernel(reX[:, idxs], ally, copy_X=True)
            def preconv(a, b, res, org_res):
               
                #a: c c'
                #b: n c h w
                #res: c
           
                w = np.tensordot(a, b, [[0], [1]])
                r = np.tensordot(res, b, [[0], [1]]).sum((1,2)) + org_res
                return np.rollaxis(w, 1, 0), r
            newW2, newB2 = preconv(inW, W2, inB, B2)
        #elif dcfgs.ls == cfgs.solvers.lowparams: #线性
        elif 0:
            reg = LinearRegression(copy_X=True, n_jobs=-1)
            assert dcfgs.fc_ridge == 0
            assert dcfgs.dic.alter == 0, "Z changed"
            reg.fit(Z[:, idxs], reY)#对训练集X, y进行训练
            newW2 = reg.coef_[np.newaxis,:,np.newaxis,np.newaxis] * W2[:, idxs, :,:]
            newB2 = reg.intercept_
        elif 0:#dcfgs.nonlinear_fc
            newW2, newB2 = nonlinear_fc(X[:,idxs,...].reshape((N,-1)), Y)
            newW2 = newW2.reshape((n,rank, h, w))
        elif 0:#dcfgs.nofc
            newW2 = W2[:, idxs, :,:]
            newB2 = np.zeros(n)
        else:#执行这个
            #传入X中beta不为0对应的列，W也是不为0的列
            #对剪枝后的模型进行修正，得到新的w和b
            newW2, newB2 = fc_kernel(X[:,idxs,...].reshape((N,-1)), Y, W=W2[:, idxs,...].reshape(n,-1), B=B2)
            
            newW2 = newW2.reshape((n,rank, h, w))#将得到新的w展成四维卷积核
        if verbose:
            timer.toc(show='ls')
        if not norank:
            lasso_alpha = alpha
        if verbose:print(rank)
        if DEBUG:
            #print(np.where(idxs))
            newX = X[:, idxs, ...]#newX为不为0对应的列
            return newX, newW2, newB2
        else:
            return idxs, newW2, newB2
   
'''     
    def fc_kernel(X, Y, copy_X=True, W=None, B=None, ret_reg=False,fit_intercept=True):
        
        #return: n c
        
        assert copy_X == True
        assert len(X.shape) == 2
        if dcfgs.ls == cfgs.solvers.gd:  
            w = Worker()
            def wo():
                from .GDsolver import fc_GD
                a,b=fc_GD(X,Y, W, B, n_iters=1)
                return {'a':a, 'b':b}
            outputs = w.do(wo)
            return outputs['a'], outputs['b']
        elif dcfgs.ls == cfgs.solvers.tls:
            return tls(X,Y, debug=True)
        elif dcfgs.ls == cfgs.solvers.keras:
            _reg=keras_kernel()
            _reg.fit(X, Y, W, B)
            return _reg.coef_, _reg.intercept_
        elif dcfgs.ls == cfgs.solvers.lightning:
            #_reg = SGDRegressor(eta0=1e-8, intercept_decay=0, alpha=0, verbose=2)
            _reg = CDRegressor(n_jobs=-1,alpha=0, verbose=2)
            if 0:
                _reg.intercept_=B
                _reg.coef_=W
        elif dcfgs.fc_ridge > 0:  #0
            _reg = Ridge(alpha=dcfgs.fc_ridge)
        else:#执行这个   线性回归
            _reg = LinearRegression(n_jobs=-1 , copy_X=copy_X, fit_intercept=fit_intercept)
    
        _reg.fit(X, Y)#给定X 和Y，拟合最好的W 
    
        if ret_reg:
            return _reg
    
        return _reg.coef_, _reg.intercept_  #相当于w和b
    '''