# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:29:31 2018

@author: 谢志鹏
"""
import tensorflow  as tf
import cv2
import sys
sys.path.append('game/')
import wrapped_flappy_bird as game
import random 
import numpy as np
from collections import deque

#基本参数
GAME = 'bird'
ACTIONS = 2#两种行为 往上 往下
OBSERVE =  10000#观察结果--随机跑  通过观察10000帧图像 把效果好的拿出来
ECPLORE = 200000 #迭代次数
FINAL_EPSION = 0.0001   #网络不稳定是时 --多探索 ，稳定时-少探索网络模型
INITIAL_EPSILON = 0.1 # starting value of epsilon
INITIAL = 0.0001 #探索预设值


REPLAY_MEMORY = 50000 #进行储存观察的当前数据
BATCH = 32 #制定一个batch=32

def createNetwork():
    #三层卷积架构
    #权重值初始化
    W_conv1 = weights_variable([8,8,4,32])#8*8 4张图像 output 32
    b_conv1 = bias_variable([32])#32个特征图
    
    W_conv2 = weights_variable([4,4,32,64])#4*4 32张图像 output 64
    b_conv2 = bias_variable([64])#64个特征图
    
    W_conv3 = weights_variable([3,3,64,64])#3*3 64张图像 output 64
    b_conv3 = bias_variable([64])#64个特征图
    
    #全连接层
    W_fc1 = weights_variable([1600,512]) #h*w/c=80*80/4=1600 ,512维向量
    b_fc1 = bias_variable([512])  #输出512维向量
    
    W_fc2 = weights_variable([512,ACTIONS]) #得到两种操作
    b_fc2 = bias_variable([ACTIONS])  
    
    #当前输入   input
    s = tf.placeholder('float',[None,80,80,4])
    
    h_conv1 = tf.nn.relu(conv2d(s,W_conv1,4) + b_conv1) #卷积操作 4：stride=4
    #池化
    h_pool1 = max_pool_2x2(h_conv1)
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,2) + b_conv2) #卷积操作 4：stride=4
#    h_pool2 = max_pool_2x2(h_conv1)
    h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3) #卷积操作 4：stride=4
    
    #全连接层变换
    #向量拉平
    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])#rehape(特征图w*h/c=80*80/4)
    #输入最终结果
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
    
    readout = tf.matmul(h_fc1,W_fc2) + b_fc2#得到预测结果
    return s,readout,h_fc1

#权重操作
def weights_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)#高斯初始化 标准差=0.01
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.01,shape = shape)#常量初始化
    return tf.Variable(initial)
def conv2d(x,W,stride):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#h,w=2,2

def trainNetwork(s,readout,h_fc1,sess):
    
    a = tf.placeholder('float', [None,ACTIONS])#定义action
    y = tf.placeholder('float', [None]) #定义当前网络下一个状态结果
    
    readout_action = tf.reduce_mean(tf.multiply(readout,a),reduction_indices=1)#得到当前结果并计算平均值
    cost = tf.reduce_mean(tf.square(y - readout_action))#计算loss y:下一帧状态值  readout_action：当前状态值
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)#优化loss 学习率1e-6
    
    #游戏环境
    game_state = game.GameState()
    
    D = deque()#将游戏变化(观察值数据)保存下来
    
    #日志保存
    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')
    
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1#默认初始化定义为1 
    #将图像一帧一帧读出来
    
    x_t,r_0,terminal = game_state.frame_step(do_nothing)#跑一帧数据
    #对x_t图像数据进行转换
    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_BGR2GRAY)#将彩色图转换为gray
    #对图像进行二值化
    ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY) #>1 ==255  <1 ==0
    #将4帧图像叠加起来==80*80*4的结构
    s_t = np.stack((x_t,x_t,x_t,x_t),axis = 2)#x_t只有一帧图像，四个堆叠在一起,指定维度=2
    
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())#将所有变量进行初始化
    checkpoint = tf.train.get_checkpoint_state('saved networks')#
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess,checkpoint.model_checkpoint_path)
        print ('successfully loaded')
    else:
        print ('load failed')
        
    epsilon = INITIAL_EPSILON#探索和开发
    
    #依次进行迭代
    t=0
    while 'flappy bird' != 'angry bird':
        #时间t 从0开始
        readout_t = readout.eval(feed_dict = {s:[s_t]})[0]#往上飞，往下飞的得分
        a_t = np.zeros([ACTIONS])#action_step 记录每一步的动作
        action_index = 0
        
        
        if t % 1 ==0: #每一步都要进行random随机选择
            #进行贪心  探索
            if random.random() <= epsilon:
                print ('random Action')
#                action_index = random.randint(ACTIONS)#随机选一个方向（上下飞）
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:#找最大的得分进行开发
                action_index = np.argmax(readout_t)#得到比分最大值
                a_t[action_index] = 1 
        #保存最好的结果
        x_t1_colored,r_t,terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored,(80,80)),cv2.COLOR_BGR2GRAY)#将彩色图转换为gray
#        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        #对图像进行二值化
        ret,x_t1 = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)#得到二分值图
        x_t1 = np.reshape(x_t1,[80,80,1])   #新的图像
        #合并拼接图像
        s_t1 = np.append(x_t1,s_t[:,:,:3],axis = 2)#:,:,3：h全要,w全要,只取历史的前三个图像
#        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        #把每一个状态存储
        #每完成一个batch D结构append 当前状态(s_t),
        #当前执行action(a_t),当前执行奖励值(r_t)变成下一个结果状态(s_t1),terminal最终有没有结束
        D.append((s_t,a_t,r_t,s_t1,terminal))#把每个状态存好
#         D.append((s_t, a_t, r_t, s_t1, terminal))
        
        #指定D上限
        if len(D)>REPLAY_MEMORY:
            D.popleft()#把前面的数据除去
        
        if t > OBSERVE:#观察完结果之后  进行优化
            minibatch = random.sample(D,BATCH) #32个状态
            
            #拿出指标 
            s_j_batch = [d[0] for d in minibatch]#每一个batch将state取出来
            a_batch = [d[1] for d in minibatch]
            
            r_batch = [d[2] for d in minibatch]
            
            s_j1_batch = [d[3] for d in minibatch]
            
            #迭代训练
            y_batch= []
            
            #进行前向传播
            #batch结果取出来
            readout_j1_batch = readout.eval(feed_dict = {s:s_j1_batch})
            for i in range(0,len(minibatch)):
                #
                terminal = minibatch[i][4]#判断后阶段是否结束
                if terminal:#如果terminal已经结束了
                    y_batch.append(r_batch[i])
                
                else:#下一阶段未结束
                    #即时奖励+GAME（折扣系数）*下一个阶段带来最大的收益
                    y_batch.append(r_batch[i]+GAME*np.max(readout_j1_batch[i]))
            #执行优化
            train_step.run(feed_dict = {
                            y:y_batch,#计算收益
                            a:a_batch,#action
                            s:s_j_batch#输入s
                            })
            #进行状态更新
            s_t = s_t1
            s +=1
            if t % 10000 ==0:
                saver.save(sess,'./',global_step = t)#global_step制定保存模型第t次迭代的结果
            
            state = ''#当前状态
            if t <= OBSERVE:
                state = 'OBSERVE'
            else:
                state = 'train'
            #打印当前状态值
#            print(state)
            print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        
    
def playGame():
    #实例化session
    sess = tf.InteractiveSession()
    #定义神经网络结构
    s,readout,h_fc1 = createNetwork()
    trainNetwork(s,readout,h_fc1,sess)

def main():
    playGame()

if __name__ == '__main__':
    main()








